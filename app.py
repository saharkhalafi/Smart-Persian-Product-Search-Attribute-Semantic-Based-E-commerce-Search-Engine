import os
import re
import pickle
from typing import List, Dict, Set
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Config  
PRODUCTS_FILE = os.getenv("PRODUCTS_FILE", "productsf.xlsx")
SYNONYMS_FILE = os.getenv("SYNONYMS_FILE", "synonym.xlsx")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

FAISS_INDEX_FILE = os.path.join(ARTIFACTS_DIR, "faiss.index")
ROWID_MAP_FILE = os.path.join(ARTIFACTS_DIR, "row_ids.pkl")
TFIDF_FILE = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")
CATEGORY_MODEL_FILE = os.path.join(ARTIFACTS_DIR, "category_model.pkl")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Weights for scoring
W_SIM = 1.0
W_GENDER = 0.35
W_SUBCAT = 0.35
W_MATERIAL = 0.35
W_SIZE = 0.25
W_TITLE_TOKEN = 0.06
W_CANONICAL = 0.12
W_LEXICAL = 0.25

# Globals 
df: pd.DataFrame = None
dense = None
syn_map: Dict[str, str] = {}
categories: Set[str] = set()
genders: Set[str] = set()
sizes: Set[str] = set()
kw_scorer = None
category_model = None

# FastAPI App
app = FastAPI(title="Smart Persian Product Search", docs_url="/docs")

# Utilities 
AR_TO_FA_MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی"})

def normalize(text: str) -> str:
    if text is None: return ""
    text = str(text).translate(AR_TO_FA_MAP)
    text = re.sub(r"[‌\u200c]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

MATERIAL_GROUPS = {
    "نخی": {"نخی", "پنبه", "الیاف طبیعی"},
    "پلی‌استر": {"پلی‌استر", "پلی استر", "الیاف مصنوعی"},
    "چرم": {"چرم", "چرم مصنوعی"},
    "حریر": {"حریر", "ابریشم مصنوعی"},
    "لینن": {"لینن", "لنین"},
    "ساتن": {"ساتن"}
}
MATERIAL_LOOKUP = {normalize(i): g for g, items in MATERIAL_GROUPS.items() for i in items}

def load_synonyms(path: str) -> Dict[str, str]:
    if not os.path.exists(path): return {}
    df_syn = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    if df_syn.empty: return {}
    main_col = df_syn.columns[0]
    mapping = {}
    for _, row in df_syn.iterrows():
        main = normalize(str(row[main_col]))
        for c in df_syn.columns:
            if c == main_col: continue
            val = row.get(c)
            if pd.notna(val):
                alt = normalize(val)
                if alt and alt != main:
                    mapping[alt] = main
    return mapping

def load_products(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Products file not found: {path}")
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = ["ProductName","ProductSubCategory","BrandName","Gender","Material","Size","Description","URL","Color"]
    for c in required:
        if c not in df.columns: df[c] = ""
    df.fillna("", inplace=True)
    for c in required: df[c] = df[c].astype(str)

    df["normalized_gender"] = df["Gender"].map(normalize)
    df["normalized_size"] = df["Size"].map(normalize)
    df["normalized_subcategory"] = df["ProductSubCategory"].map(normalize)
    df["material_group"] = df["Material"].map(lambda x: MATERIAL_LOOKUP.get(normalize(x), ""))
    df["Color_norm"] = df["Color"].map(lambda x: normalize(str(x)))
    df["combined"] = (
        df["ProductName"] + " " + df["ProductSubCategory"] + " " + df["BrandName"] +
        " " + df["Gender"] + " " + df["Material"] + " " + df["Size"] +
        " " + df["Description"] + " " + df["Color"]
    ).map(normalize).map(lambda x: " ".join(x.split()[:512]))
    df["title_norm"] = df["ProductName"].map(normalize)
    return df

# Keyword Scorer
class KeywordScorer:
    def __init__(self):
        self.vectorizer = None
        self.idf_map = {}
        self.analyzer = None
    def fit(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        self.vectorizer.fit(texts)
        vocab = self.vectorizer.vocabulary_
        idf = self.vectorizer.idf_
        self.idf_map = {term: float(idf[i]) for term,i in vocab.items()}
        self.analyzer = self.vectorizer.build_analyzer()
    def save(self, path): 
        with open(path,"wb") as f: pickle.dump({"idf_map": self.idf_map},f)
    def load(self, path):
        if not os.path.exists(path): return False
        with open(path,"rb") as f: data = pickle.load(f)
        self.idf_map = data.get("idf_map",{})
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.analyzer = self.vectorizer.build_analyzer()
        return True
    def query_keywords(self, query, top_n=6):
        if not self.analyzer: return []
        terms = self.analyzer(normalize(query))
        scored = [(t,self.idf_map.get(t,0)) for t in set(terms)]
        return sorted(scored,key=lambda x:x[1],reverse=True)[:top_n]
    def lexical_presence_score(self, text: str, keywords: List):
        score = sum(w for t,w in keywords if t in text)
        total = sum(w for _,w in keywords)+1e-8
        return score/total

# Dense Index
class DenseIndex:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.row_ids = []
    def encode(self, texts: List[str]):
        import torch
        texts = [" ".join(t.split()[:512]) for t in texts]
        emb = self.model.encode(texts, batch_size=8, show_progress_bar=False, device="cuda" if torch.cuda.is_available() else "cpu")
        emb = np.array(emb, dtype="float32")
        faiss.normalize_L2(emb)
        return emb
    def build(self, df: pd.DataFrame):
        emb = self.encode(df["combined"].tolist())
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.row_ids = list(df.index)
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        with open(ROWID_MAP_FILE,"wb") as f: pickle.dump({"row_ids": self.row_ids},f)
    def try_load(self):
        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(ROWID_MAP_FILE): return False
        self.index = faiss.read_index(FAISS_INDEX_FILE)
        with open(ROWID_MAP_FILE,"rb") as f: self.row_ids = pickle.load(f)["row_ids"]
        return True

# Category Prediction
def train_category_model(df):
    X = df["combined"].tolist()
    y = df["normalized_subcategory"].tolist()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])
    pipeline.fit(X, y)
    with open(CATEGORY_MODEL_FILE,"wb") as f: pickle.dump(pipeline,f)
    return pipeline

def load_category_model():
    if not os.path.exists(CATEGORY_MODEL_FILE): return None
    with open(CATEGORY_MODEL_FILE,"rb") as f: return pickle.load(f)

def predict_category(query, model, top_n=2):
    if model is None: return []
    probs = model.predict_proba([normalize(query)])[0]
    classes = model.classes_
    ranked = sorted(zip(classes,probs), key=lambda x:x[1], reverse=True)
    return [c for c,p in ranked[:top_n] if p>0.05]

# Search Helpers
def canonicalize_query(text: str, syn_map: Dict[str,str]) -> str:
    return " ".join(syn_map.get(w,w) for w in normalize(text).split())

def extract_canonical_tokens(text: str, syn_map: Dict[str,str]) -> Set[str]:
    return {syn_map.get(w,w) for w in normalize(text).split() if w}

def extract_query_attributes(query: str):
    q_norm = normalize(query)
    feats = {"gender": "", "size": "", "material": "", "color": ""}
    for g in genders:
        if g in q_norm: feats["gender"] = g; break
    for s in sizes:
        if s in q_norm: feats["size"] = s; break
    for w in q_norm.split():
        if w in MATERIAL_LOOKUP: feats["material"] = MATERIAL_LOOKUP[w]; break
    for w in q_norm.split():
        if w in df["Color_norm"].unique(): feats["color"] = w; break
    return feats

def compute_final_score(row, query_feats, canon_tokens, keywords, semantic_score):
    score = semantic_score
    if query_feats["gender"] and query_feats["gender"] == row["normalized_gender"]: score += W_GENDER
    if query_feats["size"] and query_feats["size"] == row["normalized_size"]: score += W_SIZE
    if query_feats["material"] and query_feats["material"] == row["material_group"]: score += W_MATERIAL
    if query_feats["color"] and query_feats["color"] in row["Color_norm"]: score += W_SIM
    if any(t in row["title_norm"] for t in canon_tokens): score += W_CANONICAL
    score += W_LEXICAL * kw_scorer.lexical_presence_score(row["combined"], keywords)
    return score

# API Models
class Product(BaseModel):
    productname: str
    url: str

class SearchResponse(BaseModel):
    message: str
    results: List[Product]

# Startup
print("[INFO] Initializing system...")
df = load_products(PRODUCTS_FILE)
categories = set(df["normalized_subcategory"].unique())
genders = set(df["normalized_gender"].unique())
sizes = set(df["normalized_size"].unique())
syn_map = load_synonyms(SYNONYMS_FILE)

kw_scorer = KeywordScorer()
if not kw_scorer.load(TFIDF_FILE):
    kw_scorer.fit(df["combined"].tolist())
    kw_scorer.save(TFIDF_FILE)

dense = DenseIndex(MODEL_NAME)
if not dense.try_load(): dense.build(df)

category_model = load_category_model()
if category_model is None: category_model = train_category_model(df)

print("[INFO] Initialization complete.")

# Endpoints
@app.get("/")
def home():
    return {"message": "🔍 Smart Persian Product Search is running."}

@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(...), top_k: int = Query(5, ge=1, le=50)):
    if df is None: raise HTTPException(status_code=500, detail="Data not loaded")
    norm_q = normalize(query)
    canon_q = canonicalize_query(norm_q, syn_map)
    canon_tokens = extract_canonical_tokens(norm_q, syn_map)
    keywords = kw_scorer.query_keywords(canon_q)
    query_feats = extract_query_attributes(query)

    # Candidate filtering by predicted category
    cat_pred = predict_category(norm_q, category_model)
    candidate_idx = df[df["normalized_subcategory"].isin(cat_pred)].index.tolist() if cat_pred else df.index.tolist()
    if not candidate_idx:
        return SearchResponse(message=f"نتیجه‌ای برای «{query}» یافت نشد", results=[])

    candidates = df.loc[candidate_idx]
    candidate_emb = dense.encode(candidates["combined"].tolist())
    query_emb = dense.encode([canon_q])[0]
    faiss.normalize_L2(candidate_emb)
    faiss.normalize_L2(query_emb.reshape(1,-1))
    sim_scores = candidate_emb @ query_emb

    results = [(compute_final_score(df.loc[idx], query_feats, canon_tokens, keywords, sim_scores[i]), idx)
               for i, idx in enumerate(candidate_idx)]
    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:top_k]

    return SearchResponse(
        message=f"نتایج برای «{query}» یافت شد",
        results=[Product(productname=df.loc[i]["ProductName"], url=df.loc[i]["URL"]) for _, i in top]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
