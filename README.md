# Smart Persian Product Search: Attribute & Semantic-Based E-commerce Search Engine

## Overview

This project provides a **Smart Persian Product Search** system designed for e-commerce platforms. It enables users to find products accurately and quickly, even with ambiguous, incomplete, or imprecise queries, by combining:

- **Semantic search** using sentence embeddings (Sentence Transformers)
- **Attribute-based matching** (gender, size, color, material, subcategory)
- **Lexical keyword scoring** to boost relevance
- **Synonym mapping** for Persian-specific variations

This system is optimized for Persian online shops, where product naming and attributes often vary, and users struggle to find the right item efficiently.

---

## Problem Statement

E-commerce platforms in Persian face the following challenges:

- Users struggle to find products with different naming conventions.  
  Example: `"تاپ زنونه قرمز"` vs `"تاپ قرمز زنانه"`
- Product attributes like **size, color, material, and gender** are inconsistently entered.
- Traditional search engines fail to combine **semantic meaning with structured attributes**, reducing user satisfaction.
- Customers often abandon searches when results are irrelevant, affecting sales and engagement.

This project addresses these problems by combining **semantic embeddings**, **attribute matching**, and **keyword boosting**.

---

## Features

- Semantic similarity search using **Sentence Transformers** (`paraphrase-multilingual-MiniLM-L12-v2`)
- Attribute extraction: **Gender, Size, Material, Color, Subcategory**
- Synonym mapping for Persian terms
- TF-IDF-based keyword scoring
- FAISS-based efficient similarity search
- Optional category prediction to filter irrelevant products
- FastAPI REST API for integration with e-commerce platforms

---
## Demo
![Screenshot](https://github.com/saharkhalafi/Smart-Persian-Product-Search-Attribute-Semantic-Based-E-commerce-Search-Engine/blob/main/eg%20Result.png) 

## Architecture / Method

### Data Loading

- Product data from `.xlsx` or `.csv`
- Synonyms mapping for Persian words

### Normalization

- Normalize Persian letters (`ك → ک`, `ي → ی`)
- Remove zero-width characters and extra spaces

### Attribute Extraction

- Extract **gender, size, color, material, subcategory** from query and product metadata

### Keyword Scoring

- Compute **TF-IDF scores** of query words
- Lexical presence boosts final ranking

### Semantic Search

- Encode product descriptions with **Sentence Transformers**
- Use **FAISS** for fast similarity search

### Final Ranking

- Combine **semantic similarity** + **attribute boosts** + **lexical score**
- Return **top-K results** to the user

---






