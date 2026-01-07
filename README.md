# 📘 Backstory Consistency Verification – Track A

This repository contains a solution to verify whether a given character backstory claim is **consistent** or **inconsistent** with its source novel 📖.

The project is developed for **Track A** and **mandatorily uses the Pathway Python Framework** for data ingestion, as required by the hackathon rules ✅.

---

## 🎯 Problem Description

Given:
- A **Story ID**
- A **Character name**
- A **Backstory claim**
- The corresponding **Novel text**

The task is to determine whether the claim aligns with the information present in the novel.

### Output Labels
- `1` → Consistent  
- `0` → Inconsistent  

---

## 🧠 Approach

The solution follows an **evidence-based retrieval pipeline**:

1. The novel text is divided into fixed-size chunks ✂️  
2. These chunks are ingested using the **Pathway framework**  
3. Semantic embeddings are generated for:
   - Character-aware backstory claims  
   - Novel text chunks  
4. Cosine similarity is computed between claims and novel chunks  
5. The most relevant evidence chunk is selected 🔍  
6. A conservative similarity threshold is applied to determine consistency  

This ensures predictions are grounded in **explicit textual evidence**, not guesses.

---

## 🛠️ Technologies Used

```text
🐍 Python
🧩 Pathway Python Framework
🧠 SentenceTransformers (all-MiniLM-L6-v2)
🔥 PyTorch (CUDA / MPS / CPU auto-detection)
📊 NumPy
⏳ tqdm

📂 Project Structure
.
├── final.py
├── train.csv
├── test.csv
├── In search of the castaways.txt
├── The Count of Monte Cristo.txt
├── results.csv
└── README.md

▶️ How to Run
Install Dependencies 📦
pip install pathway sentence-transformers torch tqdm pandas numpy
