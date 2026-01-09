
   <summary><b>The Minimal Version Best for Hack-a-Thon (Choice - A)</b></summary>

# Kharagpur Data Science Hackathon (KDSH) – Track A

> **Task:** Verify whether a character backstory claim is logically and causally consistent with long-form literary narratives.

This repository presents a retrieval-based semantic consistency solution for **Track A of the Kharagpur Data Science Hackathon (KDSH)**, addressing global consistency challenges in long novels such as *The Count of Monte Cristo* and *In Search of the Castaways*.

---

## 🎯 Problem Statement

Large Language Models often fail to maintain global narrative consistency over long texts.  
This challenge reframes narrative understanding as a **binary classification problem**.

### Input
- Character name  
- Backstory / claim  
- Source novel  

### Output
- `1` → Consistent  
- `0` → Inconsistent  

**Key Challenge:** Relevant evidence is sparsely distributed across long documents, requiring effective retrieval rather than full-context generation.

---

## 🧠 Methodology

We adopt an evidence-grounded retrieval pipeline:

1. **Text Chunking**  
   Novels are split into overlapping chunks (~800 characters) to ensure dense, localized context.

2. **Data Ingestion (Pathway – Mandatory)**  
   Structured ingestion of text chunks and claims using the Pathway framework.

3. **Semantic Embeddings**  
   - Model: all-MiniLM-L6-v2 (384-dimensional)  
   - Claim embedding = Character + Backstory  
   - Corpus = novel text chunks  

4. **Similarity-Based Reasoning**  
   Cosine similarity is computed between claims and all chunks, selecting the best-matching evidence.

5. **Decision Rule (Validated Threshold)**  
   - Similarity > 0.45 → Consistent (1)  
   - Similarity ≤ 0.45 → Inconsistent (0)

---

## 🧩 Why This Works

- Scales to long narratives  
- Avoids LLM context-window limitations  
- Evidence-driven and interpretable  
- Computationally efficient  
- Fully compliant with hackathon constraints  

---

## 🛠️ Tech Stack

- Python  
- Pathway Framework  
- SentenceTransformers (all-MiniLM-L6-v2)  
- PyTorch  
- NumPy  
- Pandas  
- tqdm  

---

<details>
  <summary style="list-style: none; cursor:pointer; display: inline;">
     <b>
        <h2 style="display: inline;">
           📂 Repository Structure(Click to view)
        </h2>
     </b>
  </summary> 
&emsp;&nbsp;•<br>
&emsp;<b>└── README.md</b><br>
&emsp;<b>└── Code </b><br>
&emsp;&emsp;&emsp;&emsp;├── final.py<br>
&emsp;&emsp;&emsp;&emsp;├── train.csv<br>
&emsp;&emsp;&emsp;&emsp;├── test.csv<br>
&emsp;&emsp;&emsp;&emsp;├── In search of the castaways.txt<br>
&emsp;&emsp;&emsp;&emsp;├── The Count of Monte Cristo.txt  <br>
&emsp;&emsp;&emsp;&emsp;└── results.csv  <br>

</details>
---

## ▶️ Quick Start

Install dependencies  
```bash
pip install -r requirements.txt
```

Run the solution  
```bash
python final.py
```

**Output:**  
Generates `results.csv` containing binary consistency predictions.

---

## 🧪 Core Insight

Consistent claims exhibit strong semantic alignment with at least one specific passage in the novel.  
Inconsistent claims lack sufficient semantic evidence or contradict the narrative.

---

## 🏁 Submission

### Track: A  

**Hackathon: Kharagpur Data Science Hackathon 2025**

```
◤                                 ◥            ***     ***
    _   __  ____    ____  _   _               **   ** **   ** 
   | |/ / |  _ \  / ___|| | | |              **     ***     **
   | ' /  | | | | \___ \| |_| |               **           **
   | . \  | |_| |  ___) |  _  |                 **       **
   |_|\_\ |____/  |____/|_| |_|                   **   **
◣                                 ◢                 *

```
<br>

**| Team-mates |**
| Name | Role |
|-----|------|
| Vinayak Dhiman | Lead Developer & System Integration |
| Shaurya Swaraj | Supporting Developer |
| Diksha Jangra | Documentation & Presentation |
| Pritham Prajwin V | Analysis & Team Coordination |
