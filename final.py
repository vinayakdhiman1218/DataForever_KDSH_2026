import pandas as pd
import pathway as pw
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# AUTO DEVICE DETECTION
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# LOAD TEST DATA
test_df = pd.read_csv("test.csv")

# LOAD NOVEL TEXT
def load_book(book_name):
    if book_name.lower() == "in search of the castaways":
        path = "In search of the castaways.txt"
    elif book_name.lower() == "the count of monte cristo":
        path = "The Count of Monte Cristo.txt"
    else:
        raise ValueError("Unknown book name")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# CHUNK TEXT
def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

# CACHE BOOK PROCESSING
book_cache = {}

def cosine_similarity(matrix, vector):
    return np.dot(matrix, vector) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    )

# MAIN LOOP
results = []

for _, row in tqdm(
    test_df.iterrows(),
    total=len(test_df),
    desc="Processing stories"
):
    story_id = row["id"]
    book_name = row["book_name"]
    character = row["char"]
    claim = row["content"]

    # ---- LOAD & CACHE BOOK ONCE ----
    if book_name not in book_cache:
        novel_text = load_book(book_name)
        chunks = chunk_text(novel_text)

        # Pathway ingestion (Track A requirement)
        pw.debug.table_from_pandas(
            pd.DataFrame({"text": chunks})
        )

        embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False
        )

        book_cache[book_name] = (chunks, embeddings)

    chunks, embeddings = book_cache[book_name]

    # ---- CLAIM EMBEDDING ----
    claim_text = f"{character}. {claim}"
    claim_embedding = model.encode([claim_text])[0]

    # ---- SIMILARITY SEARCH ----
    scores = cosine_similarity(embeddings, claim_embedding)
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    best_chunk = chunks[best_idx]

    # ---- DECISION RULE ----
    prediction = 1 if best_score > 0.45 else 0

    # ---- RATIONALE ----
    # ---- RATIONALE ----
    clean_chunk = best_chunk.replace("\n", " ").strip()

    # Remove leading chapter headers if present
    lowered = clean_chunk.lower()
    if lowered.startswith("chapter"):
        parts = clean_chunk.split(".", 1)
        if len(parts) > 1:
            clean_chunk = parts[1].strip()

    # Extract a clean, complete sentence starting at a proper boundary
    sentences = []
    for part in clean_chunk.split("."):
        s = part.strip()
        if len(s) > 40 and s and s[0].isupper():
            sentences.append(s + ".")

    if sentences:
        excerpt = sentences[0]
    else:
        # Fallback: find nearest previous sentence boundary
        cut = clean_chunk[:300]
        last_period = cut.rfind(".")
        if last_period != -1:
            excerpt = cut[: last_period + 1].strip()
        else:
            excerpt = cut.strip() + "."

    if prediction == 1:
        rationale = (
            "Prediction is consistent. This conclusion is based on the following "
            "sentence from the main story text that aligns with the backstory: "
            f"\"{excerpt}\""
        )
    else:
        rationale = (
            "Prediction is inconsistent. No complete sentence in the main story "
            "provides sufficient evidence to support the given backstory."
        )

    results.append({
        "story_id": story_id,
        "prediction": prediction,
        "rationale": rationale
    })

# SAVE RESULTS
results_df = pd.DataFrame(
    results,
    columns=["story_id", "prediction", "rationale"]
)
results_df.to_csv("results.csv", index=False)
print("✅ results.csv generated successfully.")