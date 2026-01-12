import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# CONFIG
MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.45

print("🔹 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

#  LOAD DATA 
df = pd.read_csv("train.csv")

# Normalize labels safely
def normalize_label(x):
    x = str(x).strip().lower()
    if x in ["1", "consistent", "true", "yes"]:
        return 1
    if x in ["0", "inconsistent", "false", "no"]:
        return 0
    return np.nan

df["label"] = df["label"].apply(normalize_label)

if df["label"].isna().any():
    unknown_count = df["label"].isna().sum()
    print(f"⚠️ Warning: {unknown_count} rows have unknown labels. Dropping them.")
    df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

# SPLIT
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")

# LOAD BOOK TEXT 
def load_book(book_name):
    if book_name == "In Search of the Castaways":
        path = "In search of the castaways.txt"
    else:
        path = "The Count of Monte Cristo.txt"

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

#  TEXT 
def chunk_text(text, size=800, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

# CACHE 
book_cache = {}

# PREDICT FUNCTION 
def predict_single(book_name, character, claim):
    if book_name not in book_cache:
        book_text = load_book(book_name)
        chunks = chunk_text(book_text)
        embeddings = model.encode(chunks, show_progress_bar=False)
        book_cache[book_name] = embeddings
    else:
        embeddings = book_cache[book_name]

    query = f"{character}. {claim}"
    q_emb = model.encode([query])

    sims = cosine_similarity(q_emb, embeddings)[0]
    score = np.max(sims)

    return 1 if score > THRESHOLD else 0

# VALIDATION
y_true, y_pred = [], []

print("🔹 Running validation...")
for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validating"):
    pred = predict_single(
        book_name=row["book_name"],
        character=row["char"],
        claim=row["content"]
    )
    y_true.append(row["label"])
    y_pred.append(pred)

# METRICS
print("\n📊 Validation Performance")
print("-" * 40)
print("Prediction distribution:", pd.Series(y_pred).value_counts().to_dict())
print(f"Accuracy  : {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_true, y_pred, zero_division=0)*100:.2f}%")
print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0)*100:.2f}%")
print(f"F1 Score  : {f1_score(y_true, y_pred, zero_division=0)*100:.2f}%")

print("\n✅ Validation completed successfully.")
