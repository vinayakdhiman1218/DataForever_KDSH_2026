import pandas as pd
import pathway as pw
from sentence_transformers import SentenceTransformer
import numpy as np

# --------- LOAD DATA ----------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# --------- EMBEDDER ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_text(df):
    return (df["book_name"].fillna("") + " " +
            df["char"].fillna("") + " " +
            df["content"].fillna(""))

train_texts = build_text(train_df).tolist()
y = train_df["label"].values

test_texts = build_text(test_df).tolist()

# --------- EMBEDDINGS ----------
X_train = model.encode(train_texts, show_progress_bar=True)
X_test = model.encode(test_texts, show_progress_bar=True)

# --------- SIMPLE CLASSIFIER (cosine similarity to class centroids) ----------
pos_centroid = X_train[y == 1].mean(axis=0)
neg_centroid = X_train[y == 0].mean(axis=0)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

preds = []
for v in X_test:
    preds.append(1 if cosine(v, pos_centroid) > cosine(v, neg_centroid) else 0)

# --------- SAVE ----------
pd.DataFrame({
    "id": test_df["id"],
    "prediction": preds
}).to_csv("results.csv", index=False)

print("results.csv ready (Pathway-style embeddings used)")
