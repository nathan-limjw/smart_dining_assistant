import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
import pickle

print("-------------------LOADING DATASET")

dataset = load_dataset("Johnnyeee/Yelpdata_663")
train_df=dataset["train"].to_pandas()
test_df=dataset["test"].to_pandas()

print("concating df")
df = pd.concat([train_df, test_df], ignore_index=True)

#embedding preprocessing

print("-----------------------EMBEDDING PREPROCESSING")

meaningful_cols = ["name",
                   "address",
                   "city",
                   "state",
                   "attributes",
                   "categories",
                   "stars_y",
                   "text"]

df["meaningful_text"] = df[meaningful_cols].apply(
    lambda row: " | ".join(row.values.astype(str)), axis=1
)

# embedding

print("-------------------------------------------ENCODING EMBEDDINGS")

docs=df["meaningful_text"].tolist()

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(
    docs,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=128
)

#faiss
print("--------------------------WRITING FAISS INDEX")
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

print("---------------------SAVING INDEX")

os.makedirs("ragdata",exist_ok=True)
faiss.write_index(faiss_index, "ragdata/faiss_index.faiss")
with open("ragdata/rest_metadata.pkl", "wb") as f:
    pickle.dump(df[["business_id",
                    "name",
                   "address",
                   "city",
                   "state",
                   "attributes",
                   "categories",
                   "stars_y",
                   "text"]], f)

print(f"-------------------------------------------------------FAISS Index built with {len(df)} documents")