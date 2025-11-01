import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pickle
from tqdm import tqdm
import re
import gc

print("-------------------LOADING DATASET")

dataset = load_dataset("Johnnyeee/Yelpdata_663")
train_df=dataset["train"].to_pandas()
test_df=dataset["test"].to_pandas()

print("----------------COMBINING TRAIN AND TEST DF")
df = pd.concat([train_df, test_df], ignore_index=True)

print("----------------FILTERING")
df_open = df[df['is_open']==1]
df_state = df_open[~df_open['state'].isin(["HI", 'NC', 'CO', 'MT', 'XMS'])]
test_states=['PA']
df_test = df_state[df_state['state'].isin(test_states)]
df=df_test
print(f"final df shape:{len(df)}")
print(df['state'].value_counts())

#config

DIR=os.path.dirname(os.path.abspath(__file__))
DATA_PATH="/mnt/sdd/rag/ragdata_pa"
os.makedirs(DATA_PATH, exist_ok=True)
BATCH_SIZE=10000
CHUNK_SIZE=500
CHUNK_OVERLAP=200

#embedding preprocessing

print("-----------------------EMBEDDING TEXTS")

texts=df["text"].tolist()

# splitter
print("----------------------INITIALISING SPLITTER")
splitter=RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators = ["\n\n", "\n", ".", "!", "?", ",", " ",""]
)

# CONFIG
print("-----------------------EMBEDDING CONFIG")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = embedding_model.get_sentence_embedding_dimension()

# EXISTING DATA CHECK

existing_chunks = sorted([f for f in os.listdir(DATA_PATH) if f.startswith("chunks_batch_") and f.endswith(".pkl")])
processed_starts = []
for f in existing_chunks:
    m=re.search(r"chunks_batch_(\d+)\.pkl", f)
    if m:
        processed_starts.append(int(m.group(1)))
faiss_index = faiss.IndexFlatIP(dimension)

progress_file = f"{DATA_PATH}/progress.txt"
last_batch = -BATCH_SIZE #default is 0
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        last_batch = int(f.read())
print(f"RESUMING FROM LAST BATCH STARTING AT ROW {last_batch+BATCH_SIZE}")

print(f"---------------RESUMING FROM {len(processed_starts)} existing batches")

for batch_start in range(last_batch+BATCH_SIZE, len(texts), BATCH_SIZE):
    if batch_start in processed_starts:
        print(f"Skipping batch starting at row {batch_start} (already processed)")

        emb_file = f"{DATA_PATH}/chunks_batch_{batch_start}_embeddings.npy"
        if os.path.exists(emb_file):
            print(f"embddings for batch {batch_start} already exists, skip ecndoing")
            emb = np.load(emb_file)
            emb=emb.astype("float32")
            faiss.normalize_L2(emb)
            faiss_index.add(emb)
        continue

    batch_texts = texts[batch_start: batch_start+BATCH_SIZE]
    batch_chunks=[]
    batch_meta = []

    for idx, doc in enumerate(tqdm(batch_texts, desc=f"chunking rows {batch_start}")):
        original_row_index = batch_start + idx
        original_row = df.iloc[original_row_index]

        for chunk in splitter.split_text(doc):
            batch_chunks.append(chunk)
            batch_meta.append({
                "chunk_text": chunk,
                "name": original_row.get("name", ""),
                "city": original_row.get("city", ""),
                "state": original_row.get("state", ""),
                "categories": original_row.get("categories", ""),
                "review_stars": original_row.get("stars_y", ""),
                "original_row_id": original_row_index
                })

            ######################
    
    chunk_file = f"{DATA_PATH}/chunks_batch_{batch_start}.pkl"
    with open(chunk_file, "wb") as f:
        pickle.dump({"chunks": batch_chunks, "metadata": batch_meta}, f)
    
    print(f"--------ENCODING BATCH STARTING AT ROW {batch_start}")
    embeddings = embedding_model.encode(
        batch_chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64
    )

    emb_file = f"{DATA_PATH}/chunks_batch_{batch_start}_embeddings.npy"
    np.save(emb_file, embeddings)
    embeddings=embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    partial_index_file = f"{DATA_PATH}/faiss_index_partial.idx"
    faiss.write_index(faiss_index, partial_index_file)
    with open(progress_file, "w") as f:
        f.write(str(batch_start))

    del batch_chunks, batch_meta, embeddings
    gc.collect()

    print(f"COMPELTED BATCH AT ROW {batch_start}")

faiss_index_file = f"{DATA_PATH}/faiss_index.idx"
faiss.write_index(faiss_index, faiss_index_file)

print(f"----------FAISS INDEX SAVED {faiss_index_file}")