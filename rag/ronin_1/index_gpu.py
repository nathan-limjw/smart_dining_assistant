import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pickle
from tqdm import tqdm
import torch
import gc
import re

#LOADING DATA
print("-------------------LOADING DATASET")

dataset = load_dataset("Johnnyeee/Yelpdata_663")
train_df=dataset["train"].to_pandas()
test_df=dataset["test"].to_pandas()
print("----------------COMBINING TRAIN AND TEST DF")
df = pd.concat([train_df, test_df], ignore_index=True)
print("----------------FILTERING")
df_open = df[df['is_open']==1]
df=df_open
df = df [["text", 'name', 'city', 'state', 'postal_code', 'address', 'categories', 'stars_y', 'hours']]

#OUTPUT CONFIG
DIR=os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(DIR, "ragdata")
DATA_PATH="/mnt/sdd/rag/ragdata"
os.makedirs(DATA_PATH, exist_ok=True)

CHUNK_SIZE=256
CHUNK_OVERLAP=200
BATCH_SIZE = 256
SAVE_EVERY=100000

#INITIALISING MODELS
print("----------------------INITIALISING SPLITTER")
splitter=RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators = ["\n\n", "\n", ".", "!", "?", ",", " ",""]
)
print("----------------------INITIALISING EMBEDDING MODEL")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
dim = embedding_model.get_sentence_embedding_dimension()
# res = faiss.StandardGpuResources()
# cpu_index=faiss.IndexFlatIP(dim)
# faiss_index=faiss.index_cpu_to_gpu(res, 0, cpu_index)
faiss_index = faiss.IndexFlatIP(dim)

#CHECKPOINT
print("----------------CHECKING CHECKPOINTS")
existing_checkpoints = [f for f in os.listdir(DATA_PATH) if re.match(r"faiss_partial_\d+\.idx", f)]
if existing_checkpoints:
    latest = max(existing_checkpoints, key = lambda x: int(re.findall(r"\d+", x)[0]))
    latest_idx=int(re.findall(r"\d+", latest)[0])
    print(f"resuming from checkpoint {latest} (row {latest_idx})")
    faiss_index=faiss.read_index(os.path.join(DATA_PATH, latest))
    start_idx = latest_idx+1
else:
    print("no checkpoint found, starting from 0")
    faiss_index=faiss.IndexFlatIP(dim)
    start_idx=0


#BEGINNING SPLITTING AND ENCODING
print("----------------------BEGINNING ENCOING AND SPLITTING")
chunk_count=0
metadata_buffer=[]
for idx, doc in enumerate(tqdm(df.itertuples(index=False, name=None), total=len(df), desc="PROCESSING REVIEWS")):
    if idx<start_idx:
        continue

    text, name, city, state, postal, address, categories, stars_y, hours=doc

    print(f"------------------EMBEDDING FOR ROW {idx} -----------------")
    chunks = splitter.split_text(str(text))

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE
    )
    embeddings=embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    for chunk in chunks:
        metadata_buffer.append({
            "chunk_text": chunk,
            "name": name,
            "city": city,
            "state": state,
            "postal_code": postal,
            "address": address,
            "categories": categories,
            "review_stars": stars_y,
            "hours": hours,
            "original_row_id": idx
        })
    
    chunk_count+=len(chunks)

    if chunk_count>=SAVE_EVERY:
        tmp_path=os.path.join(DATA_PATH, f"faiss_partial_{idx}.idx")
        meta_tmp = os.path.join(DATA_PATH, f"metadata_partial_{idx}.pkl")
        with open(meta_tmp, "wb") as f:
            pickle.dump(metadata_buffer, f)
        faiss.write_index(faiss_index, tmp_path)
        print(f"CHECKPOINT SAVED {tmp_path}")
        chunk_count=0
        torch.cuda.empty_cache()
        metadata_buffer.clear()
        gc.collect()

faiss_last=os.path.join(DATA_PATH, f"faiss_partial_{idx}.idx")
faiss.write_index(faiss_index, faiss_last)
print("LAST FAISS INDEX FILE SAVED")

meta_last = os.path.join(DATA_PATH, "chunks_metadata.pkl")
with open(meta_last, "wb") as f:
    pickle.dump(metadata_buffer, f)
print("metadata aved")