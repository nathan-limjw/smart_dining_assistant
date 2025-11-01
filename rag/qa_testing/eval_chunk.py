import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
import torch
import json
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DIR, "chunk_eval_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

QA_PATH = os.path.join(DIR,"yelp_qa_pairs.json")
DATA_PATH=os.path.join(DIR,"chunk_sample.csv")

CHUNK_SIZES = [128,256,512,1024]
CHUNK_OVERLAPS = [0,50,100,200,300,400,500]
TOP_K=3
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def load_data():
    df= pd.read_csv(DATA_PATH)
    docs = df["text"].astype(str).tolist()
    with open(QA_PATH, "r") as f:
        qa_pairs=json.load(f)
    return docs, qa_pairs

def build_chunks(docs, chunk_size, chunk_overlap):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunk_docs = splitter.create_documents(docs)
    chunks = [c.page_content for c in chunk_docs]
    return chunks

def create_faiss(embeddings):
    dim = embeddings.shape[1]
    index=faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def eval_retrieval(model, index, chunk_embeddings, chunks, qa_pairs):
    recalls, sims = [], []

    for qa in tqdm(qa_pairs, desc="EVALUATING QA PAIRS"):
        q_vec = model.encode(qa["question"], convert_to_numpy=True, normalize_embeddings=True)
        q_np = np.array(q_vec).reshape(1,-1)

        D,I = index.search(q_np, TOP_K)
        retrieved_texts=[chunks[i] for i in I[0]]

        # answer_text = qa["answer"][:100].lower()
        # recall_hit=any(answer_text in t.lower() for t in retrieved_texts)
        # recalls.append(recall_hit)

        sims_to_q=[]
        for t in retrieved_texts:
            t_vec = model.encode(t, convert_to_numpy=True, normalize_embeddings=True)
            sim_q=np.dot(q_vec, t_vec)
            sims_to_q.append(sim_q)
        recall_hit = any(sim>0.6 for sim in sims_to_q)
        recalls.append(recall_hit)

        best_chunk=retrieved_texts[0]
        a_vec=model.encode(qa["answer"], convert_to_tensor=True).unsqueeze(0)
        best_vec=model.encode(best_chunk, convert_to_tensor=True).unsqueeze(0)
        sim = torch.nn.functional.cosine_similarity(a_vec.cpu(), best_vec.cpu()).item()
        sims.append(sim)

    recall_at_k = np.mean(recalls)
    avg_sim = np.mean(sims)
    return recall_at_k, avg_sim

def main():
    docs, qa_pairs = load_data()
    model=SentenceTransformer(MODEL_NAME)
    res= []

    for chunk_size in CHUNK_SIZES:
        for chunk_overlap in CHUNK_OVERLAPS:
            if chunk_overlap>=chunk_size:
                continue

            print(f"\n----------testing chunk size {chunk_size} and overlap {chunk_overlap}")

            chunks = build_chunks(docs, chunk_size, chunk_overlap)
            chunk_embs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
            index=create_faiss(chunk_embs)

            recall_k, avg_sim = eval_retrieval(model, index, chunk_embs, chunks, qa_pairs)
            res.append({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "recall": recall_k,
                "avg_sim": avg_sim
            })

    res_df = pd.DataFrame(res)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "chunk_eval_res2.csv"), index=False)

    plt.figure(figsize=(14,8))
    for overlap in CHUNK_OVERLAPS:
        subset=res_df[res_df["chunk_overlap"]==overlap]
        plt.plot(subset["chunk_size"], subset["recall"], marker="o", label=f"overlap = {overlap}")

    plt.xlabel('Chunk Size')
    plt.ylabel('Recall at Top K = 3')
    plt.title("Chunk Size vs Recall on QA Evaluation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "chunk_eval_recall2.png"))
    plt.close()
    print("recall png")

    plt.figure(figsize=(14,8))
    for overlap in CHUNK_OVERLAPS:
        subset=res_df[res_df["chunk_overlap"]==overlap]
        plt.plot(subset["chunk_size"], subset["avg_sim"], marker="o", label=f"overlap = {overlap}")

    plt.xlabel('Chunk Size')
    plt.ylabel('Average Semantic Similarity')
    plt.title("Chunk Size vs Semantic Similarity on QA Evaluation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "chunk_eval_similarity2.png"))
    plt.close()
    print("sim score png")

if __name__ == "__main__":
    main()