import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, index_path="ragdata/faiss_index.faiss", metadata_path="ragdata/rest_metadata.pkl"):
        print("LOADING FAISS INDEX AND METADATA")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata=pickle.load(f)
        
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def retrieve(self, query, top_k=5):
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, top_k)
        res=[]
        for idx, score in zip(indices[0], scores[0]):
            info=self.metadata.iloc[idx].to_dict()
            info["score"]=float(score)
            res.append(info)
        return res