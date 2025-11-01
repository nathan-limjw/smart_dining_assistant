import faiss
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import json
import spacy

#########PATHS
# DIR = os.path.dirname(os.path.abspath(__file__)) # local only
print("initialising paths")
DATA_PATH = "/mnt/sdd/rag/ragdata_pa"
INDEX_PATH = os.path.join(DATA_PATH, "faiss_index.idx")
METADATA_PATH = os.path.join(DATA_PATH, "pa_metadata.pkl")
CITIES_PATH = os.path.join("/mnt/sdd/rag", "city_aliases.json")

########CITIES 
nlp=spacy.load("en_core_web_sm")

def normalize_name(name):
    return re.sub(r"[^\w\s]", "", name.lower()).strip()

def detect_city_query(query, alias_map):
    doc=nlp(query)
    q_norm = normalize_name(query)
    detected=None

    for key in alias_map.keys():
        for ent in doc:
            if key in ent.text.lower():
                negated = any(child.dep_ == "neg" or child.text.lower() in ["not", 'except', 'without']
                              for child in ent.lefts)
                if negated:
                    print(f"negated city {key}")
                    continue
                detected=alias_map[key]
                break
        if detected:
            break
    return detected


#########RETRIEVAL

class Retriever:
    def __init__(self):
        
        print("LOADING FAISS INDEX AND METADATA")
        self.index = faiss.read_index(INDEX_PATH)
        self.metadata_df = pd.read_pickle(METADATA_PATH)

        print(self.metadata_df.columns)
        print(self.metadata_df.head())
        print("SUCCESSFULLY LOADED FAISS INDEX AND METADATA")

        with open(CITIES_PATH, "r") as f:
            self.city_aliases=json.load(f)
        print("loaded cities json")

        if self.index.ntotal != len(self.metadata_df):
            print("WARNING: index size and metadata size do not match")

        print("init embedding model")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def retrieve(self
                 , query
                 , top_k=5
                 , analyzer=None
                 ):
        detected_city = detect_city_query(query, self.city_aliases)
        if detected_city:
            print(f"city detected: {detected_city}")
        else:
            print("no city detected")

        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        query_emb=query_emb.astype("float32")
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k*20)
        res = []

        for idx, score in zip(indices[0], scores[0]):
            metadata = self.metadata_df.iloc[int(idx)]

            if detected_city:
                city_norm = normalize_name(metadata.get("city", ""))
                if normalize_name(detected_city) != city_norm:
                    continue
            
            r = metadata.to_dict()
            r["retrieval_score"] = float(score)

            if analyzer is not None:
                text_df = pd.DataFrame({"text": [res["chunk_text"]]})
                sentiment=analyzer.analyze_reviews(text_df)
                r["sentiment"] = sentiment.iloc[0].get("sentiment", "unknown")
            
            res.append(r)
            if len(res)>=top_k:
                break
        return res
    
if __name__ == "__main__":
    print("importing SentimentAnalyzer")
    from sentiment_model import SentimentAnalyzer

    print("-------------retrieval testing")
    retriever=Retriever()
    analyzer=SentimentAnalyzer()

    queries=[
        "good sushi in Philadelphia",
        "restaurants to avoid in Ardmore",
        "cheap korean food",
        "best restaurant to do to"
    ]

    for q in queries:
        print(f"\n:----------------query: {q}")
        result = retriever.retrieve(q, top_k=5, analyzer=analyzer)
        for i, res in enumerate(result):
            print(f"\n -------------------RESULT {i+1}:----------------")
            print(f"{res.get('review_stars', '?')} | sentiment: {res.get('sentiment')}")
            print(f"{res.get('city')} | {res.get('name')}")
            print(f"score: {res['retrieval_score']:.4f}")
            print(f"{res['chunk_text']}")