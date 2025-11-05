import faiss
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import json
import spacy
from sentiment_analysis.src import SentimentAnalyzer

#########PATHS
# # LOCAL PATHS
# DIR = os.path.dirname(os.path.abspath(__file__)) 
# DATA_PATH = os.path.join(DIR,"ragdata_pa")
# INDEX_PATH = os.path.join(DATA_PATH, "faiss_index.idx")
# METADATA_PATH = os.path.join(DATA_PATH, "pa_metadata.pkl")
# CITIES_PATH = os.path.join(DIR, "city_aliases.json")

# RONIN MACHINE PATHS
print("initialising paths")
DATA_PATH = "/opt/dlami/nvme/smart_dining_assistant/rag/ragdata_pa"
INDEX_PATH = os.path.join(DATA_PATH, "faiss_index.idx")
METADATA_PATH = os.path.join(DATA_PATH, "pa_metadata.pkl")
CITIES_PATH = os.path.join("/opt/dlami/nvme/smart_dining_assistant/rag", "city_aliases.json")

########CITIES 
nlp=spacy.load("en_core_web_sm")

def normalize_name(name):
    return re.sub(r"[^\w\s]", "", name.lower()).strip()

def detect_city_query(query, alias_map):
    q_norm = normalize_name(query)
    detected=None

    for key, canonical in alias_map.items():
        if key in q_norm:
            if any(neg in q_norm for neg in [f"not {key}", f"without {key}", f"except {key}"]):
                print(f"negated city {key}")
                continue
            detected = canonical
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
                 , sem_sim_w=0.8
                 , sentiment_w=0.2
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

        #analyse query sentiment
        if analyzer is not None:
            query_sentiment, query_confidence = None, 1.0
            query_df = pd.DataFrame({"text":[query]})
            query_sent = analyzer.analyze_reviews(query_df).iloc[0]
            query_sentiment = query_sent["sentiment"]
            query_confidence = query_sent["sentiment_confidence"]
            print(f"query sentiment: {query_sentiment}, conf: {query_confidence:.2f}")

        for idx, score in zip(indices[0], scores[0]):
            metadata = self.metadata_df.iloc[int(idx)]

            if detected_city:
                city_norm = normalize_name(metadata.get("city", ""))
                if normalize_name(detected_city) != city_norm:
                    continue
            
            r = metadata.to_dict()
            r["retrieval_score"] = float(score)

            if analyzer is not None:
                text_df = pd.DataFrame({"text": [metadata.get("chunk_text","")]})
                sent_df=analyzer.analyze_reviews(text_df).iloc[0]
                r["sentiment"] = sent_df["sentiment"]
                r["sentiment_confidence"] = sent_df["sentiment_confidence"]

                same_sentiment = 1.0 if r["sentiment"] == query_sentiment else 0.0
                r["combined_score"] = (
                    sem_sim_w * r["retrieval_score"]
                    + sentiment_w * same_sentiment * query_confidence * r["sentiment_confidence"]
                )
            else:
                r["combined_score"] = r["retrieval_score"]
            
            res.append(r)
            if len(res)>=top_k*20:
                break

        res.sort(key=lambda x:x["combined_score"], reverse=True)
        return res[:top_k]
    
# if __name__ == "__main__":
#     print("importing SentimentAnalyzer")

#     print("-------------retrieval testing")
#     retriever=Retriever()
#     analyzer=SentimentAnalyzer()

#     queries=[
#         # "good sushi in Philadelphia",
#         # "restaurants to avoid",
#         # "cheap korean food",
#         # "best restaurant to go to",
#         # #positive food specific
#         # "best pizza",
#         # "amazing brunch spots",
#         # "good vegan restaurants",
#         # "affordable fine dining with good reviews",
#         # #negative
#         # "places to avoid",
#         # "restaurants with bad service",
#         # "overrated sushi places",
#         # "worst buffet in town",
#         # # neutral
#         # "places to eat alone",
#         # "casual dining options",
#         # "good lunch spots",
#         # "popular family-friendly restaurants",
#         # # ambiguous
#         # "restaurants that are worth the hype",
#         # "restaurants that are not worth visiting",
#         # "quiet cafes for working",
#         # "expensive but good restaurants",
#         # #city detection
#         # "good sushi in Philadelphia",
#         "cheap korean food in King of Prussia",
#         "restaurants to avoid in West Chester",
#         "vegan cafes in Blue Bell"
#     ]

#     for q in queries:
#         print(f"\n:----------------query: {q}")
#         result = retriever.retrieve(q, top_k=5, analyzer=analyzer)
#         for i, res in enumerate(result):
#             print(f"\n -------------------RESULT {i+1}:----------------")
#             print(f"{res.get('review_stars', '?')} | sentiment: {res.get('sentiment')}")
#             print(f"{res.get('city')} | {res.get('name')}")
#             print(f"score: {res['retrieval_score']:.4f}")
#             print(f"{res['chunk_text']}")

# if __name__ == "__main__":
#     print("-------------CITY DETECTION TEST-------------")

#     # load aliases
#     with open(CITIES_PATH, "r") as f:
#         city_aliases = json.load(f)

#     test_queries = [
#         "good sushi in Philadelphia",
#         "cheap korean food in King of Prussia",
#         "vegan cafes in Blue Bell",
#         "restaurants to avoid in West Chester",
#         "not Philadelphia but nearby",
#         "fine dining without Bryn Mawr",
#     ]

#     for q in test_queries:
#         detected = detect_city_query(q, city_aliases)
#         print(f"Query: {q}")
#         print(f"→ Detected city: {detected}\n")
