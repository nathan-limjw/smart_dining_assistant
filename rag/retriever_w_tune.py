import numpy as np
import os
import pandas as pd
from sentiment_analysis.src import SentimentAnalyzer
import matplotlib.pyplot as plt
from rag.retriever import INDEX_PATH, METADATA_PATH, Retriever

#########PATHS
# #local
# DIR = os.path.dirname(os.path.abspath(__file__)) # local only
# print("initialising paths")
# DATA_PATH = os.path.join(DIR, "../ragdata_ca")

# INDEX_PATH = os.path.join(DATA_PATH, "faiss_index.idx")
# METADATA_PATH = os.path.join(DATA_PATH, "rest_metadata.pkl")
# OUTPUT_PATH = os.path.join(DIR, "retriever_weight")
# os.makedirs(OUTPUT_PATH, exist_ok=True)

#ronin
DIR="/opt/dlami/nvme/smart_dining_assistant"
DATA_PATH = os.path.join(DIR, "rag","ragdata_pa")
OUTPUT_PATH = os.path.join(DIR,"rag", "retriever_weight")
os.makedirs(OUTPUT_PATH, exist_ok=True)
INDEX_PATH = os.path.join(DATA_PATH, "faiss_index.idx")
METADATA_PATH = os.path.join(DATA_PATH, "pa_metadata.pkl")

#####

retriever=Retriever()
analyzer = SentimentAnalyzer()

test_queries=[
    ("good sushi to go for", "positive"),
    ("cheap korean food", "neutral"),
    ("restaurants to not go visit", "negative"),
    ("places that are worth the hype", "positive"),
    ("what are the overhyped places", "negative"),
    ("places to chill at", "neutral")
]

retrieval_weights = np.linspace(0.0,1.0,11)
sentiment_weights = 1.0 - retrieval_weights
w_res = []

print("RETRIEVAL WEIGHT SEARCH")

for wr, ws in zip(retrieval_weights, sentiment_weights):
    total_correct=0
    total=0

    print(f"\n---------TESTING retrieval weight {wr}, sentiment weight {ws}")

    for q, expected_s in test_queries:
        res_list = retriever.retrieve(
            q,
            top_k=5,
            analyzer=analyzer
            , sem_sim_w=wr
            , sentiment_w=ws
        )

        # top_sents = [r.get("sentiment", "unknown") for r in res_list]
        # correct = sum(sent==expected_s for sent in top_sents)
        # total_correct +=correct
        # total += len(top_sents)

        chunk_scores=[]
        for r in res_list:
            sent_match = 1.0 if r.get("sentiment")==expected_s else 0.0
            sem_sim = r.get("retrieval_score", 0.0)
            chunk_quality = 0.5 * sem_sim + 0.5 * sent_match
            chunk_scores.append(chunk_quality)

        if chunk_scores:
            avg_quality = np.mean(chunk_scores)
            total_correct +=avg_quality
            total +=1

    acc = total_correct/total
    print(f"acc for weights ({wr}, {ws}: {acc:.5f})")
    w_res.append((wr,ws,acc))

    df=pd.DataFrame(w_res, columns=["semantic_similarity_score", "sentiment_weight", "accuracy"])
    csv_path = os.path.join(OUTPUT_PATH, "retrieval_weights_res2.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n results saved to {csv_path}")

    best=max(w_res, key=lambda x:x[2])
    print(f"\n best weights are: retrieval = {best[0]}, sentiment = {best[1]}, accuracy = {best[2]}")

    plt.figure(figsize=(14,8))
    plt.plot(df["semantic_similarity_score"], df["accuracy"], marker = "o")
    plt.title("Weight tuning for semantic similarity score and sentiment confidence")
    plt.xlabel("Semantic similarity score weight (Sentiment weight = 1 - semantic_similarity_score)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt_path = os.path.join(OUTPUT_PATH, "weight_tuning_plot2.png")
    plt.savefig(plt_path, dpi=300)
    plt.close()
    print(f"plot saved to {plt_path}")