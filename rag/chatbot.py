import numpy as np
import pandas as pd
import faiss
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

class Chatbot:
    def __init__(self, api_key, index_path, data_path, sentiment_model_path):
        self.df = pd.read_parquet(data_path)
        self.index=faiss.read_index(index_path)

        self.embedding_model=SentenceTransformer("sentence-transformer/all-MiniLM-L6-v2")

        self.sentiment_analyzer = pipeline("sentiment-analysis",mode=sentiment_model_path)

        self.llm=HuggingFaceEndpoint(
            repo_id="google/gemma-2b-it",
            huggingfacehub_api_token=api_key,
            temperature=0.2,
            max_new_tokens=256,
            timeout=120,
            options={"wait_for_model":True,
                     "use_cache":True}
        )

        template="""
        You are a helpful assistance chatbot that answers the user's query based on Yelp restaurant reviews.
        Use the following pieces of context from Yelp restaurant reviews to answer the user's questions. 
        If you cannot find the answer based only on the provided context, politely state that there is a lack of information in the given reviews. 

        Context:
        ---
        {context}
        ---

        Question:
        {question}

        Answer:
        """

        self.prompt=PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def search(self, query, k=5):
        Q_vec = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(Q_vec)
        D, I = self.index.search(Q_vec, k)
        res=self.df.iloc[I[0]]["text"].tolist()
        return res
    
    def analyse_senti(self, texts):
        res=self.analyse_senti(texts)
        return res
    
    def generator(self, query):
        retireved = self.search(query, k=5)
        sentiments=self.analyse_senti(retireved)
        sentiments_summ=", ".join([r["label"] for r in sentiments])

        context="\n".join(retireved)
        inputs={"context":context, "question": query}

        rag_chain = self.prompt | self.llm | StrOutputParser()
        ans = rag_chain.invoke(inputs)

        return f"Answer: {ans}\n\n Detected snetiments in context:{sentiments_summ}"