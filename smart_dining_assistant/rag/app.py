import streamlit as st
import faiss
import pandas as pd
from chatbot import Chatbot

@st.cache_resource
def load_chatbot():
    return Chatbot(
        api_key=st.secrets["HUGGINGFACE_TOKEN"],
        index_path="data/faiss_index.index",
        data_path = "data/yelp_data.parquet",
        sentiment_model_path="models/sentiment_model"
    )

st.title("Restaurant Recommender")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        text
        """
    )

chatbot=load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role":"assistant",
        "content": "Welcome to the Restaurant Recommender! Feel free to ask for a comprehensive review of restaurants near you!"
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask anything"):
    st.session_state.messages.append({"role":"user",
                                      "content":query})
    with st.chat_messages("user"):
        st.write(query)
    
    with st.chat_mesage("assistant"):
        with st.spinner("Retrieving content and analysing..."):
            response=chatbot.generator(query)
            st.session_state.messages.append({
                "role":"assistant",
                "content":response
            })
            st.write(response)


# pip install streamlit faiss-cpu sentence-transformers transformers langchain-huggingface datasets pandas tqdm
# streamlit run app.py
