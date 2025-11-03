# main function to run chatbot
import sys
from dotenv import load_dotenv
load_dotenv() # load var from .env files

import os
import json
from sentiment_analysis.src.sentiment_api import SentimentAnalyzer
from rag.retriever import Retriever
from llm.clean import format_context_for_llm
from llm.prompts import get_system_prompt
from llm.generate import call_llm

def main():

    # load models
    print("Loading Sentiment Analysis...")
    analyzer = SentimentAnalyzer()

    print("Loading RAG Retriever...")
    retriever = Retriever()

    print("\n--- Sentiment-Aware Chatbot Ready ---")
    print("Type your query, or 'quit' to exit.")

    while True:
        try: 
            # Get user query
            user_query = input("\nYou: ")
            if user_query.lower() == 'quit':
                print("Goodbye!")
                break

            # FULL PIPELINE

            # Run Sentiment Analyis Model
            print("[Running: Sentiment Analysis...]")
            analysis_result = analyzer.analyze(user_query)
            user_sentiment = analysis_result['sentiment']
            print(f"[Debug: Sentiment detected: {user_sentiment}]")

            # Run RAG Retrieval
            print("[Running: RAG Retrieval...]")
            context_list = retriever.retrieve(user_query, top_k = 5, analyzer=analyzer)

            # Run LLM Prompt generation
            print("[Running: LLM Prompt Generation...]")
            rag_context_str = format_context_for_llm(context_list)  # format RAG context into a string
            system_prompt = get_system_prompt(user_sentiment) # get the correct system prompt based on user sentiment
            final_user_prompt = f'''
            **Context: Restaurant Information**
            {rag_context_str}
            ---
            **User Query:**
            {user_query}
            '''

            # Call the LLM
            print('[Running: LLM Response Generation...]')
            response = call_llm(
                system_prompt = system_prompt,
                user_prompt = final_user_prompt
            )

            # Display final answer
            print("\nChatbot:")
            print(response)

        except Exception as e:
            print(f"\nAn error occured: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()           

