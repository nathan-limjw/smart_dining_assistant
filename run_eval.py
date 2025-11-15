# run A/B test on baseline bot VS sentiment-aware bot (test questions from 'test_queries.json')  

import sys
import os
import json
import csv
import time
from dotenv import load_dotenv
load_dotenv() # load .env to get API key

from sentiment_analysis.src.sentiment_api import SentimentAnalyzer
from rag.retriever import Retriever
from llm import prompts, generate
from llm.clean import format_context_for_llm
from tqdm import tqdm

RESULTS_DIR = "llm_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("System starting... Loading models...")

# load models (runs once at the start)
analyzer = SentimentAnalyzer() # load sentiment model 
retriever = Retriever() # load RAG model

# load test queries
with open('test_queries.json', 'r') as f:
    test_queries = json.load(f)

print(f"Loaded {len(test_queries)} test queries.")

# output csv
output_filename = os.path.join(RESULTS_DIR, "evaluation_results.csv")
with open(output_filename, 'w', newline = '', encoding = 'utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Query", 
        "Expected Sentiment", 
        "Detected Sentiment", 
        "Response_A_Baseline", 
        "Response_B_Sentiment_Aware"
    ])

    print(f"Starting Evaluation... Results will be saved to {output_filename}")

    # loop through each query
    for item in tqdm(test_queries, desc = "Running queries"):
        user_query = item['query']
        expected_sentiment = item['sentiment']

        try:

            # run sentiment analysis
            analysis_result = analyzer.analyze(user_query)
            user_sentiment = analysis_result['sentiment']

            # run RAG retrieval
            context_list = retriever.retrieve(user_query, top_k = 5, analyzer=analyzer)
            rag_context_str = format_context_for_llm(context_list)

            # build the shared user prompt
            final_user_prompt = f'''

            ***Context: Restaurant Information***
            {rag_context_str}
            ---
            **User Query:**
            {user_query}
            '''
            
            # --- RUN VERSION A (BASELINE BOT) ---
            response_a = generate.call_llm(
                system_prompt=prompts.BASE_SYSTEM_INSTRUCTION,
                user_prompt=final_user_prompt
            )

            # --- RUN VERSION B (SENTIMENT-AWARE) ---
            sentiment_system_prompt = prompts.get_system_prompt(user_sentiment)
            response_b = generate.call_llm(
                system_prompt=sentiment_system_prompt,
                user_prompt=final_user_prompt
            )

            # --- SAVE RESPONSE TO CSV ---
            writer.writerow([
                user_query, 
                expected_sentiment, 
                user_sentiment, # what the model detected
                response_a,
                response_b
            ])

        except Exception as e:
            print(f"\n --- ERROR on query: '{user_query}' --- ")
            print(f"Error: {e}")
            writer.writerow([
                user_query,
                expected_sentiment, 
                "ERROR",
                str(e),
                str(e)
            ])

        time.sleep(20) # to avoid rate limiting

print("\n --- EVALUATION COMPLETED --- ")

        





