# Indexing and Retrieval

## Overview
This components processes active restaurants in the state of Pennsylvania from the **Yelpdata_663 dataset** (Johnnyeee, 2024), available on Hugging Face: https://huggingface.co/datasets/Johnnyeee/Yelpdata_663. 

Data is encoded using **sentence-transformers/all-MiniLM-L6-v2** and indexed using **FAISS** for semantic search. 

Retrieval combined:
- Semantic similarity score (from FAISS)
- Sentiment confidence scores (from our sentiment model SentimentAnalyzer)

A final weighted score is used for re-ranking retrieved chunks.

## Directory Structure
```
rag
│   city_aliases.json
│   index.py
│   metadata_compile.py
│   requirements.txt
│   retriever.py
│   test.ipynb
| 
├───ragdata_pa
│
├───qa_testing
│   │   chunk_sample.csv
│   │   eval_chunk.py
│   │   generate_forqa.py
│   │   yelp_qa_pairs.json
│   │
│   └───chunk_eval_plots
│           chunk_eval_recall2.png
│           chunk_eval_res2.csv
│           chunk_eval_similarity2.png
│
├───retriever_weight
│       retrieval_weights_res.csv
│       retrieval_weights_res1.csv
│       retrieval_weights_res2.csv
│       retriever_w_tune.py
│       weight_tuning_plot.png
│       weight_tuning_plot1.png
│       weight_tuning_plot2.png
```

## Quick Start

### 1. Build the Index
```bash
cd rag
python3 index.py
```
This:
- Loads filtered PA restaurants
- Splits data into chunks in batches to avoid memory problems
- Encodes data into dense vector embeddings
- Builds FAISS vector index
- Saves index to ```rag/ragdata_pa/```

### 2. Compile metadata
```bash
python3 metadata_compile.py
```
Compiles all metadata files for filtering and scoring

The ```ragdata_pa``` folder is not in this repository. Please download it through https://drive.google.com/file/d/1UnlYKEcrp2Kmtk2kj1oY7_PIuqjG3rtp/view?usp=sharing and place it under ```rag/``` locally.

### 3. Running the retriever individually
```bash
cd smart_dining_assistant
python3 -m rag.retriever
```
The retriever pipeline will:
1. Detect any cities mentioned in user query and filter metadata based on it
2. Encode user query
3. Run ```SentimentAnalyzer``` on user query for confidence score
4. Retrieve top K * 20 chunks from FAISS
5. Scoring retrieved chunks based on sentiment confidence scores and semantic similarity
6. Re-rank chunks to produce final top K outputs

You may edit test queries inside ```retriever.py``` when testing its performance locally

## Experiments
This project includes two experiment modules to optimise retrieval quality.

### 1. Chunk Size and Overlap Optimisation
```bash
cd qa_testing
python3 generate_forpa.py # to generate sample QA pairs to prompt gpt-o4
python3 eval_chunk.py # to evaluate chunk configurations
```
This evaluates:
- Chunk sizes: 128,256,512,1024
- Overlaps: 0 - 500
- Metrics:
    - Recall at Top 3: proportion of chunks with more than 60% semantic similarity to user query
    - Average semantic similarity across all chunks

Insights:
- Larger chunk sizes lead to worser recall due to context dilution
- Smaller chunk sizes work best, however we don't want context to be lost with small chunk sizes
- Overlap helps slightly

### 2. Weight tuning semantic similarity and sentiment scores in retriever
```bash
cd retriever_weight
python3 retriever_w_tune.py
```
This script tests weights from 0.0 to 1.0 for the arguments ```semantic_similarity_weight``` and ```sentiment_weight``` in ```retriever.py``` and evaluates accuracy on 6 test queries.

Insights:
- Increasing performance from 0.0 to 0.8
- Performance peaks at 0.8: retriever will prioritise semantic similarity, while rewarding sentiment matches and high confidence scores
- Drops sharply past 0.8

## Using this component in the project
Given a user query, the retriever will:
1. Detect cities
2. Retrieve relevant chunks
3. Re-rank chunks after scoring
4. Output restaurant recommendations along with its metadata for generator

## Future Work
- Expanding geographical context beyond Pennsylvania
- Support location proximity to user's query
