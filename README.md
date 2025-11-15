# Restaurant Recommender Chatbot

A sentiment-aware conversational AI system that provides personalised restaurant recommendations by analyzing both the semantic content and emotional tone of the user queries.

## Project Overview

Online review platforms provide travelers with extensive restaurant reviews, but this abundance of information ironically causes **decision paralysis** from too many chouces. More critically, existing recommendation systems ignore the fundamental aspect of human decision-making which is **emotional state**.

## Research Gap

Previous work done by Tsang (2022) invovled creating an NLP chatbot for restaurant recommendations using LSTM ensemble models trained on Hong Kong restaurant reviews. He has acheived successful implementation of semantic matching through Word2Vec and GloVe embeddings. However, these systems lacked **emotional awareness**.

Additionally, research in affective computing has demonstrated that incorporating a user's emotional state into conversational systems has led to a significant increase in user satisfaction.

## Our Solution

We address these limitations through a **sentiment-aware Retrieval-Augmented Generation** system that combines:

1. **Real-time Query Sentiment Analysis** using fine-tuned DistilBERT
2. **Emotionally-aligned Retrieval** combining semantic similarity with sentiment matching
3. **Adaptive LLM Generation** that produces contextually and emotioanlly appropriate responses

Our system ewnsures that recommendations match both what users want (semantic content) and how they feel (emotional tone).

## Components

1. **Sentiment Analysis Module** (`sentiment_analysis/`)
- Fine-tuned DistilBERT classifier on 300,000 balanced Yelp reviews
- 81.3% accuracy on 3-class sentiment (Positive, Neutral, Negative)
- 0.7% error rate on extreme polarity confusion

2. 

3. 

## Performance Metrics

### Sentiment Analysis
- **Overall Accuracy**: 81.3%
- **Positive F1-Score**: 87.4%
- **Negaitve F1-Score**: 82.8%
- **Neutral F1-Score**: 74.0%
- **Key Strength**: Only 0.7% confusion between positive and negative sentiment