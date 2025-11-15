# 🍽️ Sentiment-Aware Restaurant Recommender Chatbot

A sentiment-aware conversational AI system that provides personalised restaurant recommendations by analyzing both the semantic content and emotional tone of the user queries.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 🧠 Project Overview

Online review platforms like Yelp provide travelers with thousands of restaurant reviews, but this abundance often leads to **decision paralysis**. Traditional recommendation systems focus only on what users want (semantic intent) while ignoring how they feel (emotional state).

Our system bridges this gap through a **sentiment-aware Retrieval Augmented Generation (RAG)** architecture that:
- 🎯 Analyzes user query sentiment in real-time using fine-tuned DistilBERT
- 🔍 Retrieves emotionally-aligned restaurant reviews via hybrid semantic-sentiment scoring
- 💬 Generates contextually and emotionally appropriate responses using dynamic LLM prompting

## 📉 Research Gap

### Prior Work

Tsang (2022) developed an LSTM-based ensemble model for restaurant recommendations trained on Hong Kong restaurant reviews using Word2Vec/GloVe embeddings, enabling natural language queries. However:

- ❌ No emotional awareness - same response whether user is stressed or excited
- ❌ Static recommendation strategy - no adaptive behavior
- ❌ Semantic matching only - ignores psychological state during decision-making

Research in affective computing shows that incorporating user emotional state significantly increases satisfaction and trust.

## 💡Our Solution

We address these limitations through a **sentiment-aware Retrieval-Augmented Generation** system that combines:

1. **Real-time Query Sentiment Analysis** using fine-tuned DistilBERT
2. **Emotionally-aligned Retrieval** combining semantic similarity with sentiment matching
3. **Adaptive LLM Generation** that produces contextually and emotioanlly appropriate responses

Our system ensures that recommendations match both what users want (semantic content) and how they feel (emotional tone).

## 📁 Directory Structure
```
smart_dining_assistant/
├── 📜 README.md                    # This file
├── 📜 requirements.txt             # Root-level dependencies
├── 📜 .env                         # API keys (create manually, see setup)
│
├── 📜 app.py                       # 🚀 Main interactive chatbot
├── 📜 run_eval.py                  # A/B test script (generates responses)
├── 📜 evaluate_ablation.py         # Quantitative metrics computation
├── 📜 test_queries.json            # Test query bank (25 queries)
│
├── 📂 sentiment_analysis/          # 🎯 Sentiment Classification Module
│   ├── 📂 data/                   # train.csv, val.csv, test.csv
│   ├── 📂 models/                 # Fine-tuned DistilBERT checkpoints
│   │   └── sentiment_model/       # Best model (Config 2)
│   ├── 📂 results/                # Evaluation outputs
│   │   ├── hyperparameter_tuning/ # Tuning metrics, confusion matrix
│   │   └── model_evaluation/      # Test set results
│   ├── 📂 src/                    # Core modules
│   │   ├── sentiment_api.py       # SentimentAnalyzer class (main API)
│   │   ├── load_yelp_data.py      # Data preprocessing
│   │   ├── hyperparameter_tuning.py
│   │   └── evaluate.py
│   ├── 📜 main.py                 # Run complete pipeline
│   ├── 📜 requirements.txt
│   └── 📜 README.md
│
├── 📂 rag/                        # 🔍 Retrieval System
│   ├── 📂 ragdata_pa/             # ⚠️ Download separately (see setup)
│   │   ├── faiss_index/           # FAISS vector index
│   │   └── metadata/              # Restaurant metadata, sentiment labels
│   ├── 📂 qa_testing/             # Chunk optimization experiments
│   │   ├── eval_chunk.py          # Evaluate chunk size/overlap
│   │   └── yelp_qa_pairs.json     # Synthetic QA pairs for testing
│   ├── 📂 retriever_weight/       # Weight tuning experiments
│   │   ├── retriever_w_tune.py    # α optimization script
│   │   └── weight_tuning_plot.png
│   ├── 📜 retriever.py            # Main Retriever class
│   ├── 📜 index.py                # FAISS index builder
│   ├── 📜 metadata_compile.py     # Metadata compilation
│   ├── 📜 city_aliases.json       # City name normalization
│   ├── 📜 requirements.txt
│   └── 📜 README.md
│
├── 📂 llm/                        # 💬 LLM Generation Module
│   ├── 📜 prompts.py              # Dynamic prompt templates
│   ├── 📜 generate.py             # Gemini API wrapper
│   ├── 📜 clean.py                # Context formatting utilities
│   └── 📜 README.md
│
└── 📂 llm_results/                # Evaluation outputs
    ├── 📜 evaluation_results.csv       # Side-by-side bot responses
    └── 📜 eval_metrics_detailed.csv    # Quantitative metrics
```

## 📦 Components

### 1. **Sentiment Analysis Module** ([`sentiment_analysis/`](./sentiment_analysis/))
Fine-tuned DistilBERT classifier for 3-class sentiment detection on restaurant review text.

**Architecture:**
- **Base Model:** `distilbert-base-uncased` (6 layers, 66M parameters)
- **Training Data:** 300,000 balanced Yelp restaurant reviews
- **Classes:** Negative (stars < 3), Neutral (stars = 3), Positive (stars > 3)
- **Optimal Config:** LR=3e-5, batch=16, epochs=3, weight_decay=0.01

**Performance (45,000 test samples):**
| Metric | Value |
|--------|-------|
| Overall Accuracy | **81.27%** |
| Precision | 81.66% |
| Recall | 81.27% |
| F1 Score | 81.38% |

**Per-Class F1 Scores:**
- Positive: **87.39%** (best performance)
- Negative: **82.78%** (strong reliability)
- Neutral: **73.98%** (challenging due to mixed sentiments)

**Key Strength:** Only **0.7%** confusion between positive/negative extremes, ensuring clear polarity distinction.

📖 **[Detailed Documentation](./sentiment_analysis/README.md)**
  
**Quick Start:**
```bash
cd sentiment_analysis
pip install -r requirements.txt
python main.py  # Runs complete pipeline: data prep → tuning → training → evaluation
```

### 2. **RAG Retrieval System** ([`rag/`](./rag/))
Hybrid retrieval combining semantic similarity with sentiment alignment using FAISS vector search.

**Data Source:**
- **Dataset:** [Yelpdata_663](https://huggingface.co/datasets/Johnnyeee/Yelpdata_663) (Johnnyeee, 2024)
- **Scope:** 835,954 reviews from active restaurants in Pennsylvania, USA
- **Geographic Coverage:** ~243 cities with fuzzy city name matching

**Technical Stack:**
- **Vector Store:** FAISS with L2-normalized IndexFlatIP (Inner Product)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (lightweight, sentence-level)
- **Chunking:** RecursiveCharacterTextSplitter (size=256, overlap=200)
- **City Detection:** Fuzzy matching via SequenceMatcher + Jaccard similarity

**Hybrid Scoring Function:**
```
Score_final = α × Sim_semantic + (1-α) × (Conf_query × Conf_chunk × Match_sentiment)

where:
  α = 0.8 (semantic weight)
  1-α = 0.2 (sentiment weight)
  Conf_query, Conf_chunk = sentiment confidence scores [0,1]
  Match_sentiment = 1 if sentiments match, else 0
```

**Optimization Results:**
- **Chunk Configuration:** Size=256, Overlap=200 achieved best recall (52%) + similarity (0.493)
- **Weight Tuning:** α=0.8 balances semantic relevance with emotional alignment
- Prevents mismatched retrievals (e.g., positive reviews for "restaurants to avoid")

📖 **[Detailed Documentation](./rag/README.md)**

**Quick Start**
```bash
cd rag
pip install -r requirements.txt

# Build FAISS index (requires downloading ragdata_pa/ - see below)
python index.py

# Compile metadata for filtering
python metadata_compile.py

# Test retriever individually
cd ..
python -m rag.retriever
```

**⚠️ Data Download Required:**  
The `ragdata_pa/` folder (~1.5GB) is not included in this repo due to size.  
**Download:** [Google Drive Link](https://drive.google.com/file/d/1UnlYKEcrp2Kmtk2kj1oY7_PIuqjG3rtp/view?usp=sharing)  
**Place:** Extract to `rag/ragdata_pa/`

---

### 3. **LLM Generation Module** ([`llm/`](./llm/))

Action-oriented dynamic prompting using Google Gemini 2.5 Flash API.

**Core Philosophy:**  
Instead of just changing *tone*, we change *what the bot does* based on user emotional state.

**Prompt Architecture:**
- **Base Instruction** (all bots):
  - Use ONLY provided context (no hallucination)
  - Be concise (2-3 sentences)
  - Provide factual, grounded recommendations

- **Sentiment-Specific Actions:**

| Sentiment | Persona | Action |
|-----------|---------|--------|
| **Positive** 😊 | Cheerful Travel Guide | Give **2-3 options** with exciting details; quote review highlights (dishes, ambiance) |
| **Negative** 😫 | Warm, Empathetic Assistant | Recommend **1-2 easiest options**; explain WHY they're convenient to reduce stress |
| **Neutral** 😐 | Calm, Reliable Assistant | IF indecisive → ask clarifying question; ELSE → give direct factual answer |

**Implementation:**
- `prompts.py` - Dynamic prompt selection via `get_system_prompt(sentiment)`
- `generate.py` - Gemini 2.5 Flash API wrapper with safety settings
- `clean.py` - De-duplicates and formats RAG context for LLM consumption

📖 **[Detailed Documentation for Evaluation Results & Key Insights](./llm/README.md)**

**API Setup:**
```bash
# Create .env file in project root
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

---

---

## 🚀 Quick Start

### Prerequisites
- **Python:** 3.8 or higher
- **Google AI Studio API Key:** Required for Gemini 2.5 Flash ([Get one here](https://aistudio.google.com/app/apikey))
- **Storage:** ~2GB for FAISS index and model checkpoints
---

### Installation
```bash
# 1. Clone repository
git clone https://github.com/your-username/smart_dining_assistant.git
cd smart_dining_assistant

# 2. Install root dependencies
pip install -r requirements.txt

# 3. Install component-specific dependencies
pip install -r sentiment_analysis/requirements.txt
pip install -r rag/requirements.txt
```

---
### Data & Model Setup

#### 1. **Download FAISS Index & Metadata**

The `rag/ragdata_pa/` folder (~1.5GB) is not included in this repo.

📥 **Download:** [Google Drive Link](https://drive.google.com/file/d/1UnlYKEcrp2Kmtk2kj1oY7_PIuqjG3rtp/view?usp=sharing)
```bash
# After downloading, extract to rag/
cd rag
unzip ragdata_pa.zip  # (or extract manually)

# Verify structure:
ls ragdata_pa/
# Expected: faiss_index/, metadata/, chunks.pkl, etc.
```

#### 2. **Download Sentiment Model** (Optional)

The fine-tuned DistilBERT model is already included in `sentiment_analysis/models/sentiment_model/`.

If missing or corrupted, re-train by running:
```bash
cd sentiment_analysis
python main.py  # Runs full pipeline: data prep → training → evaluation
```

#### 3. **Set Up API Key**

Create a `.env` file in the **project root**:
```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key_here
```

🔑 Get your key: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

### Running the Chatbot
```bash
python app.py
```

**Example Interaction:**
```
--- Sentiment-Aware Chatbot Ready ---
Type your query, or 'quit' to exit.

You: I'm so excited to explore new cuisines! What's unique in Springfield?

[Running: Sentiment Analysis...]
[Debug: Sentiment detected: positive]
[Running: RAG Retrieval...]
[Running: LLM Prompt Generation...]
[Running: LLM Response Generation...]

Chatbot:
Oh, you are going to have an amazing time exploring the unique flavors 
of Springfield! For something truly special, you absolutely have to check 
out Nick's Old Original Roast Beef. You will love their amazing roast beef 
and the absolutely wonderful gravy fries that are a local sensation!

You: quit
Goodbye!
```

---

### Running A/B Evaluation

#### Step 1: Generate Bot Responses
```bash
python run_eval.py
```
- Processes all queries in `test_queries.json` (25 queries)
- Generates responses from **both bots** (Baseline & Sentiment-Aware)
- Saves to `llm_results/evaluation_results.csv`
- ⏱️ **Note:** Takes ~10 min due to API rate limits (20s delay per query)

#### Step 2: Compute Quantitative Metrics
```bash
python evaluate_ablation.py --eval_csv evaluation_results.csv --output_csv eval_metrics_detailed.csv
```
- Computes grounding score, hallucination rate, policy compliance, etc.
- Saves detailed per-query metrics to `llm_results/eval_metrics_detailed.csv`
- Prints aggregate summary table to console

**Sample Output:**
```
=== Aggregate Summary ===
Variant  Sentiment  n   Tokens  Grounding  PolicyCompliance%
A        positive   9   42.1    0.066      44.4
B        positive   9   89.3    0.101      55.6
A        negative   9   40.2    0.065      33.3
B        negative   9   58.7    0.084      55.6
...
```

---

## 👥 Team

**DSA4213 Group 21 - National University of Singapore**

Team Members: Goh Jia Yi, Lin Jiaying Melinda, Nathanael Lim Jun Wei

## 📚 References

### Core Research
1. **Tsang, W. (2022).** *NLP-based Restaurant Recommendation System using LSTM Ensemble Models*. Hong Kong restaurant reviews with Word2Vec/GloVe embeddings.
