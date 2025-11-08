# Sentiment Analysis Module

## Overivew
This module implements sentiment classification for user queries and restaurant reviews using a fine-tuned DistilBERT model. The classifier detects three sentiment classes (Positive, Neutral, Negative) and outputs confidence scores used in the retrieval system for emotionally-aligned recommendations.

## Directory Structure
```
📦sentiment_analysis
 ┣ 📂data
 ┃ ┣ 📜test.csv
 ┃ ┣ 📜train.csv
 ┃ ┗ 📜val.csv
 ┣ 📂models
 ┃ ┣ 📂sentiment_model
 ┃ ┃ ┣ 📂checkpoint-30000
 ┃ ┃ ┃ ┣ 📜config.json
 ┃ ┃ ┃ ┣ 📜model.safetensors
 ┃ ┃ ┃ ┣ 📜optimizer.pt
 ┃ ┃ ┃ ┣ 📜rng_state.pth
 ┃ ┃ ┃ ┣ 📜scheduler.pt
 ┃ ┃ ┃ ┣ 📜trainer_state.json
 ┃ ┃ ┃ ┗ 📜training_args.bin
 ┃ ┃ ┣ 📂checkpoint-45000
 ┃ ┃ ┃ ┣ 📜config.json
 ┃ ┃ ┃ ┣ 📜model.safetensors
 ┃ ┃ ┃ ┣ 📜optimizer.pt
 ┃ ┃ ┃ ┣ 📜rng_state.pth
 ┃ ┃ ┃ ┣ 📜scheduler.pt
 ┃ ┃ ┃ ┣ 📜trainer_state.json
 ┃ ┃ ┃ ┗ 📜training_args.bin
 ┃ ┃ ┣ 📜config.json
 ┃ ┃ ┣ 📜model.safetensors
 ┃ ┃ ┣ 📜special_tokens_map.json
 ┃ ┃ ┣ 📜tokenizer_config.json
 ┃ ┃ ┣ 📜training_args.bin
 ┃ ┃ ┗ 📜vocab.txt
 ┃ ┗ 📜.DS_Store
 ┣ 📂results
 ┃ ┣ 📂hyperparameter_tuning
 ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┗ 📜confusion_matrix.png
 ┃ ┃ ┣ 📂metrics
 ┃ ┃ ┃ ┣ 📜best_config.json
 ┃ ┃ ┃ ┣ 📜error_examples.csv
 ┃ ┃ ┃ ┣ 📜hyperparam_tuning_results.csv
 ┃ ┃ ┃ ┣ 📜per_class_metrics.csv
 ┃ ┃ ┃ ┣ 📜test_metrics.csv
 ┃ ┃ ┃ ┗ 📜test_predictions.csv
 ┃ ┃ ┗ 📜.DS_Store
 ┃ ┣ 📂model_evaluation
 ┃ ┃ ┣ 📂figures
 ┃ ┃ ┃ ┗ 📜confusion_matrix.png
 ┃ ┃ ┣ 📂metrics
 ┃ ┃ ┃ ┣ 📜best_config.json
 ┃ ┃ ┃ ┣ 📜error_examples.csv
 ┃ ┃ ┃ ┣ 📜hyperparam_tuning_results.csv
 ┃ ┃ ┃ ┣ 📜per_class_metrics.csv
 ┃ ┃ ┃ ┣ 📜test_metrics.csv
 ┃ ┃ ┃ ┗ 📜test_predictions.csv
 ┃ ┃ ┗ 📜.DS_Store
 ┃ ┗ 📜.DS_Store
 ┣ 📂src
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜hyperparameter_tuning.py
 ┃ ┣ 📜load_yelp_data.py
 ┃ ┗ 📜sentiment_api.py
 ┣ 📜.DS_Store
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜main.py
 ┗ 📜requirements.txt
```

## Quick Start

### Installation
```bash
cd sentiment_analysis
pip3 install -r requirements.txt
```

### Run Complete Pipeline
```bash
python3 main.py
```

This code will:
1. Load and preprocess data
2. Run hyperparameter tuning
3. Train the best model 
4. Evaluate performance
5. Run example predictions

**Expected Output**:
- Data splits were saved under `data\`
- Hyperparameter tuning results were saved under `results/hyperparameter_tuning`
- Model evaluation results saved under `results/model_evaluation`
- Example predictions printed onto console

## Model Details

### Architecture
- **Base Model**: 'distilbert-base-uncased'
- **Task**: 3-Class sentiment classification
- **Finetuning**: Trained on 300,000 balanced Yelp reviews

### Sentiment Mapping
- **Negative**: `stars_y` < 3
- **Neutral**: `stars_y` = 3
- **Positive**: `stars_y` > 3

### Hyperparameters (Config 2 - Best Performance)
```python
learning_rate = 3e-5
num_epochs = 3
batch_size = 16
weight_decay = 0.01
```

## Model Performance

### Test Set Results (45,000 reviews)
| Metric | Score |
|--------|-------|
| Accuracy | 81.27% |
| Precision | 81.66% |
| Recall | 81.27% |
| F1 Score | 81.38% |

### Per Class Performance (F1 Scores)
| Class | F1 Score |
|--------|---------|
| Negative | 82.78% |
| Neutral | 73.98% |
| Positive | 87.39% |

### Key Insights
- **Positive-to-Negative confusion Rate: 0.7%**
- Model rarely makes extreme polarity errors
- Most confusion occurs between Neutral and other classes (expected for ambiguous 3-star reviews )

