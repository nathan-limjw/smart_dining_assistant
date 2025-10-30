import os

from src import SentimentAnalyzer, evaluate_model, load_data, run_hyperparameter_tuning

os.makedirs("data", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

train, val, test = load_data()
run_hyperparameter_tuning()
evaluate_model()


# TEST
analyzer = SentimentAnalyzer()
review = "This pasta is so bombs, I love it"

result = analyzer.analyze(review)
print(f"Review: {review}")
print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
