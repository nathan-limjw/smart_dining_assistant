import os

import pandas as pd
from src import SentimentAnalyzer, evaluate_model, load_data, run_hyperparameter_tuning

os.makedirs("data", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

train, val, test = load_data()
run_hyperparameter_tuning()
evaluate_model()


# TEST
analyzer = SentimentAnalyzer()
restaurant_reviews = pd.DataFrame(
    {
        "text": [
            "The food was delicious, totally would come back!",
            "The waiter ignored us the entire time while we raised our hand",
            "Food was good but the wait was long",
            "I think the service was overall okay",
            "There were rats running around the kitchen..",
        ]
    }
)


result = analyzer.analyze_reviews(restaurant_reviews)
print(result)
