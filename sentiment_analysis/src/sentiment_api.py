from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self, model_path="sentiment_analysis/models/sentiment_model"):
        """
        Loads the trained sentiment model
        """
        print("Loading sentiment model...")
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            return_all_scores=True,
        )
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        print("Model has been loaded!")

    def analyze(self, text):
        """
        Analyzes the sentiment of a single text and returns the sentiment ('negative',
        'neutral', 'positive') and the confidence score
        """

        result = self.classifier(text)[0]
        best = max(result, key=lambda x: x["score"])
        label_num = int(best["label"].split("_")[1])

        return {
            "sentiment": self.label_map[label_num],
            "confidence": best["score"],
            "label_number": label_num,
        }

    def analyze_reviews(self, reviews_df):
        """
        Analyzes a dataframe of reviews from the 'text' column and outputs a dataframe with
        added sentiment columns
        """

        # print(f"Analyzing all {len(reviews_df)} reviews...")
        results = [self.analyze(text) for text in reviews_df["text"].tolist()]

        reviews_df["sentiment"] = [result["sentiment"] for result in results]
        reviews_df["sentiment_confidence"] = [
            result["confidence"] for result in results
        ]

        return reviews_df