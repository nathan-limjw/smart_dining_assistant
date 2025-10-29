import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import pipeline


def plot_confusion_matrix(
    y_true, y_pred, output_path="results/figures/confusion_matrix.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix for Sentiment Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return f"Confusion matrix saved to {output_path}"


def evaluate_model(model_path="models/sentiment_model", test_path="data/test.csv"):
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
    )

    test_df = pd.read_csv(test_path)

    print("\nMaking predictions on test data...")
    predictions = []
    confidences = []

    for text in test_df["text"]:
        result = classifier(text[:512])[0]
        best = max(result, key=lambda x: x["score"])

        label_num = int(best["label"].split("_")[1])
        predictions.append(label_num)
        confidences.append(best["score"])

    test_df["predicted"] = predictions
    test_df["confidence"] = confidences

    y_true = test_df["sentiment"]
    y_pred = test_df["predicted"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print("\n Metrics for Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nPer-Class Performance:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]
        )
    )

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv("results/metrics/test_metrics.csv", index=False)
    print("\nEvaluation metrics saved to 'results/metrics/test_metrics.csv'")

    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(y_true, y_pred, average=None)
    )
    per_class_df = pd.DataFrame(
        {
            "Class": ["Negative", "Neutral", "Positive"],
            "Precision": precision_per_class,
            "Recall": recall_per_class,
            "F1-Score": f1_per_class,
            "Support": support_per_class,
        }
    )
    per_class_df.to_csv("results/metrics/per_class_metrics.csv", index=False)
    print("Per-class metrics saved to 'results/metrics/per_class_metrics.csv'")

    plot_confusion_matrix(y_true, y_pred)

    test_df.to_csv("results/metrics/test_predictions.csv", index=False)
    print("Predictions saved to 'results/metrics/test_predictions.csv'")

    errors = test_df[test_df["sentiment"] != test_df["predicted"]]
    error_rate = len(errors) * 100 / len(test_df)
    print(f"Error percentage: {error_rate:.2%}")

    if len(errors) > 0:
        errors.to_csv("results/metrics/error_examples.csv", index=False)
        print("Errors saved to 'results/metrics/error_examples.csv'")

    return metrics_dict
