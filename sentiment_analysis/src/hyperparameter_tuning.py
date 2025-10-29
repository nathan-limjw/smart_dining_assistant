import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def load_and_tokenize_data():
    print("Loading data splits...")
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df = train_df[:20]
    val_df = val_df[:20]
    test_df = test_df[:20]

    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(input):
        return tokenizer(
            input["text"], padding="max_length", truncation=True, max_length=512
        )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("Tokenizing all datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("sentiment", "labels")
    val_dataset = val_dataset.rename_column("sentiment", "labels")
    test_dataset = test_dataset.rename_column("sentiment", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Data tokenized!")
    return train_dataset, val_dataset, test_dataset


def run_hyperparameter_tuning():
    print("Finding the best hyperparameters...")

    configs = [
        {
            "name": "config_1_default",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 3,
            "weight_decay": 0.01,
        },
        {
            "name": "config_2_higher_lr",
            "learning_rate": 3e-5,
            "batch_size": 16,
            "epochs": 3,
            "weight_decay": 0.01,
        },
        {
            "name": "config_3_lower_lr_more_epochs",
            "learning_rate": 1e-5,
            "batch_size": 16,
            "epochs": 5,
            "weight_decay": 0.01,
        },
        {
            "name": "config_4_larger_batch",
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "weight_decay": 0.01,
        },
        {
            "name": "config_5_lower_weight_decay",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 3,
            "weight_decay": 0.001,
        },
    ]

    train_dataset, val_dataset, test_dataset = load_and_tokenize_data()

    results = []

    for i, config in enumerate(configs):
        print(f"Current Config: {config['name']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Weight Decay: {config['weight_decay']}")

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )

        training_args = TrainingArguments(
            output_dir=f"models/{config['name']}",
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=32,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            warmup_steps=500,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=f"results/logs/{config['name']}",
            logging_steps=100,
            save_total_limit=2,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        print("\nTraining with configs...")
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60

        print("\nEvaluating on validation dataset...")
        eval_results = trainer.evaluate()

        result_entry = {
            "config_name": config["name"],
            "config_number": i + 1,
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "weight_decay": config["weight_decay"],
            "val_accuracy": eval_results["eval_accuracy"],
            "val_precision": eval_results["eval_precision"],
            "val_recall": eval_results["eval_recall"],
            "val_f1": eval_results["eval_f1"],
            "val_loss": eval_results["eval_loss"],
            "training_time_minutes": training_time,
        }

        results.append(result_entry)

        print("\nTraining Results:")
        print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"Time: {training_time:.1f} mins")

    os.makedirs("results/metrics", exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_accuracy", ascending=False)
    results_df.to_csv("results/metrics/hyperparam_tuning_results.csv", index=False)

    print("Best configuration for Model!")
    best_config = results_df.iloc[0]
    print(f"Config: {best_config['config_name']}")
    print(f"Accuracy: {best_config['val_accuracy']:.4f}")

    best_config_dict = {
        "config_name": best_config["config_name"],
        "learning_rate": float(best_config["learning_rate"]),
        "batch_size": int(best_config["batch_size"]),
        "epochs": int(best_config["epochs"]),
        "weight_decay": float(best_config["weight_decay"]),
        "val_accuracy": float(best_config["val_accuracy"]),
        "val_f1": float(best_config["val_f1"]),
    }

    with open("results/metrics/best_config.json", "w") as f:
        json.dump(best_config_dict, f, indent=4)

    source_dir = f"models/{best_config['config_name']}"
    dest_dir = "models/sentiment_model"

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    shutil.copytree(source_dir, dest_dir)

    print("Best model has been saved under: models/sentiment_model")
    print(
        "Results have been saved under 'results/metrics/hyperparam_tuning_results.csv'"
    )


if __name__ == "__main__":
    run_hyperparameter_tuning()
