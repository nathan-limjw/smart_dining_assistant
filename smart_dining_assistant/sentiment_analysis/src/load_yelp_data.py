import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_data():
    print("Loading dataset...")
    ds = load_dataset("Johnnyeee/Yelpdata_663")

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    train_df = train_df[
        train_df["categories"].str.contains("restaurant", case=False, na=False)
    ]
    test_df = test_df[
        test_df["categories"].str.contains("restaurant", case=False, na=False)
    ]
    print(
        f"Filtered dataset to only retain restaurant reviews: train = {len(train_df)}, test = {len(test_df)} "
    )

    train_df["sentiment"] = train_df["stars_x"].apply(
        lambda x: 0 if x < 3 else (1 if x == 3 else 2)
    )
    test_df["sentiment"] = test_df["stars_x"].apply(
        lambda x: 0 if x < 3 else (1 if x == 3 else 2)
    )

    train_df = train_df[["text", "sentiment"]].dropna()
    test_df = test_df[["text", "sentiment"]].dropna()

    min_train_samples = train_df["sentiment"].value_counts().min()
    samples_per_class = min(100000, min_train_samples)

    print(f"Sampling {samples_per_class} from each class")

    train_balanced = pd.concat(
        [
            train_df[train_df["sentiment"] == 0].sample(
                n=samples_per_class, random_state=42
            ),
            train_df[train_df["sentiment"] == 1].sample(
                n=samples_per_class, random_state=42
            ),
            train_df[train_df["sentiment"] == 2].sample(
                n=samples_per_class, random_state=42
            ),
        ]
    )
    train_final, val = train_test_split(
        train_balanced,
        test_size=0.2,
        stratify=train_balanced["sentiment"],
        random_state=42,
    )

    min_train_samples = test_df["sentiment"].value_counts().min()
    test_samples_per_class = min(15000, min_train_samples)

    test_balanced = pd.concat(
        [
            test_df[test_df["sentiment"] == 0].sample(
                n=test_samples_per_class, random_state=42
            ),
            test_df[test_df["sentiment"] == 1].sample(
                n=test_samples_per_class, random_state=42
            ),
            test_df[test_df["sentiment"] == 2].sample(
                n=test_samples_per_class, random_state=42
            ),
        ]
    )

    train_final.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test_balanced.to_csv("data/test.csv", index=False)

    print("Data saved to /data")

    return train_final, val, test_balanced


if __name__ == "__main__":
    load_data()
