import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_yelp_data():
    print("Loading dataset...")

    ds = load_dataset("Johnnyeee/Yelpdata_663")
    df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)
    print(f"Loaded {len(df)} rows")

    if "categories" in df.columns:
        df = df[df["categories"].str.contains("restaurant", case=False, na=False)]
        print(f"Filtered dataset to only retain restaurant reviews: {len(df)} records")

    df["sentiment"] = df["stars_x"].apply(
        lambda x: 0 if x < 3 else (1 if x == 3 else 2)
    )

    df = df[["text", "sentiment"]].dropna()

    df_balanced = pd.concat(
        [
            df[df["sentiment"] == 0].sample(n=100000, random_state=42),
            df[df["sentiment"] == 1].sample(n=100000, random_state=42),
            df[df["sentiment"] == 2].sample(n=100000, random_state=42),
        ]
    )

    train, temp = train_test_split(
        df_balanced, test_size=0.3, stratify=df_balanced["sentiment"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["sentiment"], random_state=42
    )

    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print(f"Saved train ({len(train)}), val ({len(val)}), test ({len(test)})")

    return train, val, test


if __name__ == "__main__":
    load_yelp_data()
