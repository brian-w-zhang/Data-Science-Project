import pandas as pd
from datasets import Dataset
from data.text_cleaner import clean_tweet

def load_kaggle_disaster_csv(path: str) -> Dataset:
    df = pd.read_csv(path)
    df = df[["text", "target"]].dropna()

    # Clean tweets
    df["text"] = df["text"].apply(clean_tweet)

    df = df.rename(columns={"target": "label"})
    return Dataset.from_pandas(df)