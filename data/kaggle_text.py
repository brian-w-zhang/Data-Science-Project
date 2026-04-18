import re
import pandas as pd
from datasets import Dataset

def clean_tweet(text: str) -> str:
    """Remove URLs, mentions, non-ASCII characters, and excess whitespace."""
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                  # @mentions
    text = re.sub(r"[^\x00-\x7F]+", "", text)         # non-ASCII (emojis, â, €, ™, etc.)
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text

def load_kaggle_disaster_csv(path: str) -> Dataset:
    df = pd.read_csv(path)
    df = df[["text", "target"]].dropna()
    df = df.rename(columns={"target": "label"})
    return Dataset.from_pandas(df)