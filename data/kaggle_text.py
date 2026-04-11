import pandas as pd
from datasets import Dataset

def load_kaggle_disaster_csv(path: str) -> Dataset:
    df = pd.read_csv(path)
    df = df[["text", "target"]].dropna()
    df = df.rename(columns={"target": "label"})
    return Dataset.from_pandas(df)