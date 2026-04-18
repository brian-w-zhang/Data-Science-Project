import re

def clean_tweet(text: str) -> str:
    # Handle NaN / non-string
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 2. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 3. Remove non-ASCII chars (emojis, weird encodings)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # 4. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
