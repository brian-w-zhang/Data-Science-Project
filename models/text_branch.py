from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased"


def create_text_classifier(num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    return model, tokenizer


def load_text_encoder(model_dir: str):
    """
    Load DistilBERT encoder from a fine-tuned HF model directory.
    Ignores classifier head.
    """
    encoder = AutoModel.from_pretrained(model_dir)
    return encoder


def load_text_classifier(model_dir: str):
    """
    Load the full fine-tuned classifier (for standalone text-only inference).
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer