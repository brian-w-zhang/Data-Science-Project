import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased"


class TextDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        inputs = (item["input_ids"], item["attention_mask"])
        label  = item["label"]
        return inputs, label


class TextClassificationWrapper(nn.Module):
    """
    Wraps a HuggingFace AutoModelForSequenceClassification
    so that forward(*inputs) returns logits directly.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids, attention_mask):
        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


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