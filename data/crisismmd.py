import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_crisismmd_annotations(root: str) -> pd.DataFrame:
    """Load and merge all CrisisMMD annotation .tsv files."""
    ann_dir = os.path.join(root, "annotations")
    dfs = []
    for fname in os.listdir(ann_dir):
        if fname.endswith(".tsv"):
            event_name = fname.replace(".tsv", "")
            df = pd.read_csv(os.path.join(ann_dir, fname), sep="\t", encoding="latin1")
            df["disaster_event"] = event_name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_fusion_dataframe(combined: pd.DataFrame) -> pd.DataFrame:
    """Keep necessary columns and build fusion labels."""
    df = combined[["tweet_id", "image_id", "text_info", "image_info", "tweet_text", "image_path"]].copy()
    df["text_label"] = (df["text_info"] == "informative").astype(int)
    df["image_label"] = (df["image_info"] == "informative").astype(int)
    df["label"] = ((df["text_label"] == 1) | (df["image_label"] == 1)).astype(int)
    return df


def split_train_test(df: pd.DataFrame, test_size=0.2, seed=42):
    return train_test_split(df, test_size=test_size, random_state=seed)


def make_train_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_eval_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CrisisVisionDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.df["label"] = (self.df["image_info"] == "informative").astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.loc[idx, "image_path"]
        img_path = os.path.join(self.root_dir, rel_path)

        try:
            image = Image.open(img_path)
            # Handle palette/alpha correctly
            if image.mode == "P":
                # Palette with possible transparency
                image = image.convert("RGBA")
            if image.mode == "RGBA":
                # Composite on white background (or black, your choice)
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # 3 = alpha channel
                image = background
            else:
                # Most images: just ensure RGB
                image = image.convert("RGB")
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        label = int(self.df.loc[idx, "label"])
        return image, torch.tensor(label, dtype=torch.long)


class MultimodalCrisisDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, image_transform=None, max_length=128):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir   # "./CrisisMMD_v2.0", not ".../data_image/"
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.loc[idx, "image_path"]
        img_path = os.path.join(self.root_dir, rel_path)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        if self.image_transform:
            image = self.image_transform(image)

        text = str(self.df.loc[idx, "tweet_text"])
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        label = int(self.df.loc[idx, "label"])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }