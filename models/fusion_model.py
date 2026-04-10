import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50


class FusionDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps MultimodalCrisisDataset so each item is:
      ((input_ids, attention_mask, image), label)
    which matches the expectation of train_one_epoch / evaluate.
    """
    def __init__(self, base_ds):
        self.ds = base_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        inputs = (item["input_ids"], item["attention_mask"], item["image"])
        label  = item["label"]
        return inputs, label


class MultimodalFusionNetwork(nn.Module):
    def __init__(self, text_model_dir: str, vision_weights_path: str, device: str):
        super().__init__()
        # Text encoder
        self.text_extractor = AutoModel.from_pretrained(text_model_dir)
        for p in self.text_extractor.parameters():
            p.requires_grad = False

        # Vision encoder
        self.vision_extractor = resnet50(weights=None)
        self.vision_extractor.fc = nn.Linear(2048, 2)
        state = torch.load(vision_weights_path, map_location=device)
        self.vision_extractor.load_state_dict(state)
        self.vision_extractor.fc = nn.Identity()
        for p in self.vision_extractor.parameters():
            p.requires_grad = False

        self.fusion_classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, input_ids, attention_mask, images):
        text_output = self.text_extractor(input_ids=input_ids, attention_mask=attention_mask)
        context = text_output.last_hidden_state[:, 0, :]  # [B, 768]
        evidence = self.vision_extractor(images)          # [B, 2048]
        x = torch.cat([context, evidence], dim=1)         # [B, 2816]
        return self.fusion_classifier(x)