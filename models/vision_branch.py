import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights

class VisionDatasetWrapper(Dataset):
    """Wraps a base vision dataset so each item is ((image,), label)."""
    def __init__(self, base_ds):
        self.ds = base_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        image, label = self.ds[idx]
        return (image,), label


def build_resnet_classifier(num_classes=2, pretrained=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_vision_encoder(weights_path: str, device):
    """
    Load ResNet50 backbone with pretrained weights, return model with fc=Identity.
    """
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    # amputate head
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model