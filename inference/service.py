import io
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

from models.fusion_model import MultimodalFusionNetwork
from data.crisismmd import make_eval_transforms


class DisasterClassifier:
    def __init__(
        self,
        text_model_dir: str,
        vision_weights_path: str,
        fusion_weights_path: str,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_dir)
        self.img_transform = make_eval_transforms()

        self.model = MultimodalFusionNetwork(
            text_model_dir=text_model_dir,
            vision_weights_path=vision_weights_path,
            device=self.device,
        ).to(self.device)

        state = torch.load(fusion_weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.max_length = max_length
        self.class_names = ["not_disaster", "disaster"]

    @torch.inference_mode()
    def predict(self, text: Optional[str], image_bytes: Optional[bytes]) -> Dict[str, Any]:
        if not text and not image_bytes:
            raise ValueError("At least one of text or image must be provided.")

        batch = {}

        if text:
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["input_ids"] = enc["input_ids"].to(self.device)
            batch["attention_mask"] = enc["attention_mask"].to(self.device)
        else:
            # Dummy text for shape; or adjust model to handle None
            enc = self.tokenizer(
                "",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["input_ids"] = enc["input_ids"].to(self.device)
            batch["attention_mask"] = enc["attention_mask"].to(self.device)

        if image_bytes:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_tensor = self.img_transform(pil_img).unsqueeze(0).to(self.device)
        else:
            # Dummy image; for a real project you might want a text-only mode
            dummy = Image.new("RGB", (224, 224), color="black")
            img_tensor = self.img_transform(dummy).unsqueeze(0).to(self.device)

        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=img_tensor,
        )
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(probs.argmax())
        return {
            "label": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probs": {self.class_names[i]: float(p) for i, p in enumerate(probs)},
        }