import torch
import os


class ModelLoader:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        model = torch.load(self.model_path, map_location=self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return model

    def get_model(self):
        return self.model