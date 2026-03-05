import torch
import numpy as np
import cv2


class EmbeddingGenerator:
    def __init__(self, model):
        self.model = model

    def preprocess(self, image):
        image = cv2.resize(image, (112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # normalize to [-1,1]
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image, dtype=torch.float32)

    def l2_normalize(self, embedding):
        return embedding / np.linalg.norm(embedding)

    def generate(self, image):
        input_tensor = self.preprocess(image)

        with torch.no_grad():
            embedding = self.model(input_tensor)

        embedding = embedding.cpu().numpy().flatten()
        embedding = self.l2_normalize(embedding)

        return embedding