from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class ImageEmbedder:
    def __init__(self, model_name):
        #"openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, image_paths):
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings.cpu().numpy()

class LlavaImageEmbedder:
    def __init__(self, model_name="llava"):
        self.model_name = model_name  # optional: store for logging/debugging

    def embed(self, image_path):
        # LLaVA doesn't need this, just return the path or raw bytes
        return image_path  # or open(image_path, "rb").read() if you prefer

def load_image_embedder(config):
    return ImageEmbedder(model_name = 'openai/clip-vit-base-patch32')