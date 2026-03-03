"""Image embedding using CLIP / open_clip."""

from pathlib import Path

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from configs.default import IMAGE_EMBED_MODEL

_CLIP_MODEL_MAP = {
    "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
    "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
}


class ImageEmbedder:
    """Wraps an open_clip model for image embedding."""

    def __init__(self, model_name: str = IMAGE_EMBED_MODEL):
        arch, pretrained = _CLIP_MODEL_MAP.get(model_name, ("ViT-B-32", "openai"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

    @torch.no_grad()
    def embed_image(self, image_path: str | Path) -> list[float]:
        """Embed a single image file."""
        img = Image.open(str(image_path)).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    @torch.no_grad()
    def embed_images(self, image_paths: list[str | Path], batch_size: int = 16) -> list[list[float]]:
        """Embed a batch of image files."""
        all_embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
            batch_paths = image_paths[i : i + batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(str(p)).convert("RGB")
                    tensors.append(self.preprocess(img))
                except Exception as e:
                    print(f"Skipping {p}: {e}")
                    tensors.append(self.preprocess(Image.new("RGB", (224, 224))))

            batch = torch.stack(tensors).to(self.device)
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.extend(features.cpu().tolist())

        return all_embeddings

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Embed text using CLIP's text encoder (for cross-modal retrieval)."""
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()
