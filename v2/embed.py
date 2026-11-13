"""Frozen feature extractors.

ClipEmbedder: OpenAI CLIP ViT-B/16 vision tower (Hugging Face transformers), pooled
output, 768-d. Nothing in the backbone is trained.

DummyEmbedder: fixed random projection of a downsampled crop, used only to test
pipeline plumbing without a model download. Its features carry no useful signal.
"""

from __future__ import annotations

import os

import os

import cv2
import numpy as np


class ClipEmbedder:
    name = "clip-vit-base-patch16"

    def __init__(self, model_id: str = "openai/clip-vit-base-patch16", device: str | None = None):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # LOW_MEM=1: bfloat16 weights and small batches for memory-limited servers.
        # Training features were extracted in fp16 on GPU, so this is approximate.
        self.low_mem = os.environ.get("LOW_MEM", "0") == "1" and self.device == "cpu"
        self.batch = 4 if self.low_mem else 64
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        if self.low_mem:
            torch.set_num_threads(2)
            self.model = CLIPVisionModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            ).eval()
        else:
            self.model = CLIPVisionModel.from_pretrained(model_id).to(self.device).eval()
            if self.device == "cuda":
                self.model = self.model.half()
        self.dim = self.model.config.hidden_size

    def __call__(self, crops_rgb: np.ndarray) -> np.ndarray:
        """crops_rgb: (n, H, W, 3) uint8 -> (n, dim) float32."""
        torch = self.torch
        outs = []
        for i in range(0, len(crops_rgb), self.batch):
            inputs = self.processor(images=list(crops_rgb[i:i + self.batch]), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            if self.device == "cuda":
                pixel_values = pixel_values.half()
            elif self.low_mem:
                pixel_values = pixel_values.to(torch.bfloat16)
            with torch.no_grad():
                out = self.model(pixel_values=pixel_values)
            outs.append(out.pooler_output.float().cpu().numpy())
        return np.concatenate(outs, axis=0)


class DummyEmbedder:
    name = "dummy-test-only"
    dim = 64

    def __init__(self):
        rng = np.random.RandomState(0)
        self.proj = rng.randn(8 * 8 * 3, self.dim).astype(np.float32) / 10.0

    def __call__(self, crops_rgb: np.ndarray) -> np.ndarray:
        small = np.stack([cv2.resize(c, (8, 8), interpolation=cv2.INTER_AREA) for c in crops_rgb])
        flat = small.reshape(len(crops_rgb), -1).astype(np.float32) / 255.0
        return flat @ self.proj


def get_embedder(kind: str):
    if kind == "clip":
        return ClipEmbedder()
    if kind == "dummy":
        return DummyEmbedder()
    raise ValueError(f"unknown embedder: {kind}")
