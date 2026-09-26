"""Inference-time wrapper: load a trained checkpoint and predict on new images."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from deeplearning.config import ModelConfig
from deeplearning.models.cnn_mlp import build_model
from deeplearning.utils.checkpoint import load_checkpoint
from deeplearning.utils.device import resolve_device

# Same normalization/orientation fix used at training time (see datasets/emnist.py).
class _TransposeEMNIST:
  """Fixes EMNIST's transposed image orientation (picklable, unlike a lambda)."""
  def __call__(self, img):
    return img.transpose(2, 1)

_FIX_ORIENTATION = _TransposeEMNIST()

_MEAN, _STD = (0.1751,), (0.3332,)

_PREPROCESS = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        _FIX_ORIENTATION,
        transforms.Normalize(_MEAN, _STD),
    ]
)


class Predictor:
    """Loads a model checkpoint once and exposes `.predict(image_path)` /
    `.predict_batch(tensor)` for reuse across many inputs.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_cfg: ModelConfig,
        device: str = "auto",
        class_names: list[str] | None = None,
    ):
        self.device = resolve_device(device)
        self.model = build_model(model_cfg.input_channels, model_cfg.num_classes)
        load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.class_names = class_names

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        """Predict the class of a single image file (any format PIL can open)."""
        image = Image.open(image_path)
        tensor = _PREPROCESS(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())

        return {
            "predicted_class": self.class_names[pred_idx] if self.class_names else pred_idx,
            "confidence": float(probs[pred_idx].item()),
            "probabilities": probs.cpu().tolist(),
        }

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Predict class indices for a pre-batched, pre-normalized tensor [N, 1, 28, 28]."""
        images = images.to(self.device)
        logits = self.model(images)
        return logits.argmax(dim=1).cpu()
