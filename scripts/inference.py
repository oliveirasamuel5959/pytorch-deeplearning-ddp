#!/usr/bin/env python
"""CLI entrypoint for running inference with a trained checkpoint.

Usage:
    uv run scripts/infer.py --checkpoint outputs/emnist_cnn/checkpoints/best.pt --image digit.png
"""

from __future__ import annotations

import argparse
import json
import sys

from deeplearning.config import ModelConfig
from deeplearning.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image with a trained EMNIST model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="simple_cnn", help="Model registry key used at training time")
    parser.add_argument("--num-classes", type=int, default=62)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ModelConfig(name=args.model, num_classes=args.num_classes)

    predictor = Predictor(checkpoint_path=args.checkpoint, model_cfg=model_cfg, device=args.device)
    result = predictor.predict(args.image)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
