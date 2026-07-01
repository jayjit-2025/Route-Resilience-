#!/usr/bin/env python3
"""Download pretrained model weights for Route Resilience."""
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("Note: Route Resilience uses torchvision pretrained models.")
print("Weights are downloaded automatically on first model use.")
print("For custom fine-tuned weights, place .pth files in the models/ directory.")
print("✅ Models directory ready at:", MODELS_DIR.absolute())
