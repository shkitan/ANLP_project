import os
from pathlib import Path

MODELS_DIR = os.path.join(os.getcwd(), "models")
PRETRAINED_MODELS = os.path.join(MODELS_DIR, "pretrained")
FINETUNED_MODELS = os.path.join(MODELS_DIR, "finetuned")


EMBEDDING_DIR = Path(os.path.join(os.getcwd(), "embedding"))
DATASETS_DIR = Path(os.path.join(os.getcwd(), "datasets"))