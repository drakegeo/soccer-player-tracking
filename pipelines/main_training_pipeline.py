"""
Training pipeline — trains a YOLOv8 model on the prepared dataset.

Usage:
  python pipelines/main_training_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.config import DATASET_DIR, MODELS_DIR, MODEL_SIZE, EPOCHS, IMG_SIZE, BATCH_SIZE
from src.model_training.train import train_model


def run_training_pipeline() -> None:
    print("=== Training YOLOv8 ===")
    weights = train_model(
        dataset_yaml=str(DATASET_DIR / "dataset.yaml"),
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        output_dir=str(MODELS_DIR),
    )
    print(f"\nModel ready -> {weights}")
    print("Next: python pipelines/main_inference_pipeline.py")


if __name__ == "__main__":
    run_training_pipeline()
