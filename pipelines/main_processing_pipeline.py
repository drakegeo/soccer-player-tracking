"""
Processing pipeline — converts labeled frames into a YOLO-ready dataset.

Steps:
  1. Convert labelme JSON labels in data/frames/ to YOLO TXT format
  2. Split frames + labels into train / val / test
  3. Write data/dataset/dataset.yaml

Run after labeling all frames with labelme:
  python pipelines/main_processing_pipeline.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.config import FRAMES_DIR, DATASET_DIR, CLASS_NAMES, TRAIN_RATIO, VAL_RATIO
from src.data_processing.labelme_to_yolo import convert_labelme_to_yolo
from src.data_processing.prepare_dataset import split_dataset


def run_processing_pipeline() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        print(f"Cleared {DATASET_DIR}")

    print("=== Step 1: Converting labelme JSON labels to YOLO TXT ===")
    n = convert_labelme_to_yolo(str(FRAMES_DIR), CLASS_NAMES)
    if n == 0:
        print("No labelme JSON files found — label your frames with labelme first.")
        return

    print("\n=== Step 2: Splitting into train / val / test ===")
    split_dataset(
        frames_dir=str(FRAMES_DIR),
        labels_dir=str(FRAMES_DIR),
        output_dir=str(DATASET_DIR),
        class_names=CLASS_NAMES,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    print(f"\nDataset ready in {DATASET_DIR}")
    print("Next: python pipelines/main_training_pipeline.py --step train")


if __name__ == "__main__":
    run_processing_pipeline()
