import random
import shutil
import yaml
from pathlib import Path


def split_dataset(
    frames_dir: str,
    labels_dir: str,
    output_dir: str,
    class_names: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Split labeled frames into train/val/test and write dataset.yaml.

    Args:
        frames_dir: Directory containing .jpg frames.
        labels_dir: Directory containing YOLO .txt label files (same stem as frames).
        output_dir: Root output directory for the YOLO dataset.
        class_names: Ordered list of class names matching label IDs.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.
    """
    frames_dir = Path(frames_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    labeled = [
        f for f in sorted(frames_dir.glob("*.jpg"))
        if (labels_dir / f.stem).with_suffix(".txt").exists()
    ]

    if not labeled:
        raise FileNotFoundError(
            f"No labeled frames found. Ensure .txt files exist in {labels_dir}."
        )

    random.seed(seed)
    random.shuffle(labeled)

    n = len(labeled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": labeled[:n_train],
        "val": labeled[n_train: n_train + n_val],
        "test": labeled[n_train + n_val:],
    }

    for split, files in splits.items():
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            shutil.copy(img_path, img_out / img_path.name)
            label_path = (labels_dir / img_path.stem).with_suffix(".txt")
            shutil.copy(label_path, lbl_out / label_path.name)

        print(f"  {split}: {len(files)} samples")

    _write_dataset_yaml(output_dir, class_names)


def _write_dataset_yaml(output_dir: Path, class_names: list[str]) -> None:
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"dataset.yaml written -> {yaml_path}")
