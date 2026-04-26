from pathlib import Path
from ultralytics import YOLO


def train_model(
    dataset_yaml: str,
    model_size: str = "yolov8n",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    output_dir: str = "models",
) -> str:
    """
    Train a YOLOv8 detection model on the prepared soccer dataset.

    Args:
        dataset_yaml: Path to dataset.yaml.
        model_size: YOLO variant — yolov8n (fastest) to yolov8m (most accurate).
        epochs: Number of training epochs.
        img_size: Input image resolution.
        batch_size: Training batch size.
        output_dir: Directory where run artefacts are saved.

    Returns:
        Path to best model weights (best.pt).
    """
    model = YOLO(f"yolo_models/{model_size}.pt")

    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir,
        name="soccer_player_detector",
        exist_ok=True,
    )

    best_weights = Path(output_dir) / "soccer_player_detector" / "weights" / "best.pt"
    print(f"Training complete. Best weights -> {best_weights}")
    return str(best_weights)
