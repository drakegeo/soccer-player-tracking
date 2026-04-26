from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths
VIDEO_PATH = ROOT / "data" / "video" / "sample.mp4"
FRAMES_DIR = ROOT / "data" / "frames"
DATASET_DIR = ROOT / "data" / "dataset"
MODELS_DIR = ROOT / "models"

# Frame extraction
EXTRACTION_FPS = 2.0
MAX_FRAMES = 80

# Dataset
CLASS_NAMES = ["white_team", "red_team", "referee"]
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

# Training
MODEL_SIZE = "yolov8m"
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 4

# Inference
CONF_THRESHOLD = 0.6
