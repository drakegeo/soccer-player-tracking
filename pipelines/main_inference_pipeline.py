"""
Inference pipeline — loads the trained model and runs player tracking analytics on a video.

Outputs:
  - models/output_annotated.mp4  → video with bounding boxes and per-frame counts
  - models/player_counts.csv     → per-frame player counts per team

Usage:
  python pipelines/main_inference_pipeline.py
  python pipelines/main_inference_pipeline.py --source data/video/sample.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.config import MODELS_DIR, CLASS_NAMES, CONF_THRESHOLD, VIDEO_PATH
from src.model_training.inference import count_players_per_frame


def run_inference_pipeline(source: str) -> None:
    best_weights = MODELS_DIR / "soccer_player_detector" / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"No trained model found at {best_weights}")
        print("Run python pipelines/main_training_pipeline.py first.")
        return

    print("=== Running inference ===")
    print(f"Model  : {best_weights}")
    print(f"Source : {source}")

    count_players_per_frame(
        model_path=str(best_weights),
        source=source,
        class_names=CLASS_NAMES,
        conf_threshold=CONF_THRESHOLD,
        output_video_path=str(MODELS_DIR / "output_annotated.avi"),
        output_csv_path=str(MODELS_DIR / "player_counts.csv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and player count analytics")
    parser.add_argument(
        "--source",
        default=str(VIDEO_PATH),
        help="Video file or image directory (default: config VIDEO_PATH)",
    )
    args = parser.parse_args()
    run_inference_pipeline(args.source)


if __name__ == "__main__":
    main()
