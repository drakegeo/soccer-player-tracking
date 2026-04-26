"""
Labeling pipeline — extracts frames from video and prepares them for labeling.

  extract  →  sample frames from the video into data/frames/

Label with: labelme data/frames --labels data/frames/predefined_classes.txt --nodata --autosave
Labels save as JSON alongside frames in data/frames/ and reload automatically.
Once all frames are labeled run: python pipelines/main_processing_pipeline.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.config import VIDEO_PATH, FRAMES_DIR, CLASS_NAMES, EXTRACTION_FPS, MAX_FRAMES
from src.data_labeling.video_to_frames import extract_frames
from src.data_labeling.prepare_dataset import write_predefined_classes  # labeling helper stays in data_labeling


def step_extract() -> None:
    print("=== Extracting frames ===")
    existing = list(Path(FRAMES_DIR).glob("*.jpg"))
    if existing:
        print(f"Frames already exist ({len(existing)} found in {FRAMES_DIR}) — skipping extraction.")
    else:
        n = extract_frames(VIDEO_PATH, FRAMES_DIR, fps=EXTRACTION_FPS, max_frames=MAX_FRAMES)
        print(f"Done — {n} frames ready in {FRAMES_DIR}")
    write_predefined_classes(FRAMES_DIR, CLASS_NAMES)
    print("Next: labelme data/frames --labels data/frames/predefined_classes.txt --nodata --autosave")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames and prepare for labeling")
    parser.add_argument("--step", choices=["extract"], required=True)
    args = parser.parse_args()

    if args.step == "extract":
        step_extract()


if __name__ == "__main__":
    main()
