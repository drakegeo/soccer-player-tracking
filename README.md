# Soccer Player Tracking

Detect and count soccer players per team in match footage using YOLOv8.

## Setup

Requires **Python 3.12** (PyTorch does not yet ship wheels for 3.13+).

```powershell
# Create and activate virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Labeling Tool — labelme

labelme saves annotations as JSON files alongside the frames.

### Install

```powershell
pip install labelme
```

### Launch

```powershell
labelme data/frames --labels data/frames/predefined_classes.txt
```

### Classes

| ID | Name        | Covers                          |
|----|-------------|---------------------------------|
| 0  | white_team  | All outfield + GK white team    |
| 1  | red_team    | All outfield + GK red team      |
| 2  | referee     | Referee(s)                      |

---

## Pipeline

### Step 1 — Extract frames and prepare for labeling

```powershell
python pipelines/main_labeling_pipeline.py --step extract
```

Samples the video at 2 fps into `data/frames/` and writes `predefined_classes.txt`.

### Step 2 — Label with labelme

```powershell
labelme data/frames --labels data/frames/predefined_classes.txt
```

Label all frames in `data/frames/`. Annotations autosave as `.json` files alongside each frame.

### Step 3 — Build the YOLO dataset

```powershell
python pipelines/main_processing_pipeline.py
```

Converts labelme JSON → YOLO TXT, clears `data/dataset/`, then splits labeled frames into `train/val/test` and writes `data/dataset/dataset.yaml`.

### Step 4 — Train

```powershell
python pipelines/main_training_pipeline.py --step train
```

### Step 5 — Inference (count players per frame)

```powershell
python pipelines/main_inference_pipeline.py
python pipelines/main_inference_pipeline.py --source data/video/sample.mp4
```

Outputs annotated video to `models/output_annotated.avi` and per-frame counts to `models/player_counts.csv`. Referees are detected but hidden from the output.

---

## Project Structure

```
soccer_player_tracking/
├── data/
│   ├── video/           # Input videos
│   ├── frames/          # Extracted frames + labelme JSON labels
│   └── dataset/         # Train/val/test split + dataset.yaml
├── models/              # Training artefacts, output video, counts CSV
├── yolo_models/         # Base YOLOv8 weights (yolov8n.pt, yolov8m.pt)
├── src/
│   ├── data_labeling/
│   │   └── video_to_frames.py       # Frame extraction
│   ├── data_processing/
│   │   ├── yolo_to_labelme.py       # Convert existing YOLO TXT → labelme JSON
│   │   ├── labelme_to_yolo.py       # Convert labelme JSON → YOLO TXT
│   │   └── prepare_dataset.py       # Train/val/test split + dataset.yaml
│   └── model_training/
│       ├── train.py                 # YOLOv8 training
│       └── inference.py             # Per-frame player counting
└── pipelines/
    ├── config.py
    ├── main_labeling_pipeline.py
    ├── main_processing_pipeline.py
    ├── main_training_pipeline.py
    └── main_inference_pipeline.py
```
