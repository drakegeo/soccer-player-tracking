import json
import cv2
from pathlib import Path


def convert_yolo_to_labelme(frames_dir: str, class_names: list[str]) -> int:
    """
    Convert YOLO TXT label files to labelme JSON format in-place.

    Skips frames that already have a .json file (won't overwrite).

    Args:
        frames_dir: Directory containing .jpg frames and YOLO .txt label files.
        class_names: Ordered class list — index = YOLO class ID.

    Returns:
        Number of files converted.
    """
    frames_dir = Path(frames_dir)
    converted = 0

    for txt_path in sorted(frames_dir.glob("*.txt")):
        if txt_path.name == "predefined_classes.txt":
            continue
        json_path = txt_path.with_suffix(".json")
        if json_path.exists():
            continue
        img_path = txt_path.with_suffix(".jpg")
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        shapes = []
        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            xmin = (cx - bw / 2) * w
            ymin = (cy - bh / 2) * h
            xmax = (cx + bw / 2) * w
            ymax = (cy + bh / 2) * h
            shapes.append({
                "label": label,
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            })

        data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        json_path.write_text(json.dumps(data, indent=2))
        converted += 1

    print(f"Converted {converted} YOLO TXT files to labelme JSON in {frames_dir}")
    return converted
