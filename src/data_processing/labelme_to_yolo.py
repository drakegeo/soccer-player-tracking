import json
from pathlib import Path


def convert_labelme_to_yolo(frames_dir: str, class_names: list[str]) -> int:
    """
    Convert labelme JSON label files to YOLO TXT format in-place.

    Args:
        frames_dir: Directory containing labelme .json label files.
        class_names: Ordered class list — index = YOLO class ID.

    Returns:
        Number of files converted.
    """
    frames_dir = Path(frames_dir)
    converted = 0

    for json_path in sorted(frames_dir.glob("*.json")):
        data = json.loads(json_path.read_text())
        img_w = data["imageWidth"]
        img_h = data["imageHeight"]

        lines = []
        for shape in data.get("shapes", []):
            if shape["shape_type"] != "rectangle":
                continue
            label = shape["label"]
            if label not in class_names:
                continue
            class_id = class_names.index(label)
            (x1, y1), (x2, y2) = shape["points"]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        txt_path = json_path.with_suffix(".txt")
        txt_path.write_text("\n".join(lines))
        converted += 1

    print(f"Converted {converted} labelme JSON files to YOLO TXT in {frames_dir}")
    return converted
