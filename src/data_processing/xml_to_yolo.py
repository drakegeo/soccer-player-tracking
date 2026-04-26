import xml.etree.ElementTree as ET
from pathlib import Path


def convert_xml_to_yolo(labels_dir: str, class_names: list[str]) -> int:
    """
    Convert Pascal VOC XML label files to YOLO TXT format in-place.

    Each XML produces a matching .txt file in the same directory.

    Args:
        labels_dir: Directory containing Pascal VOC .xml files.
        class_names: Ordered class list — index = YOLO class ID.

    Returns:
        Number of files converted.
    """
    labels_dir = Path(labels_dir)
    converted = 0

    for xml_path in sorted(labels_dir.glob("*.xml")):
        txt_path = xml_path.with_suffix(".txt")
        lines = _parse_xml(xml_path, class_names)
        txt_path.write_text("\n".join(lines))
        converted += 1

    print(f"Converted {converted} XML files to YOLO TXT in {labels_dir}")
    return converted


def _parse_xml(xml_path: Path, class_names: list[str]) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in class_names:
            continue
        class_id = class_names.index(name)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines
