import csv
import cv2
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

# BGR colors per team; referee is intentionally absent (hidden)
_TEAM_COLORS = {
    "white_team": (255, 255, 255),
    "red_team":   (0, 0, 255),
}


def count_players_per_frame(
    model_path: str,
    source: str,
    class_names: list[str],
    conf_threshold: float = 0.4,
    output_video_path: str | None = None,
    output_csv_path: str | None = None,
) -> list[dict]:
    """
    Run inference and count visible players per team for every frame.
    Referees are detected but excluded from counts and the output video.

    Args:
        model_path: Path to trained YOLO weights (.pt).
        source: Path to a video file or directory of images.
        class_names: Class names ordered by label ID.
        conf_threshold: Minimum detection confidence.
        output_video_path: If provided, save an annotated video here.
        output_csv_path: If provided, save per-frame counts as CSV here.

    Returns:
        List of dicts: [{"frame": 0, "white_team": 3, "red_team": 4}, ...]
    """
    model = YOLO(model_path)
    results_log = []
    writer = None

    cap = cv2.VideoCapture(str(source))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    team_names = [n for n in class_names if n in _TEAM_COLORS]

    for frame_idx, result in enumerate(
        model.predict(source=str(source), conf=conf_threshold, stream=True, verbose=False)
    ):
        counts: dict[str, int] = defaultdict(int)
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id < len(class_names):
                name = class_names[class_id]
                if name in _TEAM_COLORS:
                    counts[name] += 1

        entry = {"frame": frame_idx, **{name: counts.get(name, 0) for name in team_names}}
        results_log.append(entry)

        if output_video_path:
            annotated = _annotate_frame(result, class_names)
            _draw_counts(annotated, counts, team_names)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    source_fps,
                    (w, h),
                )
            writer.write(annotated)

    if writer:
        writer.release()

    if output_csv_path:
        _write_csv(results_log, output_csv_path)

    _print_summary(results_log, team_names)
    return results_log


def _annotate_frame(result, class_names: list[str]):
    frame = result.orig_img.copy()
    for box in result.boxes:
        class_id = int(box.cls)
        if class_id >= len(class_names):
            continue
        name = class_names[class_id]
        color = _TEAM_COLORS.get(name)
        if color is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def _draw_counts(frame, counts: dict, team_names: list[str]) -> None:
    y = 20
    for name in team_names:
        color = _TEAM_COLORS[name]
        cv2.putText(frame, f"{name}: {counts.get(name, 0)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        y += 30


def _write_csv(results_log: list[dict], path: str) -> None:
    if not results_log:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)
    print(f"Per-frame counts saved -> {path}")


def _print_summary(results_log: list[dict], team_names: list[str]) -> None:
    if not results_log:
        return
    print(f"\nInference complete — {len(results_log)} frames")
    for name in team_names:
        values = [r[name] for r in results_log]
        print(f"  {name}: avg {sum(values)/len(values):.1f}, max {max(values)}")
