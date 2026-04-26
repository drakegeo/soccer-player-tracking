import cv2
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    max_frames: int | None = None,
) -> int:
    """
    Sample frames from a video at the given rate and save as JPEG.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames will be saved.
        fps: Frames per second to extract.
        max_frames: Stop after saving this many frames. None means no limit.

    Returns:
        Number of frames saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(video_fps / fps)))

    print(f"Video: {video_fps:.1f} fps, {total_frames} frames — sampling every {frame_interval} frames")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and saved >= max_frames:
            break
        if frame_idx % frame_interval == 0:
            cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.jpg"), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames -> {output_dir}")
    return saved
