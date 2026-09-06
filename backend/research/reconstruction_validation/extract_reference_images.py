import argparse
from pathlib import Path
import cv2

BASE_DIR = Path(r"D:\project-quartz\final_research_evidence\reconstruction_multi\batch_inputs")
VIDEO_NAMES = ["01.mov", "01.mp4", "1.mov", "1.mp4"]

def pick_best_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    # sample frames from 10% to 60% of the video
    sample_positions = []
    start = max(1, int(frame_count * 0.10))
    end = max(start + 1, int(frame_count * 0.60))
    steps = 20
    for i in range(steps):
        pos = start + int((end - start) * i / max(steps - 1, 1))
        sample_positions.append(min(pos, frame_count - 1))

    best_score = -1
    best_frame = None
    best_idx = None

    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()  # sharpness score

        if score > best_score:
            best_score = score
            best_frame = frame
            best_idx = pos

    cap.release()

    if best_frame is None:
        raise RuntimeError(f"Could not extract any usable frame: {video_path}")

    return best_frame, best_idx, best_score

def parse_args():
    parser = argparse.ArgumentParser(description="Extract specimen reference images from first capture videos.")
    parser.add_argument("--only", help="Extract only one specimen ID, e.g. QZ-03.")
    return parser.parse_args()

def main():
    args = parse_args()
    if not BASE_DIR.exists():
        raise SystemExit(f"Base folder not found: {BASE_DIR}")

    specimen_dirs = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    if args.only:
        specimen_dirs = [p for p in specimen_dirs if p.name == args.only]
    if not specimen_dirs:
        raise SystemExit("No specimen folders found.")

    for specimen_dir in sorted(specimen_dirs):
        specimen_id = specimen_dir.name
        videos_dir = specimen_dir / "videos"
        reference_dir = specimen_dir / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)

        video_path = None
        for name in VIDEO_NAMES:
            candidate = videos_dir / name
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            print(f"[SKIP] {specimen_id}: no first video found in {videos_dir}")
            continue

        out_path = reference_dir / f"{specimen_id}_reference.jpg"

        try:
            frame, idx, score = pick_best_frame(video_path)
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                raise RuntimeError(f"Failed to save image: {out_path}")
            print(f"[OK] {specimen_id}: {out_path}  (frame={idx}, sharpness={score:.2f})")
        except Exception as e:
            print(f"[FAIL] {specimen_id}: {e}")

if __name__ == "__main__":
    main()
