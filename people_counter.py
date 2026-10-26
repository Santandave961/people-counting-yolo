"""
Real-Time People Counting using YOLO + ROI + Line Crossing
-----------------------------------------------------------
- YOLO detection + tracking (Ultralytics model.track) with persistent IDs
- ROI-based filtering (only people inside the polygon are counted)
- Line crossing detection (UP and DOWN counting)
- Center point smoothing using deque
- Unique counting using ID memory (each ID is counted once)
- Real-time visualization with OpenCV
- CSV export: every crossing is logged (time, ID, direction, running totals)
- Heatmap: shows where people walk most, saved as an image at the end

Install:  pip install ultralytics opencv-python numpy
Run:      python people_counter.py --source people.mp4
          python people_counter.py --source 0            (webcam)
Quit:     press 'q'
"""

import argparse
import csv
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIG  (ROI and line are fractions of frame size, so any resolution works)
# ----------------------------------------------------------------------------
# ROI polygon: (x, y) fractions -> top-left, top-right, bottom-right, bottom-left
ROI_FRAC = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]   # whole screen

# Counting line: ((x1, y1), (x2, y2)) fractions
LINE_FRAC = ((0.0, 0.55), (1.0, 0.55))

SMOOTH_WINDOW = 5        # number of past centers averaged per track
MARK_ALPHA = 0.35        # opacity of the green mark on counted people (0 to 1)
STALE_AFTER = 60         # frames before a lost track's history is purged

GREEN, RED, YELLOW, BLUE, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 128, 0), (255, 255, 255)


def side_of_line(p, a, b):
    """Cross product sign: which side of line a->b the point p is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def within_segment(p, a, b):
    """True if p's projection falls on the segment a-b (not the infinite line)."""
    ab = np.array(b, dtype=float) - np.array(a, dtype=float)
    ap = np.array(p, dtype=float) - np.array(a, dtype=float)
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
    return 0.0 <= t <= 1.0


def heat_layer(heat):
    """Blur the raw heat counts and turn them into a colored layer plus a mask."""
    blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=25)
    peak = blur.max()
    if peak <= 0:
        return None, None
    norm = (blur / peak * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)   # blue = low, red = busiest
    mask = (norm > 20)[..., None]                         # only color where people walked
    return colored, mask


def blend_heat(base, colored, mask, alpha=0.55):
    """Blend the heat layer over an image, only where there is heat."""
    if colored is None:
        return base
    mixed = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)
    return np.where(mask, mixed, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video path or webcam index")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--output", default="output.mp4", help="annotated video path ('' to disable)")
    ap.add_argument("--no-show", action="store_true", help="disable the live window")
    ap.add_argument("--windowed", action="store_true", help="use a normal window instead of full screen")
    ap.add_argument("--show-guides", action="store_true", help="draw the ROI polygon and counting line")
    ap.add_argument("--csv", default="crossings.csv", help="crossing log file ('' to disable)")
    ap.add_argument("--heatmap", default="heatmap.png", help="heatmap image saved at the end ('' to disable)")
    ap.add_argument("--live-heatmap", action="store_true", help="show the heatmap on the video while running")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Convert fractional config to pixels
    roi = np.array([(int(x * W), int(y * H)) for x, y in ROI_FRAC], dtype=np.int32)
    line_a = (int(LINE_FRAC[0][0] * W), int(LINE_FRAC[0][1] * H))
    line_b = (int(LINE_FRAC[1][0] * W), int(LINE_FRAC[1][1] * H))

    model = YOLO(args.model)

    # Heatmap + CSV state
    heat = np.zeros((H, W), dtype=np.float32)   # how often people stood at each pixel
    heat_cache = (None, None)                   # cached colored layer for --live-heatmap
    base_frame = None                           # first clean frame, used as heatmap background
    csv_file = open(args.csv, "w", newline="") if args.csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["timestamp", "video_time_s", "track_id", "direction", "up_total", "down_total"])

    # State
    history = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))  # raw centers per ID
    prev_side = {}                                              # last side-of-line per ID
    last_seen = {}                                              # last frame index per ID
    counted_ids = set()                                         # ID memory -> unique counting
    count_up, count_down = 0, 0
    frame_idx = 0

    # Display window: full screen by default
    window = "People Counter"
    if not args.no_show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if args.windowed:
            cv2.resizeWindow(window, 1280, 720)
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if base_frame is None:
            base_frame = frame.copy()

        # Detection + tracking (class 0 = person), persist=True keeps IDs across frames
        results = model.track(
            frame, persist=True, classes=[0], conf=args.conf,
            tracker=args.tracker, verbose=False,
        )[0]

        in_roi_now = 0
        marks = []   # boxes of people already counted -> get the green mark

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.int().cpu().tolist()

            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # ---- ROI filter: ignore anyone whose center is outside the polygon
                if cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) < 0:
                    history.pop(tid, None)
                    prev_side.pop(tid, None)
                    continue
                in_roi_now += 1
                last_seen[tid] = frame_idx

                # ---- Heatmap: add heat where this person is standing (feet point)
                fx = min(max(int((x1 + x2) / 2), 0), W - 1)
                fy = min(max(int(y2), 0), H - 1)
                heat[fy, fx] += 1

                # ---- Center smoothing (moving average over a deque)
                history[tid].append((cx, cy))
                sx = float(np.mean([p[0] for p in history[tid]]))
                sy = float(np.mean([p[1] for p in history[tid]]))
                smooth = (sx, sy)

                # ---- Line crossing: compare side of line between frames
                cur_side = side_of_line(smooth, line_a, line_b)
                old_side = prev_side.get(tid)

                if (old_side is not None and tid not in counted_ids
                        and old_side * cur_side < 0
                        and within_segment(smooth, line_a, line_b)):
                    if old_side < 0 < cur_side:      # moved from above to below
                        count_down += 1
                        direction = "DOWN"
                    else:                            # moved from below to above
                        count_up += 1
                        direction = "UP"
                    counted_ids.add(tid)             # each ID counted only once

                    # ---- CSV export: one row per crossing
                    if csv_writer:
                        csv_writer.writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            round(frame_idx / fps, 2), tid, direction, count_up, count_down,
                        ])
                        csv_file.flush()

                prev_side[tid] = cur_side

                # ---- Drawing
                counted = tid in counted_ids
                color = GREEN if counted else WHITE
                if counted:
                    marks.append((int(x1), int(y1), int(x2), int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.circle(frame, (int(sx), int(sy)), 4, RED, -1)

                pts = np.array(history[tid], dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, BLUE, 2)

        # ---- Green mark on every person who has been counted
        if marks:
            overlay = frame.copy()
            for (x1, y1, x2, y2) in marks:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), GREEN, -1)   # fill the body
            frame = cv2.addWeighted(overlay, MARK_ALPHA, frame, 1 - MARK_ALPHA, 0)
            for (x1, y1, x2, y2) in marks:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
                chest = ((x1 + x2) // 2, int(y1 + 0.35 * (y2 - y1)))
                cv2.circle(frame, chest, 10, GREEN, -1)                  # solid green dot
                cv2.circle(frame, chest, 14, WHITE, 2)
                cv2.putText(frame, "COUNTED", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

        # Purge stale tracks so memory doesn't grow forever
        if frame_idx % 30 == 0:
            for tid in [t for t, f in last_seen.items() if frame_idx - f > STALE_AFTER]:
                history.pop(tid, None)
                prev_side.pop(tid, None)
                last_seen.pop(tid, None)

        # ---- Live heatmap overlay (refreshed every 10 frames to stay fast)
        if args.live_heatmap:
            if frame_idx % 10 == 1:
                heat_cache = heat_layer(heat)
            frame = blend_heat(frame, *heat_cache)

        # ---- ROI, line, and HUD
        if args.show_guides:   # ROI + counting line are hidden unless --show-guides is passed
            cv2.polylines(frame, [roi], True, WHITE, 2)
            cv2.line(frame, line_a, line_b, RED, 3)

        hud = [f"UP: {count_up}", f"DOWN: {count_down}",
               f"TOTAL: {count_up + count_down}", f"IN ROI: {in_roi_now}"]
        cv2.rectangle(frame, (10, 10), (200, 20 + 28 * len(hud)), (0, 0, 0), -1)
        for i, text in enumerate(hud):
            cv2.putText(frame, text, (20, 38 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):   # q or Esc to quit
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if csv_file:
        csv_file.close()
        print(f"Crossing log saved -> {args.csv}")
    if args.heatmap and base_frame is not None:
        colored, mask = heat_layer(heat)
        cv2.imwrite(args.heatmap, blend_heat(base_frame, colored, mask))
        print(f"Heatmap saved -> {args.heatmap}")

    print(f"Final -> UP: {count_up} | DOWN: {count_down} | TOTAL: {count_up + count_down}")


if __name__ == "__main__":
    main()