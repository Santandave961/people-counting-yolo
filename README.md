# Real-Time People Counting using YOLO + ROI + Line Crossing

A real-time people counting system built with Ultralytics YOLO. It detects and tracks every person in a video or camera feed, counts who crosses a line (UP and DOWN), marks counted people in green, logs every crossing to CSV, and generates a traffic heatmap.

![Heatmap](heatmap.png)

> Add a screenshot or GIF of the green "COUNTED" marks here, for example `![Demo](demo.gif)`.

## Features

- YOLO detection + tracking (Ultralytics `model.track`)
- Persistent track IDs
- ROI-based filtering (the default ROI is the whole screen)
- Line crossing detection with separate UP and DOWN counts
- Center smoothing using a `deque`, which prevents fake crossings from jitter
- Unique counting using ID memory, so each person is counted once
- Green mark on every counted person
- CSV export of every crossing
- Heatmap of where people walk
- Real-time visualization with OpenCV (full screen by default)

## Tech Stack

Python | Ultralytics YOLO | OpenCV | NumPy

## Installation

```bash
git clone https://github.com/Santandave961/people-counting-yolo.git
cd people-counting-yolo
pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`) download automatically the first time you run it.

## Usage

Run on a video file:

```bash
python people_counter.py --source Movement.mp4
```

Run on a webcam:

```bash
python people_counter.py --source 0
```

Run on an IP or CCTV camera:

```bash
python people_counter.py --source "rtsp://user:pass@192.168.1.10:554/stream"
```

Press `q` or `Esc` to quit.

## Options

| Option | Default | What it does |
|---|---|---|
| `--source` | `0` | Video path, webcam index, or stream URL |
| `--model` | `yolo11n.pt` | YOLO weights (use `yolo11s.pt` for better accuracy) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--output` | `output.mp4` | Annotated video path (`""` to disable) |
| `--csv` | `crossings.csv` | Crossing log file (`""` to disable) |
| `--heatmap` | `heatmap.png` | Heatmap image saved at the end (`""` to disable) |
| `--live-heatmap` | off | Show the heatmap on the video while running |
| `--show-guides` | off | Draw the ROI and the counting line |
| `--windowed` | off | Use a normal window instead of full screen |
| `--no-show` | off | Run without the live window (faster) |

## Outputs

- `output.mp4`: annotated video with counters and green marks
- `crossings.csv`: one row per crossing (`timestamp, video_time_s, track_id, direction, up_total, down_total`)
- `heatmap.png`: traffic heatmap (blue = quiet, red = busiest)

## Configuration

At the top of `people_counter.py`:

```python
ROI_FRAC = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]   # whole screen
LINE_FRAC = ((0.0, 0.55), (1.0, 0.55))                         # counting line
SMOOTH_WINDOW = 5                                              # frames averaged
MARK_ALPHA = 0.35                                              # green mark opacity
```

Values are fractions of the frame, so they work at any resolution. Change the `0.55` in `LINE_FRAC` to move the line up or down.

## How it works

1. YOLO detects people and the tracker gives each one a persistent ID.
2. Only people whose center is inside the ROI are processed.
3. Each person's center is averaged over the last few frames to remove jitter.
4. The system checks which side of the line the smoothed center is on. A change of side means a crossing, and the direction gives UP or DOWN.
5. The ID is saved to memory so that person is never counted twice.
6. Counted people get a green mark, and the crossing is written to the CSV.

## Use cases

- **Retail:** compare real foot traffic to sales and find peak hours
- **Restaurants and malls:** staff for busy periods
- **Banks and offices:** measure queues and busy times
- **Events, churches, stadiums:** keep to safe occupancy limits
- **Transport hubs and clinics:** monitor crowd flow and waiting-room load

## Limitations

- Accuracy drops in very crowded scenes where people overlap heavily.
- Tracker IDs can switch when people are hidden for a long time. Try `--tracker botsort.yaml` or a larger model.
- The line should be placed so people cross it in a clear direction.

## Author

**Wisdom Okparaji** | Data Scientist and ML Engineer

- GitHub: [Santandave961](https://github.com/Santandave961)
- Portfolio: [santandave961.github.io](https://santandave961.github.io)
- X: [@Santandave961](https://x.com/Santandave961)

Open to ML Engineer and Data Scientist roles at startups. DMs are open.

## License

MIT
