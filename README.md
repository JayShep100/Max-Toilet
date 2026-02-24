# Max-Toilet

A detection logger for your dog's toilet pad area using a Tapo camera.  
When Max has a **wee** or a **poo** on the pad the event is automatically classified and saved — with a full UTC timestamp — to both a CSV and a JSON Lines log file.

---

## How it works

```
Tapo camera (RTSP)
      │
      ▼
 TapoCamera            ← connects to the RTSP stream
      │  frame-by-frame
      ▼
 PadDetector           ← background subtraction (MOG2) + HSV colour analysis
      │  DetectionEvent
      ▼
 EventLogger           ← appends to toilet_events.csv  &  toilet_events.json
```

1. **Motion detection** – OpenCV's MOG2 background subtractor watches the
   configured pad region-of-interest (ROI).  When enough pixels change, the
   dog is considered "present".
2. **Classification** – After the dog leaves (motion stops), the pad ROI is
   analysed in HSV colour space:
   - **Wee** → yellowish pixels (hue ≈ 20–35°)
   - **Poo** → dark brownish pixels (hue ≈ 5–20°, low saturation/value)
   - **Unknown** → dog was present but no clear colour signature
3. **Logging** – Every event is appended to both log files with a UTC
   timestamp, event type, confidence score, and raw pixel counts.

---

## Requirements

- Python 3.9+
- A Tapo camera accessible on your local network (C200, C310, etc.)
- The camera's RTSP stream enabled (Settings → Advanced → RTSP)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/JayShep100/Max-Toilet.git
cd Max-Toilet

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Create your configuration file
cp config.example.json config.json
```

Edit **config.json**:

```json
{
  "camera": {
    "stream_url": "rtsp://admin:YOUR_PASSWORD@192.168.1.100:554/stream1"
  },
  "detection": {
    "pad_roi": { "x": 100, "y": 150, "width": 400, "height": 300 }
  },
  "logging": {
    "log_dir": "logs"
  }
}
```

| Key | Description |
|---|---|
| `camera.stream_url` | Full RTSP URL of your Tapo camera |
| `detection.pad_roi` | Pixel coordinates of the toilet pad in the camera frame. Set to `null` to use the full frame. |
| `detection.motion_threshold` | MOG2 sensitivity (default 500) |
| `detection.color_change_pixel_threshold` | Minimum changed pixels to classify an event (default 300) |
| `logging.log_dir` | Directory where log files are written (default `logs/`) |

---

## Running

```bash
python -m src.main --config config.json
```

Press **Ctrl+C** to stop gracefully.

---

## Log files

Both files are written to the `logs/` directory (configurable).

### `toilet_events.csv`

```
timestamp_utc,event_type,confidence,motion_pixel_count,wee_pixels,poo_pixels
2025-06-01T08:42:11.034Z,wee,0.9231,47,18432,1200
2025-06-01T14:07:55.811Z,poo,0.8750,63,540,22016
```

### `toilet_events.json` (JSON Lines)

```json
{"timestamp_utc": "2025-06-01T08:42:11.034Z", "event_type": "wee", "confidence": 0.9231, "motion_pixel_count": 47, "wee_pixels": 18432, "poo_pixels": 1200}
{"timestamp_utc": "2025-06-01T14:07:55.811Z", "event_type": "poo", "confidence": 0.875,  "motion_pixel_count": 63, "wee_pixels": 540,   "poo_pixels": 22016}
```

---

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

---

## Project structure

```
Max-Toilet/
├── config.example.json   # Configuration template
├── requirements.txt
├── src/
│   ├── camera.py         # Tapo RTSP stream connection
│   ├── detector.py       # Motion detection & wee/poo classification
│   ├── logger.py         # CSV + JSON Lines event logging
│   └── main.py           # Entry point
├── tests/
│   ├── test_detector.py
│   └── test_logger.py
└── logs/                 # Created at runtime
```
