# Max-Toilet

A detection logger for your dog's toilet pad area using a Tapo camera.  
When Max has a **wee** or a **poo** on the pad the event is automatically classified and saved — with a full UTC timestamp — to both a CSV and a JSON Lines log file.

---

## How it works

### Live monitoring

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

### Historical backfill (last 30 days)

```
Tapo camera (local API / cloud)
      │
      ▼
 TapoCloudDownloader   ← lists recording dates, downloads MP4 files via pytapo
      │  MP4 files
      ▼
 process_video_file()  ← feeds every frame through PadDetector
      │  DetectionEvent (with original UTC timestamp)
      ▼
 EventLogger           ← same CSV + JSON output as live mode
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

## Windows setup (step-by-step)

> These instructions are written for Windows 10 / 11 using **PowerShell** or
> **Command Prompt**.  All commands are exactly what you type — no Linux
> knowledge required.
>
> ⚠️ **Important:** type (or paste) **one line at a time** and press **Enter**
> after each line before moving on to the next.  Pasting multiple lines at once
> is the most common source of errors.

### 1 · Install Python

1. Go to **https://www.python.org/downloads/** and click the yellow
   *"Download Python 3.x.x"* button.
2. Run the installer.  **Important:** on the first screen tick
   **"Add Python to PATH"** before clicking *Install Now*.
3. When it finishes, open a new **PowerShell** window (press
   `Windows + R`, type `powershell`, press Enter) and confirm Python
   works:
   ```powershell
   python --version
   ```
   You should see something like `Python 3.12.3`.

### 2 · Install Git

1. Go to **https://git-scm.com/download/win** and download the
   installer.
2. Run it with all default settings and click through to *Install*.
3. Close and reopen PowerShell, then confirm:
   ```powershell
   git --version
   ```

### 3 · Download Max-Toilet and set up a virtual environment

In PowerShell, run **each of the following lines one at a time** (press Enter
after every line and wait for it to finish before typing the next):

```powershell
cd $HOME\Desktop
git clone https://github.com/JayShep100/Max-Toilet.git
cd Max-Toilet
```

> **Tip – "already exists" error?**  If `git clone` prints
> `fatal: destination path 'Max-Toilet' already exists`, a previous
> clone attempt left an incomplete folder.  Delete it and try again:
> ```powershell
> Remove-Item -Recurse -Force $HOME\Desktop\Max-Toilet
> git clone https://github.com/JayShep100/Max-Toilet.git
> cd Max-Toilet
> ```

✅ **Stop and verify the clone worked.**  After `cd Max-Toilet`, run:

```powershell
dir requirements.txt
```

You should see a line like `requirements.txt` in the output.  If you get
*"File Not Found"* the clone did not download the full project — delete
the folder and clone again (see tip above), then continue.

✅ **Check your prompt.**  It should now look something like:

```
PS C:\Users\YourName\Desktop\Max-Toilet>
```

If it still shows your home folder (e.g. `PS C:\Users\YourName>`) **do not
continue** — type `cd $HOME\Desktop\Max-Toilet` and press Enter until the
prompt changes.

Once you are inside the `Max-Toilet` folder, continue:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **Tip – execution-policy error?**  If PowerShell refuses to run the
> activation script, run this one line first and then run the
> `.venv\Scripts\Activate.ps1` line again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

When activation succeeds your prompt will start with `(.venv)` — you must see
that prefix before continuing.

### 4 · Install dependencies

> ⚠️ Make sure your prompt still shows the `Max-Toilet` folder (e.g.
> `(.venv) PS C:\Users\YourName\Desktop\Max-Toilet>`).  If not, run
> `cd $HOME\Desktop\Max-Toilet` first.

```powershell
pip install -r requirements.txt
```

This installs OpenCV, pytapo, numpy, and everything else the project
needs.  It may take a minute or two.

### 5 · Create your configuration file

> ⚠️ Run this from inside the `Max-Toilet` folder (same prompt as above).

```powershell
copy config.example.json config.json
```

Open **config.json** in Notepad (or VS Code / any text editor) and
fill in your camera details:

```json
{
  "camera": {
    "host": "192.168.1.100",
    "username": "admin",
    "password": "your_camera_password",
    "stream_url": "rtsp://admin:your_camera_password@192.168.1.100:554/stream1",
    "cloud_password": "your_tapo_app_password"
  },
  "detection": {
    "pad_roi": { "x": 100, "y": 150, "width": 400, "height": 300 }
  },
  "logging": {
    "log_dir": "logs"
  },
  "cloud_backfill": {
    "days_back": 30,
    "download_dir": "downloads"
  }
}
```

| Field | What to put there |
|---|---|
| `camera.host` | The IP address of your Tapo camera (find it in your router's device list or the Tapo app under *Device Info*) |
| `camera.username` | Always `admin` for Tapo cameras |
| `camera.password` | The password you set when you first added the camera to the Tapo app |
| `camera.stream_url` | The RTSP URL — replace the IP and password with your own |
| `camera.cloud_password` | The password for your **Tapo / TP-Link** account (used for cloud backfill) |
| `detection.pad_roi` | Pixel rectangle that covers the toilet pad in the camera frame.  Set to `null` to use the whole frame. |

> **How do I find the pad coordinates?**  Open VLC → *Media → Open
> Network Stream* → paste your `stream_url`.  Hover your mouse over the
> corners of the pad; VLC shows the pixel position in the status bar.

---

## Setup (macOS / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/JayShep100/Max-Toilet.git
cd Max-Toilet

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create your configuration file
cp config.example.json config.json
```

Edit **config.json** — use the same field descriptions as the Windows table above.

---

## Running

The commands are **identical on Windows (PowerShell) and macOS/Linux**.
Make sure your virtual environment is active first
(`(.venv)` prefix in the prompt).

### 6 · Start live monitoring (Windows)

```powershell
python -m src.main --config config.json
```

Press **Ctrl+C** to stop gracefully.

### Live monitoring (macOS / Linux)

```bash
python -m src.main --config config.json
```

### 7 · Historical backfill — last 30 days of cloud recordings

Run this once to process everything saved on the camera in the last 30 days:

```powershell
# Windows PowerShell (or Command Prompt)
python -m src.main --config config.json --backfill
```

```bash
# macOS / Linux
python -m src.main --config config.json --backfill
```

To limit to a shorter window, use `--days`:

```powershell
python -m src.main --config config.json --backfill --days 7
```

The backfill command will:
1. Connect to the camera's local API using the credentials in `config.json`
2. Query for all recording dates within the specified window
3. Download each recording segment as an MP4 to `downloads\` (configurable)
4. Run every frame through the detector
5. Log all detected events to the same CSV and JSON log files as live mode, using the original recording timestamp

---

## Common Windows issues

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Re-run the Python installer and tick **"Add Python to PATH"**, then open a fresh PowerShell window |
| `Scripts\Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry |
| Camera stream won't open | Make sure RTSP is enabled in the Tapo app (*Camera Settings → Advanced Settings → RTSP*) and that your laptop is on the same Wi-Fi network as the camera |
| Windows Defender Firewall blocks the stream | Allow Python through the firewall when prompted, or add an inbound rule for port 554 |
| `pip install` fails with a compiler error | Install the free **Microsoft C++ Build Tools** from https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

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

```powershell
# Windows PowerShell
python -m pytest tests/ -v
```

```bash
# macOS / Linux
python -m pytest tests/ -v
```

---

## Project structure

```
Max-Toilet/
├── config.example.json   # Configuration template
├── requirements.txt
├── src/
│   ├── camera.py           # Tapo RTSP stream connection
│   ├── cloud_downloader.py # Download recordings from the Tapo camera (up to 30 days)
│   ├── detector.py         # Motion detection & wee/poo classification
│   ├── logger.py           # CSV + JSON Lines event logging
│   ├── main.py             # Entry point (live + --backfill mode)
│   └── video_processor.py  # Process downloaded video files through the detector
├── tests/
│   ├── test_cloud_downloader.py
│   ├── test_detector.py
│   ├── test_logger.py
│   └── test_video_processor.py
└── logs/                 # Created at runtime
```
