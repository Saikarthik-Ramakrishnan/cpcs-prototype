# CPCS: Camera-Based Passenger Counting System

> **Proof of concept** · first-year ECE intern project  
> Mounts two cameras at a bus doorway, counts boarding and alighting passengers at every stop, and reconciles counts against the on-board ticket machine (POS) to detect revenue leakage — all running locally on edge hardware with no video upload to the cloud.

---

## Problem statement

Indian city bus operators lose significant revenue to fare evasion and have no independent check on their ticket machines. Manual headcounts are inconsistent; existing camera systems produce video but not structured data. CPCS turns a pair of commodity USB cameras into a real-time passenger counter whose output can be audited stop-by-stop and compared to what the POS system says it collected.

---

## How it works

```
2× USB cameras (bus doorway)
        │
        ▼
YOLOv8 person detection  ──► ByteTrack multi-object tracker
        │
        ▼
DoorCounter v6  (four recovery layers, every count audited)
  ├─ live      — zone transition observed while continuously tracked
  ├─ coast     — box lost at the line; crossing completed from measured velocity
  ├─ stitch    — fragmented track reconnected by velocity prediction
  └─ fallback  — born one side of line, died the other
        │
        ▼
TripRecorder  ──► SQLite (per-stop boardings, alightings, occupancy, POS reconciliation)
        │
        ▼
build_dashboard.py  ──► cpcs_dashboard.html
  (self-contained, no server, bilingual EN/HI, print-ready)
```

**Spec targets**

| Target | Status |
|---|---|
| ≥ 95% recall on boarding/alighting events | Validated on benchmark clip; production target |
| Runs fully locally on Orange Pi 5 (RK3588S, 6 TOPS NPU) | Pipeline runs; RKNN export in progress |
| No video upload to cloud | Enforced by design — only counts leave the device |
| Per-stop POS reconciliation + revenue leakage flagging | Implemented and demoed |
| Scalable to 500-bus fleet | SQLite → Postgres is one connection-string change |

---

## Repository structure

```
cpcs-prototype/
│
├── cpcs_poc.py           ← main capture app: runs the camera, counts, writes to DB
├── build_dashboard.py    ← generates cpcs_dashboard.html from the DB
├── requirements.txt
│
└── counter/              ← development history (each version is a standalone script)
    ├── v2_basic.py             dead-zone hysteresis crossing
    ├── v3_hysteresis.py        EMA smoothing + state machine
    ├── v4_stitch.py            velocity-predicted track stitching
    └── v5_predictive_stitch.py full stitch + in-band birth + diagnostic logs
```

The production counting engine (v6) lives inside `cpcs_poc.py` as the `DoorCounter` class. The `counter/` folder preserves the iterative development history.

---

## Quick start

**Requirements:** Python 3.10+, a webcam or video file.

```bash
git clone https://github.com/Saikarthik-Ramakrishnan/cpcs-prototype.git
cd cpcs-prototype
pip install -r requirements.txt
```

**Run on the benchmark clip** (download `test_1.mp4` separately — see below):

```bash
python cpcs_poc.py --source test_1.mp4 --route "47A" --bus "DL-1PC-4432"
```

**Controls while the video window is open:**

| Key | Action |
|-----|--------|
| `n` | Commit the current stop and move to the next |
| `p` | Toggle simulated POS ticket counts |
| `q` | End the trip and quit |

**Build the dashboard:**

```bash
python build_dashboard.py
# opens cpcs_dashboard.html — double-click it in any browser
```

**Recommended settings for best accuracy:**

```bash
python cpcs_poc.py \
  --source test_1.mp4 \
  --model yolov8s.pt \
  --imgsz 960 \
  --route "47A" \
  --bus "DL-1PC-4432"
```

`yolov8s.pt` downloads automatically (~22 MB) on first run.

---

## Dashboard features

The generated `cpcs_dashboard.html` is a self-contained single-file report — no server, no framework, no dependencies beyond a browser.

**Fleet overview tab** (commissioner / management view)
- Aggregate KPIs across all logged trips: total passengers, flagged stops, revenue at risk
- Per-trip bar charts for boardings and revenue at risk
- Trip table with one-click drill-down

**Trip detail tab** (operations / revenue view)
- Occupancy curve with capacity reference line and flag pins at discrepant stops
- Diverging boardings/alightings bar chart per stop
- Counting method breakdown per stop (live / coast / fallback) — the audit trail
- Timeline scatter of every individual crossing event, colored by method
- Sortable per-stop table with per-stop data confidence percentage
- Plain-language revenue reconciliation with rupee-at-risk figures

**Controls**
- Live fare (₹) and capacity inputs — all figures recompute instantly
- `हिंदी` button — full bilingual toggle (English / Hindi)
- `Print report` — switches to light theme, prints both tabs, restores your theme
- Dark / light theme toggle

---

## Counting engine: accuracy layers

The v6 `DoorCounter` runs four layers in descending order of evidence quality:

### 1. `live`: direct observation
A track's smoothed centroid crosses the hysteresis band in a single direction. Requires `MIN_AGE = 2` frames of existence and was not previously counted. This is the high-confidence path.

### 2. `coast`: dead reckoning
When a box vanishes (detection void at the line, the dominant failure mode on low-res footage), the track's last measured velocity is extrapolated forward for up to `COAST_MAX_GAP = 12` frames. If the predicted position crosses the band, the crossing fires. Guards: `MIN_AGE ≥ 4`, `|vy| ≥ 2.0 px/frame`. Disable with `--no-coast`.

### 3. `stitch`: fragment reconnection
A new track born near where a quiet track *would be now* (velocity × gap) inherits the old track's history. This reconnects a fragmented crossing across a detection void without requiring the missing boxes.

### 4. `fallback`: birth-to-death trajectory
A track that was born on one side of the line and retired on the other, without ever being counted, produces a crossing event on cleanup. Last resort.

Every event is written to the DB with its `how` tag. The dashboard shows the breakdown per stop, and a **data confidence** figure (fraction of counts that were `live`) rather than an unverifiable accuracy claim.

---

## Running on Orange Pi 5 (edge deployment)

The Orange Pi 5 (RK3588S) has a 6 TOPS NPU that can run a quantised RKNN model much faster than the ARM CPU. The planned path:

```
yolov8n.pt  →  export to ONNX  →  rknn-toolkit2 INT8 quantisation  →  .rknn model
```

CPU inference (`yolov8n.pt`) already runs on the Orange Pi 5 at ~8–12 fps on 640px input, which is sufficient for bus-door counting. NPU inference targets ≥ 20 fps. Edge deployment is week 3 of the one-month roadmap.

---



## Per-bus calibration

The counting line is a virtual tripwire. A fixed line in the middle of the frame is a guess; a calibrated line matches the real doorway geometry of each bus. This is the single biggest accuracy lever for a live deployment, and it is how one codebase scales to a whole fleet: each bus is one config file, no code changes.

### Calibrate a bus

```bash
python calibrate.py --source recordings/<session>/cam0.mp4 --route "47A" --bus "DL-1PC-4432"
```

Controls: click two points to place the line, `[` and `]` adjust the dead zone band, `f` flips which side counts as boarding, `space` grabs the next frame, `s` saves to `config.yaml`, `q` quits.

### Run the counter against the calibrated line

```bash
python cpcs_poc.py --source recordings/<session>/cam0.mp4 --config config.yaml
```

If no config is given and no `config.yaml` is present, the line falls back to horizontal across the middle of the frame, preserving earlier behaviour.

### Config file

Everything that varies between buses lives in `config.yaml`:

```yaml
bus:
  route: "47A"
  bus_id: "DL-1PC-4432"
camera:
  index: 0
  line: [10, 300, 1900, 340]   # calibrated tripwire in pixels; null for horizontal mid
  dead_zone: 22
  flip: false
detection:
  model: yolov8n.pt
  imgsz: 640
  conf: 0.10
runtime:
  fps: 30
  coast: true
economics:
  fare: 15
  capacity: 45
```

The counting line supports any position and any angle, so a tilted or wide doorway can be matched exactly. The geometry reduces to the original `y` based math for a horizontal line, so existing results are unchanged.


## On-bus testing (edge board)

Real-bus validation runs in three phases so hardware acceleration and
real-world footage are never debugged at the same time.

### Phase 1: record only (no inference)
`cpcs_recorder.py` is board-agnostic (x86 or ARM, no NPU, no model). It
captures 2-3 synchronized USB camera feeds to timestamped `.mp4` files plus
a `frames.csv` per camera for later alignment.

```bash
pip install opencv-python
python cpcs_recorder.py --list                       # see available cameras
python cpcs_recorder.py --cams 0 2 --route "47A" --bus "DL-1PC-4432"
# Ctrl-C to stop; footage lands in ./recordings/
```

### Phase 2: replay through the pipeline (dev machine)
Copy the recordings off the board and run each camera clip through the
counting app exactly like the benchmark clip:

```bash
python cpcs_poc.py --source recordings/<session>/cam0.mp4 --route "47A" --bus "DL-1PC-4432"
python build_dashboard.py
```

Hand-count the footage and compare — this is the first real accuracy number.

### Phase 3: live inference on the board
Only after Phases 1-2 confirm the footage and counts, push inference onto the
board's accelerator and run in real time. The acceleration path depends on the
board: RKNN (Rockchip), OpenVINO (Intel NPU), or HailoRT (Hailo M.2 card).


## One-month roadmap

| Week | Phase | Deliverable |
|---|---|---|
| 1 | Detection & counting pipeline | Laptop counter, 2 cameras, IN/OUT logged |
| 2 | Tracking, Re-ID & state machine | Full trip logged to SQLite per-stop |
| 3 | Edge deployment & API layer | ≥ 20 fps on Orange Pi 5 |
| 4 | Integration, dashboard & validation | 10-trip accuracy report, clean repo |

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Detection | YOLOv8n/s (Ultralytics) | Fast, ONNX-exportable, RKNN-convertible |
| Tracker | ByteTrack (built into Ultralytics) | No appearance features needed; works on limited hardware |
| Backend | SQLite (local) → Postgres (production) | One connection-string change to scale |
| Edge inference | RKNN Toolkit 2 + INT8 quantisation | Official RK3588S NPU SDK |
| Dashboard | Self-contained HTML + ECharts | No server; works offline on-bus; printable |

---

## Known limitations and next steps

- **Re-ID across a 60-second gap** (someone who exits and re-boards): not implemented in month 1. Occupancy math is still correct for this case; it produces one alighting and one boarding, which is accurate for load analysis. OSNet-based Re-ID is month 2.
- **Pose classification** (seated vs standee): deferred.
- **POS integration**: currently simulated with a configurable fare-evasion rate. Real integration requires the POS vendor API, which is a month-3 deliverable.
- **Coast counts are inferences**: `--no-coast` disables dead reckoning entirely for use cases that prefer a missed count over a potentially wrong one. The confidence figure in the dashboard always reflects this distinction.

---

## Hardware

| Role | Hardware |
|---|---|
| Edge device | Orange Pi 5 (RK3588S, 6 TOPS NPU) |
| Cameras | 2× ELP USB wide-angle, 1080p, 120° FOV |
| Dev machine | Any laptop running Python 3.10+ |

---

## Benchmark video

The test clip used during development (`test_1.mp4`, 402×300, 43 s) is from the [saimj7/People-Counting-in-Real-Time](https://github.com/saimj7/People-Counting-in-Real-Time) repository and is not included here due to file size. Download it and place it next to `cpcs_poc.py` to reproduce the benchmark results.

---

## Author

**Saikarthik Ramakrishnan**  
Second-year ECE, Shiv Nadar University Delhi  
Intern  
[github.com/Saikarthik-Ramakrishnan](https://github.com/Saikarthik-Ramakrishnan)

---

*This is a proof of concept. POS figures in the dashboard are simulated. Production deployment on a live fleet is a separate phase pending hardware integration and field validation.*
