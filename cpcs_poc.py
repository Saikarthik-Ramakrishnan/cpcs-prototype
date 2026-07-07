"""
CPCS PoC — capture + count + log a trip to SQLite.   (counter v6 inside)

Counting engine layers, in order of trust:
  live      zone transition observed while continuously tracked
  coast     track lost detection but its velocity-propagated position crossed
            the line (dead reckoning; inference, max 12-frame horizon)
  stitch    fragmented track reconnected by velocity prediction; the joined
            track then counts live or by fallback
  fallback  track born on one side of the line, died on the other

Every event is written to the DB with its method ("how"), so the dashboard
can report data confidence instead of an unverifiable perfection claim.

DEMO CONTROLS (video window focused):
    n   commit the current stop, roll to the next
    p   toggle simulated POS ticket counts
    q   end trip and quit

Run (recommended PoC settings — stronger detector, higher resolution):
    python cpcs_poc.py --source test_1.mp4 --model yolov8s.pt --imgsz 960 \
                       --route "47A" --bus "DL-1PC-4432"
Fast mode (weaker but quicker):
    python cpcs_poc.py --source test_1.mp4
Disable dead reckoning:
    python cpcs_poc.py --source test_1.mp4 --no-coast

Then:  python build_dashboard.py   ->  open cpcs_dashboard.html
"""

import argparse
import math
import os
import random
import sqlite3
from datetime import datetime

import cv2
from ultralytics import YOLO

from cpcs_geometry import CountingLine
from cpcs_config import load_config

# ---------------- counting tunables ----------------
DEAD_ZONE      = 22
EMA_ALPHA      = 0.5
MIN_AGE        = 2
MODEL_CONF     = 0.10
STALE          = 40
SNAP_GAP       = 4
STITCH_MIN_GAP = 2
STITCH_MAX_GAP = 25
STITCH_PRED_D  = 60
STITCH_BASE_D  = 30
STITCH_PER_FR  = 8
# --- dead reckoning (coast) ---
COAST_MAX_GAP  = 12    # propagate a lost track at most this many frames
COAST_MIN_VD   = 2.0   # px/frame across the line; slower tracks are not coasted
COAST_MIN_AGE  = 4     # only coast tracks with enough observed history

POS_FLAG_TOL   = 1

TRACKER_YAML = "bytetrack_cpcs.yaml"
TRACKER_CONFIG = """\
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 50
match_thresh: 0.85
fuse_score: True
"""


def ensure_tracker_config():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, TRACKER_YAML)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(TRACKER_CONFIG)
    return path


class DoorCounter:
    """v6 crossing engine: live + coast + stitch + fallback, all audited.

    Geometry is delegated to a CountingLine, so the tripwire can sit at any
    position and angle (calibrated per bus). Passing a plain int for `line`
    builds a horizontal line at that y for backward compatibility, in which
    case the arithmetic is identical to the original y-based counter.
    """

    def __init__(self, line, enable_coast=True):
        if isinstance(line, (int, float)):
            # legacy: horizontal line at y = line, spanning a wide frame
            self.line = CountingLine(0, int(line), 100000, int(line),
                                     dead_zone=DEAD_ZONE)
        else:
            self.line = line
        self.enable_coast = enable_coast
        self.tracks = {}
        self.frame_idx = 0

    def _try_stitch(self, cy, cx):
        best, best_score = None, float("inf")
        for otid, s in self.tracks.items():
            gap = self.frame_idx - s["last_seen"]
            if gap < STITCH_MIN_GAP or gap > STITCH_MAX_GAP:
                continue
            pred_y = s["cy"] + s["vy"] * gap
            pred_x = s["cx"] + s["vx"] * gap
            d_pred = math.hypot(pred_y - cy, pred_x - cx)
            d_raw = math.hypot(s["cy"] - cy, s["cx"] - cx)
            allow_raw = STITCH_BASE_D + STITCH_PER_FR * gap
            dir_ok = True
            if abs(s["vy"]) > 0.3:
                dir_ok = (s["vy"] > 0) == (cy > s["cy"])
            if (d_pred < STITCH_PRED_D) or (d_raw < allow_raw and dir_ok):
                score = min(d_pred, d_raw)
                if score < best_score:
                    best_score, best = score, otid
        if best is not None:
            s = self.tracks[best]
            inherited = (s["birth_zone"], s["zone"], s["counted"], s["age"])
            del self.tracks[best]
            return inherited
        return None

    def update(self, boxes_xyxy, ids):
        """Returns list of (direction, how, stitched) events this frame."""
        self.frame_idx = self.frame_idx + 1
        fired = []
        seen_now = set()

        for box, tid in zip(boxes_xyxy, ids):
            x1, y1, x2, y2 = box
            tid = int(tid)
            seen_now.add(tid)
            cy = (y1 + y2) / 2.0
            cx = (x1 + x2) / 2.0

            d = self.line.signed_distance(cx, cy)
            st = self.tracks.get(tid)
            if st is None:
                z = self.line.zone_from_d(d)
                birth_z, counted, age, stitched, inband = z, False, 0, False, False
                inh = self._try_stitch(cy, cx)
                if inh is not None:
                    birth_z, z, counted, age = inh
                    stitched = True
                elif z is None:
                    birth_z = "above" if d < 0 else "below"
                    z = birth_z
                    inband = True
                st = {"d_ema": d, "cx": cx, "cy": cy, "raw_y": cy,
                      "vx": 0.0, "vy": 0.0,
                      "zone": z, "birth_zone": birth_z, "age": age,
                      "last_seen": self.frame_idx, "counted": counted,
                      "stitched": stitched, "inband": inband}
                self.tracks[tid] = st
            else:
                gap = self.frame_idx - st["last_seen"]
                dyv = (cy - st["cy"]) / max(gap, 1)
                dxv = (cx - st["cx"]) / max(gap, 1)
                st["vy"] = 0.5 * dyv + 0.5 * st["vy"]
                st["vx"] = 0.5 * dxv + 0.5 * st["vx"]
                st["d_ema"] = d if gap > SNAP_GAP else \
                    EMA_ALPHA * d + (1 - EMA_ALPHA) * st["d_ema"]
                st["raw_y"] = cy
                st["cx"] = cx
                st["cy"] = cy
                st["last_seen"] = self.frame_idx

            st["age"] += 1
            nz = self.line.zone_from_d(st["d_ema"])
            if nz is not None and st["zone"] is not None and nz != st["zone"] \
                    and st["age"] >= MIN_AGE and not st["counted"]:
                if st["zone"] == "above" and nz == "below":
                    fired.append(("boarding", "live", st["stitched"]))
                    st["counted"] = True
                elif st["zone"] == "below" and nz == "above":
                    fired.append(("alighting", "live", st["stitched"]))
                    st["counted"] = True
            if nz is not None:
                st["zone"] = nz

        # --- dead reckoning: coast unseen tracks along their velocity ---
        if self.enable_coast:
            for tid, s in self.tracks.items():
                if tid in seen_now or s["counted"]:
                    continue
                gap = self.frame_idx - s["last_seen"]
                if gap < 1 or gap > COAST_MAX_GAP:
                    continue
                nx, ny = self.line.normal()
                vd = s["vx"] * nx + s["vy"] * ny   # velocity across the line
                if s["age"] < COAST_MIN_AGE or abs(vd) < COAST_MIN_VD:
                    continue
                pred_d = s["d_ema"] + vd * gap
                nz = self.line.zone_from_d(pred_d)
                if nz is not None and s["zone"] is not None and nz != s["zone"]:
                    if s["zone"] == "above" and nz == "below":
                        fired.append(("boarding", "coast", s["stitched"]))
                        s["counted"] = True
                        s["zone"] = nz
                    elif s["zone"] == "below" and nz == "above":
                        fired.append(("alighting", "coast", s["stitched"]))
                        s["counted"] = True
                        s["zone"] = nz

        # --- retire stale tracks with birth-to-death fallback ---
        stale = [t for t, s in self.tracks.items()
                 if self.frame_idx - s["last_seen"] > STALE]
        for t in stale:
            s = self.tracks[t]
            if not s["counted"] and s["birth_zone"] and s["zone"] \
                    and s["birth_zone"] != s["zone"]:
                if s["birth_zone"] == "above" and s["zone"] == "below":
                    fired.append(("boarding", "fallback", s["stitched"]))
                elif s["birth_zone"] == "below" and s["zone"] == "above":
                    fired.append(("alighting", "fallback", s["stitched"]))
            del self.tracks[t]

        return fired

    def flush(self):
        fired = []
        for s in self.tracks.values():
            if not s["counted"] and s["birth_zone"] and s["zone"] \
                    and s["birth_zone"] != s["zone"]:
                if s["birth_zone"] == "above" and s["zone"] == "below":
                    fired.append(("boarding", "fallback_eof", s["stitched"]))
                elif s["birth_zone"] == "below" and s["zone"] == "above":
                    fired.append(("alighting", "fallback_eof", s["stitched"]))
        self.tracks.clear()
        return fired


class TripRecorder:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS trips (
        trip_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        route      TEXT, bus_id TEXT,
        started_at TEXT, ended_at TEXT
    );
    CREATE TABLE IF NOT EXISTS stops (
        stop_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id         INTEGER, seq INTEGER, stop_name TEXT,
        committed_at    TEXT,
        boardings       INTEGER, alightings INTEGER,
        occupancy_after INTEGER,
        pos_count       INTEGER, discrepancy INTEGER, flagged INTEGER
    );
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id  INTEGER, stop_seq INTEGER,
        frame    INTEGER, direction TEXT, how TEXT, ts TEXT
    );
    """

    def __init__(self, db_path, route, bus_id):
        self.db = sqlite3.connect(db_path)
        self.db.executescript(self.SCHEMA)
        cur = self.db.execute(
            "INSERT INTO trips(route,bus_id,started_at) VALUES(?,?,?)",
            (route, bus_id, datetime.now().isoformat(timespec="seconds")))
        self.trip_id = cur.lastrowid
        self.db.commit()
        self.seq = 1
        self.occupancy = 0
        self.stop_boardings = 0
        self.stop_alightings = 0

    def record_event(self, frame, direction, how):
        if direction == "boarding":
            self.stop_boardings += 1
        else:
            self.stop_alightings += 1
        self.db.execute(
            "INSERT INTO events(trip_id,stop_seq,frame,direction,how,ts) "
            "VALUES(?,?,?,?,?,?)",
            (self.trip_id, self.seq, frame, direction, how,
             datetime.now().isoformat(timespec="seconds")))

    def commit_stop(self, stop_name=None, pos_count=None):
        self.occupancy += self.stop_boardings - self.stop_alightings
        self.occupancy = max(self.occupancy, 0)
        if pos_count is None:
            discrepancy, flagged = 0, 0
        else:
            discrepancy = self.stop_boardings - pos_count
            flagged = 1 if abs(discrepancy) > POS_FLAG_TOL else 0
        self.db.execute(
            "INSERT INTO stops(trip_id,seq,stop_name,committed_at,boardings,"
            "alightings,occupancy_after,pos_count,discrepancy,flagged) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (self.trip_id, self.seq, stop_name or f"Stop {self.seq}",
             datetime.now().isoformat(timespec="seconds"),
             self.stop_boardings, self.stop_alightings, self.occupancy,
             pos_count if pos_count is not None else -1,
             discrepancy, flagged))
        self.db.commit()
        row = (self.seq, self.stop_boardings, self.stop_alightings,
               self.occupancy, pos_count, discrepancy, flagged)
        self.seq += 1
        self.stop_boardings = 0
        self.stop_alightings = 0
        return row

    def end_trip(self):
        self.db.execute("UPDATE trips SET ended_at=? WHERE trip_id=?",
                        (datetime.now().isoformat(timespec="seconds"),
                         self.trip_id))
        self.db.commit()
        self.db.close()


def simulate_pos(camera_boardings):
    if camera_boardings == 0:
        return 0
    if random.random() < 0.35:
        return max(0, camera_boardings - random.randint(1, 3))
    return camera_boardings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--db", default="cpcs.db")
    ap.add_argument("--route", default="Demo Route")
    ap.add_argument("--bus", default="BUS-001")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--no-coast", action="store_true",
                    help="disable dead-reckoning counts")
    ap.add_argument("--config", default=None,
                    help="per-bus config.yaml (calibrated line, model, fps)")
    args = ap.parse_args()

    # config supplies per-bus defaults; explicit CLI flags still win
    cfg = load_config(args.config)
    route = args.route if args.route != "Demo Route" else cfg["bus"]["route"]
    bus = args.bus if args.bus != "BUS-001" else cfg["bus"]["bus_id"]
    model_name = args.model if args.model != "yolov8n.pt" \
        else cfg["detection"]["model"]
    imgsz = args.imgsz if args.imgsz != 640 else cfg["detection"]["imgsz"]
    enable_coast = (not args.no_coast) and cfg["runtime"].get("coast", True)
    cam_cfg = cfg["camera"]

    source = int(args.source) if args.source.isdigit() else args.source
    tracker_path = ensure_tracker_config()

    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)
    rec = TripRecorder(args.db, route, bus)
    counter = None
    line = None

    sim_pos = True
    calibrated = cam_cfg.get("line") is not None
    print("=" * 52)
    print(f"CPCS PoC capture v2  |  counter v6  |  trip {rec.trip_id}")
    print(f"model={model_name} imgsz={imgsz} "
          f"coast={'on' if enable_coast else 'off'}  "
          f"line={'calibrated' if calibrated else 'horizontal-mid'}")
    print("keys:  n = next stop   p = toggle POS sim   q = quit")
    print("=" * 52)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        if counter is None:
            if cam_cfg.get("line") is not None:
                x1, y1, x2, y2 = cam_cfg["line"]
                line = CountingLine(x1, y1, x2, y2,
                                    dead_zone=cam_cfg.get("dead_zone", DEAD_ZONE),
                                    flip=cam_cfg.get("flip", False))
            else:
                line = CountingLine.horizontal_mid(w, h, dead_zone=DEAD_ZONE)
            counter = DoorCounter(line, enable_coast=enable_coast)

        res = model.track(frame, classes=[0], conf=MODEL_CONF,
                          imgsz=imgsz, persist=True,
                          tracker=tracker_path, verbose=False)
        boxes = res[0].boxes
        xyxy, ids = [], []
        if boxes.id is not None:
            xyxy = boxes.xyxy.tolist()
            ids = boxes.id.tolist()
            for b, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = b
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"id {int(tid)}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for direction, how, _st in counter.update(xyxy, ids):
            rec.record_event(frame_idx, direction, how)

        cv2.line(frame, (int(line.x1), int(line.y1)),
                 (int(line.x2), int(line.y2)), (0, 255, 255), 2)
        hud = (f"stop {rec.seq}  IN {rec.stop_boardings}  "
               f"OUT {rec.stop_alightings}  occ {rec.occupancy}  "
               f"POSsim {'on' if sim_pos else 'off'}")
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imshow("CPCS PoC - n=stop  p=POS  q=quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            sim_pos = not sim_pos
        elif key == ord("n"):
            pos = simulate_pos(rec.stop_boardings) if sim_pos else None
            row = rec.commit_stop(pos_count=pos)
            flag = "  << FLAGGED" if row[6] else ""
            print(f"stop {row[0]}: in {row[1]} out {row[2]} occ {row[3]} "
                  f"pos {row[4]} d {row[5]}{flag}")

    if counter is not None:
        for direction, how, _ in counter.flush():
            rec.record_event(frame_idx, direction, how)
    if rec.stop_boardings or rec.stop_alightings:
        pos = simulate_pos(rec.stop_boardings) if sim_pos else None
        rec.commit_stop(pos_count=pos)
    rec.end_trip()

    cap.release()
    cv2.destroyAllWindows()
    print("=" * 52)
    print(f"trip {rec.trip_id} written to {args.db}")
    print("build the report:  python build_dashboard.py")
    print("=" * 52)


if __name__ == "__main__":
    main()
