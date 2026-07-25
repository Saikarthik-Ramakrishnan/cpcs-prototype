"""
CPCS PoC — capture + count + log a trip to SQLite.

Pipeline:  camera/clip -> DoorCounter (v5 logic) -> TripRecorder (SQLite)

This is the "Path A" backbone: crossings stop being throwaway variables and
become structured per-stop records a dashboard can read.

DEMO CONTROLS (keys, while the video window is focused):
    n   commit the current stop and roll on to the next
        (writes boardings/alightings/occupancy for the stop just finished)
    p   toggle "simulate POS" — injects a realistic ticket-machine count with
        occasional shortfalls, so the discrepancy-flagging is visible in a demo
        without a real ticket machine wired up
    q   end the trip and quit

Run:
    python cpcs_poc.py --source test_1.mp4 --route "47A" --bus "DL-1PC-4432"
    python cpcs_poc.py --source 0 --imgsz 1280      # live webcam as the "door"

Then view it:
    streamlit run dashboard.py

The SQLite file (cpcs.db by default) is the hand-off between the two.
"""

import argparse
import math
import os
import random
import sqlite3
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ---------------- counting tunables (ported from v5) ----------------
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

# discrepancy flagging
POS_FLAG_TOL   = 1     # |camera_boardings - pos_count| > this => flag the stop

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


def zone_of(y, line_y):
    if y < line_y - DEAD_ZONE:
        return "above"
    if y > line_y + DEAD_ZONE:
        return "below"
    return None


class DoorCounter:
    """v5 crossing logic as a reusable engine.

    Feed it the tracked boxes for each frame; it returns the crossing events
    that fired this frame as a list of ('boarding'|'alighting', how) tuples.
    """

    def __init__(self, line_y):
        self.line_y = line_y
        self.tracks = {}
        self.frame_idx = 0

    def _try_stitch(self, cy, cx):
        best, best_score = None, float("inf")
        for otid, s in self.tracks.items():
            gap = self.frame_idx - s["last_seen"]
            if gap < STITCH_MIN_GAP or gap > STITCH_MAX_GAP:
                continue
            pred_y = s["ema"] + s["vy"] * gap
            pred_x = s["cx"] + s["vx"] * gap
            d_pred = math.hypot(pred_y - cy, pred_x - cx)
            d_raw = math.hypot(s["ema"] - cy, s["cx"] - cx)
            allow_raw = STITCH_BASE_D + STITCH_PER_FR * gap
            dir_ok = True
            if abs(s["vy"]) > 0.3:
                dir_ok = (s["vy"] > 0) == (cy > s["ema"])
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
        """boxes_xyxy: list of [x1,y1,x2,y2]; ids: list of int track ids.
        Returns list of (event, how, stitched) for crossings fired this frame."""
        self.frame_idx += 1
        fired = []
        seen_now = set()

        for box, tid in zip(boxes_xyxy, ids):
            x1, y1, x2, y2 = box
            tid = int(tid)
            seen_now.add(tid)
            cy = (y1 + y2) / 2.0
            cx = (x1 + x2) / 2.0

            st = self.tracks.get(tid)
            if st is None:
                z = zone_of(cy, self.line_y)
                birth_z, counted, age, stitched, inband = z, False, 0, False, False
                inh = self._try_stitch(cy, cx)
                if inh is not None:
                    birth_z, z, counted, age = inh
                    stitched = True
                elif z is None:
                    birth_z = "above" if cy < self.line_y else "below"
                    z = birth_z
                    inband = True
                st = {"ema": cy, "cx": cx, "raw_y": cy, "vx": 0.0, "vy": 0.0,
                      "zone": z, "birth_zone": birth_z, "age": age,
                      "last_seen": self.frame_idx, "counted": counted,
                      "stitched": stitched, "inband": inband}
                self.tracks[tid] = st
            else:
                gap = self.frame_idx - st["last_seen"]
                dy = (cy - st["raw_y"]) / max(gap, 1)
                dx = (cx - st["cx"]) / max(gap, 1)
                st["vy"] = 0.5 * dy + 0.5 * st["vy"]
                st["vx"] = 0.5 * dx + 0.5 * st["vx"]
                st["ema"] = cy if gap > SNAP_GAP else \
                    EMA_ALPHA * cy + (1 - EMA_ALPHA) * st["ema"]
                st["raw_y"] = cy
                st["cx"] = cx
                st["last_seen"] = self.frame_idx

            st["age"] += 1
            nz = zone_of(st["ema"], self.line_y)
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

        # retire stale tracks with birth-to-death fallback
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
        """Call at end of video: fire fallback for any still-open crossings."""
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
    """Owns the SQLite DB and the trip/stop/occupancy/POS state machine."""

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
        """Close the current stop, roll occupancy forward, write the row."""
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
    """Fake a ticket machine: usually matches, sometimes undercounts (evasion)."""
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
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    tracker_path = ensure_tracker_config()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    rec = TripRecorder(args.db, args.route, args.bus)
    counter = None
    sim_pos = True

    print("=" * 46)
    print("CPCS PoC capture  |  trip", rec.trip_id)
    print("keys:  n = next stop   p = toggle POS sim   q = quit")
    print("=" * 46)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        if counter is None:
            counter = DoorCounter(line_y=h // 2)

        res = model.track(frame, classes=[0], conf=MODEL_CONF,
                          imgsz=args.imgsz, persist=True,
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

        for direction, how, _stitched in counter.update(xyxy, ids):
            rec.record_event(frame_idx, direction, how)

        line_y = h // 2
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        hud = (f"stop {rec.seq}  IN {rec.stop_boardings}  "
               f"OUT {rec.stop_alightings}  occ {rec.occupancy}  "
               f"POSsim {'on' if sim_pos else 'off'}")
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imshow("CPCS PoC - press n=stop  p=POS  q=quit", frame)

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

    # end of clip: flush counter, commit the final open stop, close trip
    if counter is not None:
        for direction, how, _ in counter.flush():
            rec.record_event(frame_idx, direction, how)
    if rec.stop_boardings or rec.stop_alightings:
        pos = simulate_pos(rec.stop_boardings) if sim_pos else None
        rec.commit_stop(pos_count=pos)
    rec.end_trip()

    cap.release()
    cv2.destroyAllWindows()
    print("=" * 46)
    print(f"trip {rec.trip_id} written to {args.db}")
    print(f"view it:  streamlit run dashboard.py")
    print("=" * 46)


if __name__ == "__main__":
    main()
