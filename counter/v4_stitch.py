"""
CPCS counter v4 — stitches fragmented tracks across a detection void.

The problem v3 could not solve:
  For 2 people, YOLO produced NO box at the line (a total detection void).
  Track A died 'above' before the line; a fresh Track B was born 'below' after
  crossing. The crossing happened invisibly between two IDs, so no single-track
  logic could see it.

The fix — TRACK STITCHING:
  When a new track is born, scan the tracks that have just gone quiet (stopped
  updating 2..STITCH_FRAMES frames ago). If one is close in space, consistent in
  direction, and not yet counted, the new track is treated as its continuation:
  it inherits that track's birth_zone and committed zone, and the old track is
  merged away. The new track then completes the crossing the old one started.

  NOTE (why we scan active tracks, not a delayed pool): the new track appears
  only a few frames after the old one goes quiet, long before the old one would
  be retired as stale. The continuation candidate is still an active track.

Gates are deliberately tight to avoid stitching two different people.

Usage:
    python counter_v4.py --source test_1.mp4
    python counter_v4.py --source test_1.mp4 --headless
"""

import argparse
import csv
import os
import time

import cv2
from ultralytics import YOLO

# ---------------- tunables ----------------
DEAD_ZONE     = 22     # px half-width of hysteresis band around the line
EMA_ALPHA     = 0.5    # centroid smoothing
MIN_AGE       = 2      # frames a track must exist before it may count
MODEL_CONF    = 0.10   # low, so ByteTrack gets low-score boxes
STALE         = 40     # frames without detection before a track is dropped

# --- track stitching ---
STITCH_MIN_GAP = 2     # a candidate must have been quiet at least this many frames
STITCH_FRAMES  = 20    # ...and at most this many frames
STITCH_DIST    = 70    # px between quiet point and birth point to allow a stitch
# -------------------------------------------

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--events", default="events.csv")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)
    tracker_path = ensure_tracker_config()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)

    line_y = None
    boardings = 0
    alightings = 0

    tracks = {}
    frame_idx = 0
    t0 = time.time()

    ev_file = open(args.events, "w", newline="")
    ev = csv.writer(ev_file)
    ev.writerow(["frame", "track_id", "event", "smoothed_y", "how"])

    def fire(event, tid, y, how):
        nonlocal boardings, alightings
        if event == "boarding":
            boardings += 1
        else:
            alightings += 1
        ev.writerow([frame_idx, tid, event, round(y, 1), how])

    def try_stitch(cy, cx):
        """Find a recently-quiet active track this new detection continues.
        Returns (birth_zone, zone) to inherit, or None."""
        best = None
        best_d = STITCH_DIST + 1
        for otid, os_ in tracks.items():
            gap = frame_idx - os_["last_seen"]
            if gap < STITCH_MIN_GAP or gap > STITCH_FRAMES:
                continue
            if os_["counted"]:
                continue
            dist = ((os_["ema"] - cy) ** 2 + (os_["cx"] - cx) ** 2) ** 0.5
            if dist > STITCH_DIST:
                continue
            new_dir = 1 if cy > os_["ema"] else -1
            if os_["vy"] != 0 and os_["vy"] != new_dir:
                continue
            if dist < best_d:
                best_d = dist
                best = otid
        if best is not None:
            bz = tracks[best]["birth_zone"]
            z = tracks[best]["zone"]
            del tracks[best]
            return bz, z
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        if line_y is None:
            line_y = h // 2

        results = model.track(frame, classes=[0], conf=MODEL_CONF,
                              persist=True, tracker=tracker_path,
                              verbose=False)
        boxes = results[0].boxes

        if boxes.id is not None:
            for box, tid in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = box.tolist()
                tid = int(tid)
                cy = (y1 + y2) / 2.0
                cx = int((x1 + x2) / 2)

                st = tracks.get(tid)
                if st is None:
                    z = zone_of(cy, line_y)
                    birth_z = z
                    stitched = try_stitch(cy, cx)
                    if stitched is not None:
                        birth_z, z = stitched
                    st = {"ema": cy, "cx": cx, "vy": 0, "zone": z,
                          "birth_zone": birth_z, "age": 0,
                          "last_seen": frame_idx, "counted": False}
                    tracks[tid] = st
                else:
                    new_ema = EMA_ALPHA * cy + (1 - EMA_ALPHA) * st["ema"]
                    dy = new_ema - st["ema"]
                    st["vy"] = 1 if dy > 0.5 else (-1 if dy < -0.5 else st["vy"])
                    st["ema"] = new_ema
                    st["cx"] = cx
                    st["last_seen"] = frame_idx

                st["age"] += 1
                new_zone = zone_of(st["ema"], line_y)

                if new_zone is not None and st["zone"] is not None \
                        and new_zone != st["zone"] and st["age"] >= MIN_AGE:
                    if st["zone"] == "above" and new_zone == "below":
                        fire("boarding", tid, st["ema"], "live")
                        st["counted"] = True
                    elif st["zone"] == "below" and new_zone == "above":
                        fire("alighting", tid, st["ema"], "live")
                        st["counted"] = True

                if new_zone is not None:
                    st["zone"] = new_zone

                if not args.headless:
                    sy = int(st["ema"])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (cx, sy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"id {tid}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        stale = [t for t, s in tracks.items()
                 if frame_idx - s["last_seen"] > STALE]
        for t in stale:
            s = tracks[t]
            if not s["counted"] and s["birth_zone"] is not None \
                    and s["zone"] is not None and s["birth_zone"] != s["zone"]:
                if s["birth_zone"] == "above" and s["zone"] == "below":
                    fire("boarding", t, s["ema"], "fallback")
                elif s["birth_zone"] == "below" and s["zone"] == "above":
                    fire("alighting", t, s["ema"], "fallback")
            del tracks[t]

        if not args.headless:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
            cv2.line(frame, (0, line_y - DEAD_ZONE), (w, line_y - DEAD_ZONE),
                     (0, 160, 160), 1)
            cv2.line(frame, (0, line_y + DEAD_ZONE), (w, line_y + DEAD_ZONE),
                     (0, 160, 160), 1)
            cv2.putText(frame, f"IN: {boardings}   OUT: {alightings}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            cv2.imshow("CPCS v4 - counting line", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    for t, s in tracks.items():
        if not s["counted"] and s["birth_zone"] is not None \
                and s["zone"] is not None and s["birth_zone"] != s["zone"]:
            if s["birth_zone"] == "above" and s["zone"] == "below":
                fire("boarding", t, s["ema"], "fallback_eof")
            elif s["birth_zone"] == "below" and s["zone"] == "above":
                fire("alighting", t, s["ema"], "fallback_eof")

    elapsed = time.time() - t0
    cap.release()
    ev_file.close()
    if not args.headless:
        cv2.destroyAllWindows()

    print("=" * 40)
    print(f"frames processed : {frame_idx}")
    print(f"avg fps          : {frame_idx / max(elapsed, 1e-6):.1f}")
    print(f"boardings  (IN)  : {boardings}")
    print(f"alightings (OUT) : {alightings}")
    print(f"event log        : {args.events}  (check the 'how' column)")
    if is_file:
        print("compare the above against your hand count of the clip")
    print("=" * 40)


if __name__ == "__main__":
    main()
