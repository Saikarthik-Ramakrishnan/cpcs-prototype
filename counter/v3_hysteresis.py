"""
CPCS counter v3 — fixes track fragmentation at the counting line.

What changed from v2 and WHY:
  - The box carrying an ID would vanish right at the line, splitting one
    crossing across two track IDs, so it never counted.
  - Root cause: v2 fed YOLO a conf threshold of 0.35, which filtered out the
    low-confidence boxes BEFORE ByteTrack saw them. ByteTrack's whole job is to
    use those low-score boxes to keep a track alive through a fade-out. v2 was
    disabling the one feature that fixes this exact bug.

Three fixes:
  1. YOLO conf lowered to 0.10 so low-score boxes reach the tracker.
     new_track_thresh (0.25) still stops noise from spawning phantom tracks.
  2. track_buffer raised to 50 frames so a lost track waits longer to be
     re-matched to the same ID instead of a fresh one.
  3. Birth-to-death fallback: if a track dies without ever counting but was born
     on one side of the line and last seen committed to the other, count it on
     cleanup. Safety net for tracks that die just after crossing.

A tracker config file (bytetrack_cpcs.yaml) is written automatically next to
this script on first run, so you only need this one file.

Usage:
    python counter_v3.py                          # live webcam
    python counter_v3.py --source test_1.mp4      # benchmark on a clip
    python counter_v3.py --source test_1.mp4 --headless   # no window, fastest
"""

import argparse
import csv
import os
import time

import cv2
from ultralytics import YOLO

# ---------------- tunables ----------------
DEAD_ZONE   = 22      # px half-width of hysteresis band around the line
EMA_ALPHA   = 0.5     # centroid smoothing: higher = more responsive
MIN_AGE     = 2       # frames a track must exist before it may count
MODEL_CONF  = 0.10    # LOW on purpose: feeds low-score boxes to ByteTrack
STALE       = 40      # frames without detection before a track is dropped
# -------------------------------------------

TRACKER_YAML = "bytetrack_cpcs.yaml"

# ByteTrack config tuned for door counting. Written next to the script so you
# don't have to manage a second file by hand.
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
    """Write the tracker yaml next to this script if it isn't there yet."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, TRACKER_YAML)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(TRACKER_CONFIG)
    return path


def zone_of(y, line_y):
    """Return 'above', 'below', or None (inside the dead band)."""
    if y < line_y - DEAD_ZONE:
        return "above"
    if y > line_y + DEAD_ZONE:
        return "below"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="0 for webcam, or path to a video file")
    ap.add_argument("--headless", action="store_true",
                    help="no display window (benchmark mode)")
    ap.add_argument("--events", default="events.csv",
                    help="CSV file for per-event log")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)
    tracker_path = ensure_tracker_config()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)

    line_y = None
    boardings = 0
    alightings = 0

    # per-track state:
    #   ema        smoothed centroid y
    #   zone       last committed zone ('above'/'below'), retained inside band
    #   birth_zone zone when the track was first seen
    #   age        frames observed
    #   last_seen  frame index of last observation
    #   counted    True once this track has produced a count (prevents double)
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
                    st = {"ema": cy, "zone": z, "birth_zone": z,
                          "age": 0, "last_seen": frame_idx, "counted": False}
                    tracks[tid] = st
                else:
                    st["ema"] = EMA_ALPHA * cy + (1 - EMA_ALPHA) * st["ema"]
                    st["last_seen"] = frame_idx

                st["age"] += 1
                new_zone = zone_of(st["ema"], line_y)

                # live count on a full zone flip, after the track has matured
                if new_zone is not None and st["zone"] is not None \
                        and new_zone != st["zone"] and st["age"] >= MIN_AGE:
                    if st["zone"] == "above" and new_zone == "below":
                        fire("boarding", tid, st["ema"], "live")
                        st["counted"] = True
                    elif st["zone"] == "below" and new_zone == "above":
                        fire("alighting", tid, st["ema"], "live")
                        st["counted"] = True

                # hysteresis: only update committed zone when outside the band
                if new_zone is not None:
                    st["zone"] = new_zone

                if not args.headless:
                    sy = int(st["ema"])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (cx, sy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"id {tid}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # cleanup stale tracks — with birth-to-death fallback count
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
            cv2.imshow("CPCS v3 - counting line", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # flush any still-open tracks through the fallback at end of video
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
