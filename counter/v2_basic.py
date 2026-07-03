"""
CPCS counter v2 — hysteresis crossing logic + benchmark mode.

Usage:
    python counter_v2.py                          # live webcam
    python counter_v2.py --source clip.mp4        # benchmark on a recorded clip
    python counter_v2.py --source clip.mp4 --headless   # no GUI, fastest, prints summary

Counting events are appended to events.csv (frame, track_id, event, y).
"""

import argparse
import csv
import time

import cv2
from ultralytics import YOLO

# ---------------- tunables ----------------
DEAD_ZONE = 25      # px half-width of hysteresis band around the line
EMA_ALPHA = 0.5     # centroid smoothing: higher = more responsive, lower = smoother
MIN_AGE   = 3       # frames a track must exist before it may count
CONF      = 0.35    # detection confidence threshold
STALE     = 30      # frames without detection before a track's state is dropped
# -------------------------------------------


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

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)

    line_y = None
    boardings = 0
    alightings = 0

    # per-track state: {tid: {"ema": float, "zone": str|None, "age": int, "last_seen": int}}
    tracks = {}
    frame_idx = 0
    t0 = time.time()

    ev_file = open(args.events, "w", newline="")
    ev = csv.writer(ev_file)
    ev.writerow(["frame", "track_id", "event", "smoothed_y"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        if line_y is None:
            line_y = h // 2

        results = model.track(frame, classes=[0], conf=CONF,
                              persist=True, tracker="bytetrack.yaml",
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
                    # new track: initialise EMA at first observation
                    st = {"ema": cy, "zone": zone_of(cy, line_y),
                          "age": 0, "last_seen": frame_idx}
                    tracks[tid] = st
                else:
                    st["ema"] = EMA_ALPHA * cy + (1 - EMA_ALPHA) * st["ema"]
                    st["last_seen"] = frame_idx

                st["age"] += 1
                new_zone = zone_of(st["ema"], line_y)

                # count only on a full zone flip, after the track has matured
                if new_zone is not None and st["zone"] is not None \
                        and new_zone != st["zone"] and st["age"] >= MIN_AGE:
                    if st["zone"] == "above" and new_zone == "below":
                        boardings += 1
                        ev.writerow([frame_idx, tid, "boarding", round(st["ema"], 1)])
                    elif st["zone"] == "below" and new_zone == "above":
                        alightings += 1
                        ev.writerow([frame_idx, tid, "alighting", round(st["ema"], 1)])

                # update zone only when outside the band — inside the band the
                # previous committed zone is retained (this is the hysteresis)
                if new_zone is not None:
                    st["zone"] = new_zone

                if not args.headless:
                    sy = int(st["ema"])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (cx, sy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"id {tid}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # drop stale tracks so IDs recycled by the tracker don't inherit old state
        stale = [t for t, s in tracks.items()
                 if frame_idx - s["last_seen"] > STALE]
        for t in stale:
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
            cv2.imshow("CPCS v2 - counting line", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - t0
    cap.release()
    ev_file.close()
    if not args.headless:
        cv2.destroyAllWindows()

    print("=" * 40)
    print(f"frames processed : {frame_idx}")
    print(f"avg fps          : {frame_idx / elapsed:.1f}")
    print(f"boardings  (IN)  : {boardings}")
    print(f"alightings (OUT) : {alightings}")
    print(f"event log        : {args.events}")
    if is_file:
        print("compare the above against your hand count of the clip")
    print("=" * 40)


if __name__ == "__main__":
    main()
