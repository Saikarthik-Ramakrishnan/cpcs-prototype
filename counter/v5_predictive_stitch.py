"""
CPCS counter v5 — velocity-predicted stitching + full self-diagnostics.

Why v4 failed to change anything:
  Its stitch gate demanded the reborn box appear within 70px of where the old
  track DIED. But during a 10-15 frame detection void, a person moving ~10px
  per frame travels 100-150px. Every candidate was silently rejected as "too
  far", the stitch never fired, and v4 behaved exactly like v3.

v5 fixes:
  1. PREDICTIVE STITCH: each track carries a smoothed velocity (px/frame).
     A new track stitches to a quiet one if it appears near where that track
     WOULD BE NOW (last position + velocity * gap). The gate scales with the
     length of the void instead of being a fixed radius.
  2. IN-BAND BIRTHS: a track born inside the dead zone previously had
     birth_zone=None and could never count. Now it takes the side of the LINE
     it was born on.
  3. DOUBLE-COUNT GUARD: fragments of already-counted people stitch and
     inherit counted=True instead of becoming fresh countable tracks.
  4. INHERITED AGE: stitched tracks skip the MIN_AGE wait.
  5. DIAGNOSTICS: version banner, --save-video annotated mp4,
     stitch_debug.log (every stitch attempt + reason), tracks.csv
     (every track's life story).
  6. --imgsz to run inference at higher resolution (attacks the detection
     void itself; use 1280 on low-res clips).

Usage:
    python counter_v5.py --source test_1.mp4 --imgsz 1280 --save-video annotated.mp4
    python counter_v5.py --source test_1.mp4 --headless
"""

import argparse
import csv
import math
import os
import time

import cv2
from ultralytics import YOLO

VERSION = "CPCS counter v5"

# ---------------- tunables ----------------
DEAD_ZONE       = 22    # px half-width of hysteresis band around the line
EMA_ALPHA       = 0.5   # centroid smoothing
MIN_AGE         = 2     # frames before a (non-stitched) track may count
MODEL_CONF      = 0.10  # low, so ByteTrack receives low-score boxes
STALE           = 40    # frames without detection before a track is retired
SNAP_GAP        = 4     # same-ID rematch after a gap > this snaps EMA to raw

# --- stitching gates ---
STITCH_MIN_GAP  = 2     # candidate must be quiet at least this many frames
STITCH_MAX_GAP  = 25    # ...and at most this many
STITCH_PRED_D   = 60    # px radius around the velocity-predicted position
STITCH_BASE_D   = 30    # raw-distance fallback: base allowance...
STITCH_PER_FR   = 8     # ...plus this many px per frame of gap
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
    ap.add_argument("--tracks-log", default="tracks.csv")
    ap.add_argument("--stitch-log", default="stitch_debug.log")
    ap.add_argument("--save-video", default=None,
                    help="write an annotated mp4 to this path")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference resolution; use 1280 for low-res clips")
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    print("=" * 40)
    print(f"{VERSION} | model={args.model} imgsz={args.imgsz}")
    print("=" * 40)

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)
    tracker_path = ensure_tracker_config()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)

    line_y = None
    boardings = 0
    alightings = 0
    n_stitches = 0
    n_inband_births = 0

    # tracks[tid] = { ema, cx, raw_y, vx, vy, zone, birth_zone, age,
    #                 born, last_seen, counted, stitched, inband }
    tracks = {}
    frame_idx = 0
    writer = None
    recent_stitch_msgs = []   # (expires_frame, text) for on-video annotation
    t0 = time.time()

    ev_file = open(args.events, "w", newline="")
    ev = csv.writer(ev_file)
    ev.writerow(["frame", "track_id", "event", "smoothed_y", "how", "stitched"])

    tr_file = open(args.tracks_log, "w", newline="")
    tr = csv.writer(tr_file)
    tr.writerow(["track_id", "born_frame", "died_frame", "birth_zone",
                 "final_zone", "counted", "stitched", "inband_birth", "n_obs"])

    sd = open(args.stitch_log, "w")

    def fire(event, tid, y, how, stitched):
        nonlocal boardings, alightings
        if event == "boarding":
            boardings += 1
        else:
            alightings += 1
        ev.writerow([frame_idx, tid, event, round(y, 1), how, stitched])

    def log_track_death(tid, s, died_frame):
        tr.writerow([tid, s["born"], died_frame, s["birth_zone"], s["zone"],
                     s["counted"], s["stitched"], s["inband"], s["age"]])

    def try_stitch(cy, cx):
        """Return (birth_zone, zone, counted, age) inherited from the best
        recently-quiet track this detection continues, or None."""
        nonlocal n_stitches
        best = None
        best_score = float("inf")
        for otid, s in tracks.items():
            gap = frame_idx - s["last_seen"]
            if gap < STITCH_MIN_GAP or gap > STITCH_MAX_GAP:
                continue
            pred_y = s["ema"] + s["vy"] * gap
            pred_x = s["cx"] + s["vx"] * gap
            d_pred = math.hypot(pred_y - cy, pred_x - cx)
            d_raw = math.hypot(s["ema"] - cy, s["cx"] - cx)
            allow_raw = STITCH_BASE_D + STITCH_PER_FR * gap
            if abs(s["vy"]) > 0.3:
                dir_ok = (s["vy"] > 0) == (cy > s["ema"])
            else:
                dir_ok = True
            accept = (d_pred < STITCH_PRED_D) or (d_raw < allow_raw and dir_ok)
            sd.write(f"f{frame_idx} new@({cy:.0f},{cx:.0f}) cand id{otid} "
                     f"gap={gap} d_pred={d_pred:.0f} d_raw={d_raw:.0f} "
                     f"allow_raw={allow_raw:.0f} dir_ok={dir_ok} "
                     f"-> {'ACCEPT' if accept else 'reject'}\n")
            if accept:
                score = min(d_pred, d_raw)
                if score < best_score:
                    best_score = score
                    best = otid
        if best is not None:
            s = tracks[best]
            inherited = (s["birth_zone"], s["zone"], s["counted"], s["age"])
            log_track_death(best, s, s["last_seen"])
            del tracks[best]
            n_stitches += 1
            recent_stitch_msgs.append(
                (frame_idx + 20, f"STITCH id{best} -> new @f{frame_idx}"))
            sd.write(f"f{frame_idx} STITCHED to id{best} "
                     f"(score={best_score:.0f})\n")
            return inherited
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        if line_y is None:
            line_y = h // 2
        if args.save_video and writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            writer = cv2.VideoWriter(args.save_video,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))

        annotate = (not args.headless) or (writer is not None)

        results = model.track(frame, classes=[0], conf=MODEL_CONF,
                              imgsz=args.imgsz, persist=True,
                              tracker=tracker_path, verbose=False)
        boxes = results[0].boxes

        if boxes.id is not None:
            for box, tid in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = box.tolist()
                tid = int(tid)
                cy = (y1 + y2) / 2.0
                cx = (x1 + x2) / 2.0

                st = tracks.get(tid)
                if st is None:
                    z = zone_of(cy, line_y)
                    birth_z = z
                    counted = False
                    age = 0
                    stitched = False
                    inband = False
                    inh = try_stitch(cy, cx)
                    if inh is not None:
                        birth_z, z, counted, age = inh
                        stitched = True
                    elif z is None:
                        # born inside the band, no continuation found:
                        # take the side of the LINE it appeared on
                        birth_z = "above" if cy < line_y else "below"
                        z = birth_z
                        inband = True
                        n_inband_births += 1
                    st = {"ema": cy, "cx": cx, "raw_y": cy,
                          "vx": 0.0, "vy": 0.0,
                          "zone": z, "birth_zone": birth_z,
                          "age": age, "born": frame_idx,
                          "last_seen": frame_idx, "counted": counted,
                          "stitched": stitched, "inband": inband}
                    tracks[tid] = st
                else:
                    gap = frame_idx - st["last_seen"]
                    dy = (cy - st["raw_y"]) / max(gap, 1)
                    dx = (cx - st["cx"]) / max(gap, 1)
                    st["vy"] = 0.5 * dy + 0.5 * st["vy"]
                    st["vx"] = 0.5 * dx + 0.5 * st["vx"]
                    if gap > SNAP_GAP:
                        st["ema"] = cy      # stale smooth value: snap to raw
                    else:
                        st["ema"] = EMA_ALPHA * cy + (1 - EMA_ALPHA) * st["ema"]
                    st["raw_y"] = cy
                    st["cx"] = cx
                    st["last_seen"] = frame_idx

                st["age"] += 1
                new_zone = zone_of(st["ema"], line_y)

                if new_zone is not None and st["zone"] is not None \
                        and new_zone != st["zone"] and st["age"] >= MIN_AGE \
                        and not st["counted"]:
                    if st["zone"] == "above" and new_zone == "below":
                        fire("boarding", tid, st["ema"], "live", st["stitched"])
                        st["counted"] = True
                    elif st["zone"] == "below" and new_zone == "above":
                        fire("alighting", tid, st["ema"], "live", st["stitched"])
                        st["counted"] = True

                if new_zone is not None:
                    st["zone"] = new_zone

                if annotate:
                    sy = int(st["ema"])
                    color = (255, 0, 255) if st["stitched"] else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), color, 2)
                    cv2.circle(frame, (int(cx), sy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"id {tid}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # retire stale tracks through the birth-to-death fallback
        stale = [t for t, s in tracks.items()
                 if frame_idx - s["last_seen"] > STALE]
        for t in stale:
            s = tracks[t]
            if not s["counted"] and s["birth_zone"] is not None \
                    and s["zone"] is not None and s["birth_zone"] != s["zone"]:
                if s["birth_zone"] == "above" and s["zone"] == "below":
                    fire("boarding", t, s["ema"], "fallback", s["stitched"])
                    s["counted"] = True
                elif s["birth_zone"] == "below" and s["zone"] == "above":
                    fire("alighting", t, s["ema"], "fallback", s["stitched"])
                    s["counted"] = True
            log_track_death(t, s, s["last_seen"])
            del tracks[t]

        if annotate:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
            cv2.line(frame, (0, line_y - DEAD_ZONE), (w, line_y - DEAD_ZONE),
                     (0, 160, 160), 1)
            cv2.line(frame, (0, line_y + DEAD_ZONE), (w, line_y + DEAD_ZONE),
                     (0, 160, 160), 1)
            cv2.putText(frame, f"IN: {boardings}   OUT: {alightings}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            recent_stitch_msgs[:] = [m for m in recent_stitch_msgs
                                     if m[0] > frame_idx]
            for i, (_, msg) in enumerate(recent_stitch_msgs[-3:]):
                cv2.putText(frame, msg, (10, 55 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if writer is not None:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("CPCS v5 - counting line", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # end of video: flush open tracks through the fallback
    for t, s in tracks.items():
        if not s["counted"] and s["birth_zone"] is not None \
                and s["zone"] is not None and s["birth_zone"] != s["zone"]:
            if s["birth_zone"] == "above" and s["zone"] == "below":
                fire("boarding", t, s["ema"], "fallback_eof", s["stitched"])
                s["counted"] = True
            elif s["birth_zone"] == "below" and s["zone"] == "above":
                fire("alighting", t, s["ema"], "fallback_eof", s["stitched"])
                s["counted"] = True
        log_track_death(t, s, s["last_seen"])

    elapsed = time.time() - t0
    cap.release()
    if writer is not None:
        writer.release()
    ev_file.close()
    tr_file.close()
    sd.close()
    if not args.headless:
        cv2.destroyAllWindows()

    print("=" * 40)
    print(f"{VERSION}")
    print(f"frames processed : {frame_idx}")
    print(f"avg fps          : {frame_idx / max(elapsed, 1e-6):.1f}")
    print(f"boardings  (IN)  : {boardings}")
    print(f"alightings (OUT) : {alightings}")
    print(f"stitches made    : {n_stitches}")
    print(f"in-band births   : {n_inband_births}")
    print(f"event log        : {args.events}")
    print(f"track log        : {args.tracks_log}")
    print(f"stitch debug     : {args.stitch_log}")
    if args.save_video:
        print(f"annotated video  : {args.save_video}")
    if is_file:
        print("compare the above against your hand count of the clip")
    print("=" * 40)


if __name__ == "__main__":
    main()
