"""Replay recorded detections through DoorCounter — inference-free evaluation.

Detection + tracking is the slow part (minutes per clip on a laptop) and it is
deterministic for a fixed model/imgsz/tracker. So we run it once, dump every
tracked box to a CSV, and then replay that CSV through the counter as many
times as we like. A parameter sweep that would take days of inference takes
seconds, and every candidate sees byte-identical detections, so any difference
in the score is caused by the counter and nothing else.

Dump format (eval/dump_tracks.py writes it):
    frame,tid,cx,cy,w,h,conf

Usage:
    python eval/dump_tracks.py --source test_1.mp4 --model yolov8s.pt \
                               --imgsz 960 --out eval/dumps/test_1_s960.csv
    python eval/replay.py --dump eval/dumps/test_1_s960.csv \
                          --labels eval/labels/test_1.json
"""

import argparse
import collections
import csv
import importlib.util
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cpcs_geometry import CountingLine  # noqa: E402


def _load_poc():
    """Import cpcs_poc without requiring cv2/ultralytics to be importable."""
    import types
    for name, mod in (("cv2", types.ModuleType("cv2")),
                      ("ultralytics", types.ModuleType("ultralytics"))):
        if name not in sys.modules:
            if name == "ultralytics":
                mod.YOLO = object
            sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location(
        "cpcs_poc", os.path.join(_ROOT, "cpcs_poc.py"))
    poc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(poc)
    return poc


poc = _load_poc()
DoorCounter = poc.DoorCounter
CounterParams = poc.CounterParams


def load_dump(path):
    """-> (frames_dict {frame: [(box, tid), ...]}, n_frames, width, height)"""
    per_frame = collections.defaultdict(list)
    n_frames = 0
    with open(path) as f:
        rd = csv.DictReader(f)
        meta = {}
        for row in rd:
            fr = int(row["frame"])
            n_frames = max(n_frames, fr)
            cx, cy = float(row["cx"]), float(row["cy"])
            w, h = float(row["w"]), float(row["h"])
            per_frame[fr].append(
                ([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                 int(row["tid"])))
    hdr = _read_meta(path)
    return per_frame, max(n_frames, hdr.get("n_frames", 0)), hdr


def _read_meta(path):
    """Optional sidecar JSON with clip geometry (written by dump_tracks.py)."""
    side = os.path.splitext(path)[0] + ".meta.json"
    if os.path.exists(side):
        with open(side) as f:
            return json.load(f)
    return {}


def run(dump_path, params=None, enable_coast=True, line=None,
        dead_zone=None, width=None, height=None):
    """Replay a dump; returns the event list [(frame, direction, how)]."""
    per_frame, n_frames, meta = load_dump(dump_path)
    w = width or meta.get("width", 402)
    h = height or meta.get("height", 300)
    if line is None:
        dz = dead_zone if dead_zone is not None else poc.DEAD_ZONE
        line = CountingLine.horizontal_mid(w, h, dead_zone=dz)
    dc = DoorCounter(line, enable_coast=enable_coast, params=params)
    events = []
    for fr in range(1, n_frames + 1):
        dets = per_frame.get(fr, [])
        boxes = [b for b, _ in dets]
        ids = [t for _, t in dets]
        for direction, how, _st in dc.update(boxes, ids):
            events.append((fr, direction, how))
    for direction, how, _st in dc.flush():
        events.append((n_frames, direction, how))
    return events


# ---------------- scoring (same rules as eval/validate.py) ----------------

def load_labels(path):
    with open(path) as f:
        d = json.load(f)
    return [e for e in d["events"] if not e.get("staff", False)]


def score(events, truth, tol=45):
    """Greedy one-to-one match by smallest frame distance, same direction."""
    produced = [{"frame": f, "direction": d, "how": h} for f, d, h in events]
    pairs = []
    for pi, p in enumerate(produced):
        for ti, t in enumerate(truth):
            if p["direction"] == t["direction"] and \
                    abs(p["frame"] - t["frame"]) <= tol:
                pairs.append((abs(p["frame"] - t["frame"]), pi, ti))
    pairs.sort()
    used_p, used_t, matched = set(), set(), []
    for _, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matched.append((pi, ti))
    tp = len(matched)
    fp = [p for i, p in enumerate(produced) if i not in used_p]
    fn = [t for i, t in enumerate(truth) if i not in used_t]
    prec = tp / (tp + len(fp)) if tp + len(fp) else 0.0
    rec = tp / (tp + len(fn)) if tp + len(fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    by_how = collections.Counter(produced[pi]["how"] for pi, _ in matched)
    fp_how = collections.Counter(p["how"] for p in fp)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "tp_by_how": dict(by_how), "fp_by_how": dict(fp_how)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tolerance", type=int, default=45)
    ap.add_argument("--no-coast", action="store_true")
    ap.add_argument("--dead-zone", type=int, default=None)
    args = ap.parse_args()

    events = run(args.dump, enable_coast=not args.no_coast,
                 dead_zone=args.dead_zone)
    truth = load_labels(args.labels)
    r = score(events, truth, args.tolerance)
    print("=" * 60)
    print(f"{os.path.basename(args.dump)}  vs  "
          f"{os.path.basename(args.labels)}")
    print(f"produced {len(events)} events, truth {len(truth)}")
    for t in r["fn"]:
        print(f"    MISSED  {t['direction']:9s} @ frame {t['frame']}")
    for x in r["fp"]:
        print(f"    PHANTOM {x['direction']:9s} @ frame {x['frame']} "
              f"({x['how']})")
    print(f"precision={r['precision']:.3f} recall={r['recall']:.3f} "
          f"F1={r['f1']:.3f}  (tp={r['tp']} fp={len(r['fp'])} "
          f"fn={len(r['fn'])})")
    print(f"tp by method: {r['tp_by_how']}   fp by method: {r['fp_by_how']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
