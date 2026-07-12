"""
CPCS evaluation harness: score produced events against ground-truth labels.

Usage:
    python eval/validate.py --events events.csv --labels eval/labels/clip1.json
    python eval/validate.py --events events.csv --labels eval/labels/ --tolerance 45

events.csv is the file written by cpcs_poc.py (columns include
frame, direction, how). Labels follow eval/protocol.md.

Output: per-file and overall precision/recall/F1, plus a per-method
breakdown (how many of the matched events came from live vs coast vs
fallback, and the false-positive rate of each method).
"""

import argparse
import csv
import glob
import json
import os
import sys


def load_events(path):
    evs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            evs.append({"frame": int(row["frame"]),
                        "direction": row["direction"]
                        if "direction" in row else row["event"],
                        "how": row.get("how", "live")})
    return evs


def load_labels(path):
    with open(path) as f:
        d = json.load(f)
    return [e for e in d["events"] if not e.get("staff", False)]


def match(produced, truth, tol):
    """Greedy one-to-one matching by smallest frame distance, same direction."""
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
    tp_by_how, fp_by_how = {}, {}
    for pi, _ in matched:
        h = produced[pi]["how"]
        tp_by_how[h] = tp_by_how.get(h, 0) + 1
    for p in fp:
        h = p["how"]
        fp_by_how[h] = fp_by_how.get(h, 0) + 1
    return tp, fp, fn, tp_by_how, fp_by_how


def prf(tp, n_fp, n_fn):
    prec = tp / (tp + n_fp) if tp + n_fp else 0.0
    rec = tp / (tp + n_fn) if tp + n_fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True,
                    help="events.csv produced by cpcs_poc.py")
    ap.add_argument("--labels", required=True,
                    help="a label .json or a directory of them")
    ap.add_argument("--tolerance", type=int, default=45,
                    help="max frame distance for a match (default 45)")
    args = ap.parse_args()

    label_files = sorted(glob.glob(os.path.join(args.labels, "*.json"))) \
        if os.path.isdir(args.labels) else [args.labels]
    if not label_files:
        sys.exit("no label files found")

    produced = load_events(args.events)
    T_tp = T_fp = T_fn = 0
    T_tp_how, T_fp_how = {}, {}

    print("=" * 60)
    for lf in label_files:
        truth = load_labels(lf)
        tp, fp, fn, tp_how, fp_how = match(produced, truth, args.tolerance)
        p, r, f1 = prf(tp, len(fp), len(fn))
        print(f"{os.path.basename(lf):32s} "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f} "
              f"(tp={tp} fp={len(fp)} fn={len(fn)})")
        for t in fn:
            print(f"    MISSED  {t['direction']:9s} @ frame {t['frame']}")
        for x in fp:
            print(f"    PHANTOM {x['direction']:9s} @ frame {x['frame']} "
                  f"({x['how']})")
        T_tp += tp
        T_fp += len(fp)
        T_fn += len(fn)
        for k, v in tp_how.items():
            T_tp_how[k] = T_tp_how.get(k, 0) + v
        for k, v in fp_how.items():
            T_fp_how[k] = T_fp_how.get(k, 0) + v

    p, r, f1 = prf(T_tp, T_fp, T_fn)
    print("=" * 60)
    print(f"OVERALL  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
    print(f"         tp={T_tp}  fp={T_fp}  fn={T_fn}")
    print("per-method (true positives | false positives):")
    for h in sorted(set(list(T_tp_how) + list(T_fp_how))):
        print(f"  {h:14s} {T_tp_how.get(h, 0):4d} | {T_fp_how.get(h, 0):4d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
