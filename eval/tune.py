"""Sweep DoorCounter parameters over recorded dumps and report sensitivity.

Two things this answers, both of which matter more than "what is the best F1":

  1. Is the shipped setting on a knife edge? A parameter value that scores
     perfectly but where the neighbouring value collapses is not a result,
     it is a coincidence. `--sweep` prints the score across the whole range
     so the shape is visible.
  2. Does a change hold across detectors? Scores are aggregated over every
     dump given, so a setting that only helps one model/imgsz combination is
     visibly worse than one that helps all of them.

Usage:
    python eval/tune.py --dumps eval/dumps/test_1_*.csv \
                        --labels eval/labels/test_1.json --sweep
    python eval/tune.py --dumps eval/dumps/test_1_s960.csv \
                        --labels eval/labels/test_1.json --sweep dead_zone
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay import CounterParams, load_labels, run, score  # noqa: E402

# name -> values to try. dead_zone is not a CounterParams field (it lives on
# the CountingLine), so it is handled separately below.
GRID = {
    "dead_zone":      [10, 14, 18, 22, 26, 30, 34],
    "ema_alpha":      [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "min_age":        [1, 2, 3, 4],
    "stale":          [20, 30, 40, 60, 80],
    "snap_gap":       [2, 3, 4, 6, 8],
    "coast_max_gap":  [0, 4, 8, 12, 16, 20, 25],
    "coast_min_vd":   [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "coast_min_age":  [2, 3, 4, 6, 8],
    "stitch_max_gap": [10, 15, 25, 35, 50],
    "stitch_pred_d":  [30, 45, 60, 80, 100],
    "stitch_max_d":   [45, 60, 90, 120, 200, 10000],
}


def evaluate(dumps, truth, tol, dead_zone=None, **kw):
    """Aggregate tp/fp/fn over every dump; returns (f1, prec, rec, detail)."""
    tp = fp = fn = 0
    detail = []
    params = CounterParams().replace(**kw) if kw else CounterParams()
    for d in dumps:
        ev = run(d, params=params, dead_zone=dead_zone)
        r = score(ev, truth, tol)
        tp += r["tp"]
        fp += len(r["fp"])
        fn += len(r["fn"])
        detail.append((os.path.basename(d), r["tp"], len(r["fp"]),
                       len(r["fn"])))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tolerance", type=int, default=45)
    ap.add_argument("--sweep", nargs="*", default=None,
                    help="params to sweep (default: all)")
    args = ap.parse_args()

    dumps = []
    for pat in args.dumps:
        dumps += sorted(glob.glob(pat)) or [pat]
    dumps = [d for d in dumps if d.endswith(".csv")]
    truth = load_labels(args.labels)

    base = evaluate(dumps, truth, args.tolerance)
    print("=" * 66)
    print(f"dumps: {', '.join(os.path.basename(d) for d in dumps)}")
    print(f"BASELINE (shipped defaults)  F1={base[0]:.4f} "
          f"P={base[1]:.3f} R={base[2]:.3f}")
    for name, tp, fp, fn in base[3]:
        print(f"    {name:26s} tp={tp:3d} fp={fp:3d} fn={fn:3d}")
    print("=" * 66)

    names = args.sweep if args.sweep else list(GRID)
    for name in names:
        if name not in GRID:
            print(f"(no grid for {name})")
            continue
        print(f"\n{name}:")
        for v in GRID[name]:
            kw = {"dead_zone": v} if name == "dead_zone" else {name: v}
            f1, p, r, _ = evaluate(dumps, truth, args.tolerance, **kw)
            mark = "  <- default" if _is_default(name, v) else ""
            flag = "" if f1 >= base[0] - 1e-9 else "  (worse)"
            print(f"   {name}={str(v):>7}  F1={f1:.4f} P={p:.3f} "
                  f"R={r:.3f}{flag}{mark}")


def _is_default(name, v):
    import replay
    if name == "dead_zone":
        return v == replay.poc.DEAD_ZONE
    return getattr(CounterParams(), name, object()) == v


if __name__ == "__main__":
    main()
