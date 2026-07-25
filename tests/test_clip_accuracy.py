"""Accuracy regression gate on the annotated clip.

The scenario suite pins counter behaviour on synthetic tracks. This pins the
number that actually gets quoted: the score on real recorded detections from
test_1.mp4, replayed through the counter with no inference (so it runs in
milliseconds and needs neither torch nor the video file).

If a counter change drops the clip score, this fails and names the frame.
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from replay import load_labels, run, score  # noqa: E402

LABELS = os.path.join(_ROOT, "eval", "labels", "test_1.json")

# dump -> (min tp, max false positives). test_1_s960 is the configuration the
# demo runs (yolov8s @ 960) and is the only one that is currently perfect; the
# weaker configurations are pinned at what they actually achieve, so a
# regression in any of them is caught too.
EXPECTED = {
    "test_1_s960.csv":  (10, 0),
    "test_1_s1280.csv": (9, 0),
    "test_1_n640.csv":  (9, 0),
    "test_1_n960.csv":  (8, 0),
}


def _dump(name):
    return os.path.join(_ROOT, "eval", "dumps", name)


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_clip_accuracy_does_not_regress(name, expected):
    path = _dump(name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present (regenerate with eval/dump_tracks.py)")
    min_tp, max_fp = expected
    r = score(run(path), load_labels(LABELS))
    detail = ("; ".join(f"missed {t['direction']}@{t['frame']}"
                        for t in r["fn"]) or "no misses")
    assert r["tp"] >= min_tp, f"{name}: tp={r['tp']} < {min_tp} ({detail})"
    assert len(r["fp"]) <= max_fp, (
        f"{name}: {len(r['fp'])} false positives: "
        + "; ".join(f"{x['direction']}@{x['frame']} ({x['how']})"
                    for x in r["fp"]))


def test_demo_configuration_is_exact():
    """The configuration the demo records with must count every person and
    invent none."""
    path = _dump("test_1_s960.csv")
    if not os.path.exists(path):
        pytest.skip("dump not present")
    r = score(run(path), load_labels(LABELS))
    assert (r["precision"], r["recall"]) == (1.0, 1.0)


def test_labels_are_internally_consistent():
    with open(LABELS) as f:
        d = json.load(f)
    frames = [e["frame"] for e in d["events"]]
    assert frames == sorted(frames), "labels must be in frame order"
    assert all(e["direction"] in ("boarding", "alighting")
               for e in d["events"])
    assert sum(e["direction"] == "boarding" for e in d["events"]) == 7
    assert sum(e["direction"] == "alighting" for e in d["events"]) == 3
