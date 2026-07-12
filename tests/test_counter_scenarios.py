"""DoorCounter v6 scenario suite - the permanent regression gate.

House rule: no counting bug is ever fixed without a scenario here that
failed before the fix. These cases encode every failure mode found during
development (see counter/ history and the technical review).
"""
import importlib.util
import os

from cpcs_geometry import CountingLine

_POC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "cpcs_poc.py")
spec = importlib.util.spec_from_file_location("cpcs_poc", _POC)
poc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poc)
DoorCounter = poc.DoorCounter


def feed_y(dc, frames):
    fired = []
    for fr in frames:
        boxes = [[190, y - 20, 210, y + 20] for (_, y) in fr]
        ids = [t for (t, _) in fr]
        fired += dc.update(boxes, ids)
    fired += dc.flush()
    return [(d, h) for (d, h, _) in fired]


def feed_x(dc, frames):
    fired = []
    for fr in frames:
        boxes = [[x - 10, 140, x + 10, 160] for (_, x) in fr]
        ids = [t for (t, _) in fr]
        fired += dc.update(boxes, ids)
    fired += dc.flush()
    return [(d, h) for (d, h, _) in fired]


def hline():
    return CountingLine.horizontal_mid(400, 300, dead_zone=22)


def test_clean_boarding_counts_once_live():
    seq = [[(1, 100)], [(1, 120)], [(1, 140)], [(1, 160)], [(1, 180)], [(1, 200)]]
    assert feed_y(DoorCounter(hline()), seq) == [("boarding", "live")]


def test_detection_void_before_line_recovered_by_coast():
    seq = [[(1, 98)], [(1, 108)], [(1, 118)], [(1, 128)], [(1, 138)]]
    seq += [[] for _ in range(15)]
    assert feed_y(DoorCounter(hline()), seq) == [("boarding", "coast")]


def test_slow_drifter_vanishing_is_not_invented():
    seq = [[(1, 131)], [(1, 132)], [(1, 133)], [(1, 134)], [(1, 135)]]
    seq += [[] for _ in range(15)]
    assert feed_y(DoorCounter(hline()), seq) == []


def test_approach_and_retreat_never_counts():
    seq = [[(1, 100)], [(1, 120)], [(1, 135)], [(1, 140)],
           [(1, 135)], [(1, 120)], [(1, 100)]]
    assert feed_y(DoorCounter(hline()), seq) == []


def test_coast_then_reappearance_does_not_double_count():
    seq = [[(1, 98)], [(1, 108)], [(1, 118)], [(1, 128)], [(1, 138)]]
    seq += [[] for _ in range(7)]
    seq += [[(2, 218)], [(2, 228)], [(2, 238)]]
    got = feed_y(DoorCounter(hline()), seq)
    assert len(got) == 1 and got[0][0] == "boarding"


def test_fragmented_alighting_counts_exactly_once():
    seq = [[(1, 260)], [(1, 250)], [(1, 240)], [(1, 230)], [(1, 220)]]
    seq += [[] for _ in range(3)]
    seq += [[(2, 182)], [(2, 172)], [(2, 162)], [(2, 120)], [(2, 110)]]
    got = feed_y(DoorCounter(hline()), seq)
    assert len(got) == 1 and got[0][0] == "alighting"


def test_no_coast_flag_disables_dead_reckoning():
    seq = [[(1, 98)], [(1, 108)], [(1, 118)], [(1, 128)], [(1, 138)]]
    seq += [[] for _ in range(15)]
    assert feed_y(DoorCounter(hline(), enable_coast=False), seq) == []


def test_vertical_line_counts_horizontal_crossing():
    V = CountingLine(200, 0, 200, 300, dead_zone=15)
    seq = [[(1, 120)], [(1, 150)], [(1, 180)], [(1, 210)], [(1, 240)], [(1, 270)]]
    got = feed_x(DoorCounter(V), seq)
    assert len(got) == 1


def test_legacy_int_constructor_matches_explicit_line():
    seq = [[(1, 100)], [(1, 120)], [(1, 140)], [(1, 160)], [(1, 180)], [(1, 200)]]
    a = feed_y(DoorCounter(150), seq)
    b = feed_y(DoorCounter(hline()), seq)
    assert a == b == [("boarding", "live")]


def test_two_people_crossing_opposite_directions():
    seq = [[(1, 100), (2, 200)], [(1, 130), (2, 170)], [(1, 160), (2, 140)],
           [(1, 190), (2, 110)], [(1, 220), (2, 90)]]
    got = feed_y(DoorCounter(hline()), seq)
    dirs = sorted(d for d, _ in got)
    assert dirs == ["alighting", "boarding"]
