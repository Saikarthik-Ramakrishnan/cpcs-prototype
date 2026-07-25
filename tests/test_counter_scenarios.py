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
CounterParams = poc.CounterParams


def feed_pts(dc, frames):
    """frames: list of [(tid, x, y), ...] — lets a scenario place tracks apart
    in x, which the y-only feeders cannot express."""
    fired = []
    for fr in frames:
        boxes = [[x - 10, y - 20, x + 10, y + 20] for (_, x, y) in fr]
        ids = [t for (t, _, _) in fr]
        fired += dc.update(boxes, ids)
    fired += dc.flush()
    return [(d, h) for (d, h, _) in fired]


def spent_donor_then_new_person():
    """The frame-750 failure, reduced.

    A person crosses and is counted, then stands still long enough for their
    velocity to decay to ~0, then their detection drops. Eighteen frames later
    an unrelated person appears at the far end of the frame. Because the raw
    distance gate grows 8 px/frame with no ceiling, 18 frames buys a 174 px
    allowance, which is most of a 402 px frame -- so the newcomer is stitched
    onto the spent track, inherits counted=True, and their genuine crossing is
    never recorded. Correct behaviour is two boardings.
    """
    seq = [[(1, 200, 100)], [(1, 200, 120)], [(1, 200, 140)],
           [(1, 200, 160)], [(1, 200, 180)], [(1, 200, 200)]]
    seq += [[(1, 200, 200)] for _ in range(8)]      # stand still, vy -> ~0
    seq += [[] for _ in range(17)]                  # detection drops
    seq += [[(2, 200, 30)], [(2, 200, 60)], [(2, 200, 90)], [(2, 200, 120)],
            [(2, 200, 150)], [(2, 200, 180)], [(2, 200, 210)]]
    return seq


def test_spent_track_is_not_adopted_and_second_person_is_counted():
    got = feed_pts(DoorCounter(hline()), spent_donor_then_new_person())
    assert [d for d, _ in got] == ["boarding", "boarding"], got


def test_counted_guard_alone_prevents_the_theft():
    """Pin the counted-donor rule on its own: disable the distance ceiling so
    it cannot be what rescues the count."""
    p = CounterParams().replace(stitch_max_d=1e9)
    got = feed_pts(DoorCounter(hline(), params=p),
                   spent_donor_then_new_person())
    assert [d for d, _ in got] == ["boarding", "boarding"], got


def test_distance_ceiling_alone_prevents_the_theft():
    """Pin the distance ceiling on its own: allow counted tracks as donors so
    the counted rule cannot be what rescues the count."""
    p = CounterParams().replace(stitch_counted=True)
    got = feed_pts(DoorCounter(hline(), params=p),
                   spent_donor_then_new_person())
    assert [d for d, _ in got] == ["boarding", "boarding"], got


def test_scenario_actually_failed_before_the_fix():
    """Guard against the regression test quietly becoming vacuous: with both
    rules disabled the original bug must still reproduce (one count lost)."""
    p = CounterParams().replace(stitch_counted=True, stitch_max_d=1e9)
    got = feed_pts(DoorCounter(hline(), params=p),
                   spent_donor_then_new_person())
    assert [d for d, _ in got] == ["boarding"], got


def test_legitimate_fragment_of_uncounted_person_still_stitches():
    """The fix must not break stitching, which exists to rejoin an uncounted
    person's fragmented track."""
    seq = [[(1, 200, 260)], [(1, 200, 250)], [(1, 200, 240)],
           [(1, 200, 230)], [(1, 200, 220)]]
    seq += [[] for _ in range(3)]
    seq += [[(2, 200, 182)], [(2, 200, 172)], [(2, 200, 162)],
            [(2, 200, 120)], [(2, 200, 110)]]
    got = feed_pts(DoorCounter(hline()), seq)
    assert len(got) == 1 and got[0][0] == "alighting", got


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
