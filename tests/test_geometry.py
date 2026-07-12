"""CountingLine geometry: horizontal equivalence, angles, flip, serialization."""
from cpcs_geometry import CountingLine


def test_horizontal_reduces_to_y_minus_m():
    L = CountingLine(0, 150, 400, 150, dead_zone=22)
    assert L.zone_of(200, 100) == "above"
    assert L.zone_of(200, 200) == "below"
    assert L.zone_of(200, 150) is None
    assert abs(L.signed_distance(200, 180) - 30) < 1e-6
    assert L.normal() == (0.0, 1.0)


def test_vertical_line_separates_left_right():
    V = CountingLine(200, 0, 200, 300, dead_zone=10)
    za, zb = V.zone_of(150, 100), V.zone_of(250, 100)
    assert za != zb and None not in (za, zb)


def test_diagonal_line_separates_sides():
    D = CountingLine(0, 0, 300, 300, dead_zone=5)
    assert D.zone_of(250, 50) != D.zone_of(50, 250)


def test_flip_swaps_sides():
    F = CountingLine(0, 150, 400, 150, dead_zone=22, flip=True)
    assert F.zone_of(200, 100) == "below"
    assert F.zone_of(200, 200) == "above"


def test_dict_round_trip():
    L = CountingLine(10, 20, 300, 40, dead_zone=18, flip=True)
    L2 = CountingLine.from_dict(L.as_dict())
    for x, y in [(50, 10), (150, 35), (280, 60)]:
        assert L.zone_of(x, y) == L2.zone_of(x, y)


def test_horizontal_mid_factory():
    L = CountingLine.horizontal_mid(400, 300, dead_zone=22)
    assert L.zone_of(200, 100) == "above"
    assert L.zone_of(200, 250) == "below"
