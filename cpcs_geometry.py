"""
CountingLine — the virtual tripwire, positioned per bus.

Generalizes the counter from a fixed horizontal mid-frame line to a line drawn
at ANY position and ANY angle, defined by two pixel points. This is what lets
the same code work on any bus doorway: the line is calibrated to the door, not
assumed to be the middle of the frame.

Convention:
  signed_distance(x, y) is the perpendicular distance from the line, signed.
  zone "above" = negative side, "below" = positive side, None = inside the band.
  A crossing above -> below is a boarding; below -> above is an alighting.
  If the camera's geometry puts "inside the bus" on the wrong side, set flip=True
  (the calibration tool exposes this as a single key press).

For a horizontal line (0, m) - (W, m) the math reduces exactly to the old
behaviour: signed_distance = y - m, so existing results are unchanged.
"""

import math


class CountingLine:
    def __init__(self, x1, y1, x2, y2, dead_zone=22, flip=False):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.dead_zone = dead_zone
        self.flip = flip
        dx, dy = (x2 - x1), (y2 - y1)
        length = math.hypot(dx, dy) or 1.0
        # unit normal (perpendicular to the line)
        self.nx = -dy / length
        self.ny = dx / length

    def signed_distance(self, x, y):
        d = (x - self.x1) * self.nx + (y - self.y1) * self.ny
        return -d if self.flip else d

    def zone_of(self, x, y):
        return self.zone_from_d(self.signed_distance(x, y))

    def zone_from_d(self, d):
        if d < -self.dead_zone:
            return "above"
        if d > self.dead_zone:
            return "below"
        return None

    def normal(self):
        if self.flip:
            return (-self.nx, -self.ny)
        return (self.nx, self.ny)

    @classmethod
    def horizontal_mid(cls, w, h, dead_zone=22):
        """Default line: horizontal, across the middle — old behaviour."""
        return cls(0, h // 2, w, h // 2, dead_zone=dead_zone)

    def as_dict(self):
        return {"line": [self.x1, self.y1, self.x2, self.y2],
                "dead_zone": self.dead_zone, "flip": self.flip}

    @classmethod
    def from_dict(cls, d):
        x1, y1, x2, y2 = d["line"]
        return cls(x1, y1, x2, y2,
                   dead_zone=d.get("dead_zone", 22),
                   flip=d.get("flip", False))
