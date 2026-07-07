"""
Per-bus configuration — the "one file per vehicle" that makes this scale.

Everything that changes between buses lives here, so deploying on a new vehicle
is editing a YAML file, never editing code. This is the concrete answer to the
spec's "scalable to a 500-bus fleet" requirement: 500 buses = 500 config files,
one shared codebase.

Load order in the apps: if a config path is given (or config.yaml exists next
to the script) it is used; otherwise sensible defaults apply and the counting
line falls back to horizontal-mid, preserving old behaviour.
"""

import os

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


DEFAULT = {
    "bus": {"route": "Demo Route", "bus_id": "BUS-001"},
    "camera": {
        "index": 0,
        # line = [x1, y1, x2, y2] in pixels; null => horizontal mid at runtime
        "line": None,
        "dead_zone": 22,
        "flip": False,
    },
    "detection": {"model": "yolov8n.pt", "imgsz": 640, "conf": 0.10},
    "runtime": {"fps": 30, "coast": True},
    "economics": {"fare": 15, "capacity": 45},
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    """Return a config dict. Missing keys are filled from DEFAULT."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "config.yaml")
        path = cand if os.path.exists(cand) else None
    if path and os.path.exists(path):
        if not _HAVE_YAML:
            raise RuntimeError("config.yaml found but PyYAML is not installed "
                               "(pip install pyyaml)")
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULT, loaded)
    return dict(DEFAULT)


def save_config(cfg, path="config.yaml"):
    if not _HAVE_YAML:
        raise RuntimeError("PyYAML is required to save config (pip install pyyaml)")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path
