"""Config loading: defaults, deep-merge partial overrides."""
import os
import tempfile

from cpcs_config import load_config


def test_defaults_load_without_file():
    c = load_config("/nonexistent/path.yaml")
    assert c["economics"]["fare"] == 15
    assert c["camera"]["dead_zone"] == 22
    assert c["camera"]["line"] is None


def test_partial_override_deep_merges():
    p = tempfile.mktemp(suffix=".yaml")
    with open(p, "w") as f:
        f.write("bus:\n  route: '47A'\n"
                "camera:\n  line: [10, 300, 1900, 340]\n  flip: true\n")
    try:
        c = load_config(p)
        assert c["bus"]["route"] == "47A"
        assert c["camera"]["line"] == [10, 300, 1900, 340]
        assert c["camera"]["flip"] is True
        assert c["detection"]["model"] == "yolov8n.pt"
        assert c["economics"]["capacity"] == 45
    finally:
        os.remove(p)
