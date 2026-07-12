"""Test fixtures: stub heavy deps so the counting logic tests run anywhere."""
import sys
import types
import os

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_ul = types.ModuleType("ultralytics")
_ul.YOLO = object
sys.modules.setdefault("ultralytics", _ul)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
