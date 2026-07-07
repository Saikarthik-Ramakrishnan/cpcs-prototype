"""
CPCS calibration tool — draw the counting line for a specific bus door.

Load a frame from a clip or a live camera, click two points to place the
counting line where the doorway threshold actually is, adjust the dead-zone
band, flip the in/out direction if needed, and save it all to config.yaml.
cpcs_poc.py then reads that file and counts against the calibrated line.

Why this matters (say this in the demo): the counting line is a virtual
tripwire. A fixed line in the middle of the frame is a guess; a calibrated
line matches the real door geometry of each bus. Per-bus config files are
also how one codebase scales to a whole fleet.

Run:
    # calibrate from the first clear frame of a recording
    python calibrate.py --source recordings/<session>/cam0.mp4

    # or calibrate live from a camera
    python calibrate.py --source 0

Controls:
    left-click twice   set the two endpoints of the counting line
    [  /  ]            decrease / increase dead-zone width
    f                  flip which side counts as "boarding" (inside the bus)
    space              grab the next frame (video source)
    r                  reset the line
    s                  save to config.yaml and quit
    q                  quit without saving
"""

import argparse

import cv2

from cpcs_config import load_config, save_config
from cpcs_geometry import CountingLine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--route", default=None)
    ap.add_argument("--bus", default=None)
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    if not ret:
        print("could not read from source:", args.source)
        return
    h, w = frame.shape[:2]

    cfg = load_config(args.config if _exists(args.config) else None)
    if args.route:
        cfg["bus"]["route"] = args.route
    if args.bus:
        cfg["bus"]["bus_id"] = args.bus

    state = {"pts": [], "dead": cfg["camera"].get("dead_zone", 22),
             "flip": cfg["camera"].get("flip", False), "frame": frame}

    # seed from existing calibrated line if present
    if cfg["camera"].get("line"):
        x1, y1, x2, y2 = cfg["camera"]["line"]
        state["pts"] = [(x1, y1), (x2, y2)]

    win = "CPCS calibrate - click 2 pts | [ ] dead | f flip | space next | s save | q quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state["pts"]) >= 2:
                state["pts"] = []
            state["pts"].append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    print("calibrating on a %dx%d frame. click two points for the line." % (w, h))

    while True:
        disp = state["frame"].copy()

        # draw the line + dead-zone band + direction arrow
        if len(state["pts"]) == 2:
            (x1, y1), (x2, y2) = state["pts"]
            line = CountingLine(x1, y1, x2, y2,
                                dead_zone=state["dead"], flip=state["flip"])
            cv2.line(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            nx, ny = line.normal()
            # band edges
            ox, oy = int(nx * state["dead"]), int(ny * state["dead"])
            cv2.line(disp, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (0, 160, 160), 1)
            cv2.line(disp, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), (0, 160, 160), 1)
            # "IN" arrow points to the boarding side (positive normal after flip)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = int(mx + nx * 40), int(my + ny * 40)
            cv2.arrowedLine(disp, (mx, my), (ax, ay), (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(disp, "IN", (ax + 4, ay + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif len(state["pts"]) == 1:
            cv2.circle(disp, state["pts"][0], 5, (0, 255, 255), -1)

        hud1 = f"dead_zone={state['dead']}  flip={state['flip']}  pts={len(state['pts'])}/2"
        hud2 = f"route={cfg['bus']['route']}  bus={cfg['bus']['bus_id']}"
        cv2.putText(disp, hud1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp, hud2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            print("quit without saving.")
            break
        elif key == ord("["):
            state["dead"] = max(4, state["dead"] - 2)
        elif key == ord("]"):
            state["dead"] = min(120, state["dead"] + 2)
        elif key == ord("f"):
            state["flip"] = not state["flip"]
        elif key == ord("r"):
            state["pts"] = []
        elif key == ord(" "):
            ret, nf = cap.read()
            if ret:
                state["frame"] = nf
        elif key == ord("s"):
            if len(state["pts"]) != 2:
                print("set two points before saving.")
                continue
            (x1, y1), (x2, y2) = state["pts"]
            cfg["camera"]["line"] = [int(x1), int(y1), int(x2), int(y2)]
            cfg["camera"]["dead_zone"] = int(state["dead"])
            cfg["camera"]["flip"] = bool(state["flip"])
            path = save_config(cfg, args.config)
            print(f"saved calibrated line to {path}")
            print(f"  line = {cfg['camera']['line']}  dead_zone={state['dead']}  flip={state['flip']}")
            print(f"run:  python cpcs_poc.py --source {args.source} --config {path}")
            break

    cap.release()
    cv2.destroyAllWindows()


def _exists(p):
    import os
    return os.path.exists(p)


if __name__ == "__main__":
    main()
