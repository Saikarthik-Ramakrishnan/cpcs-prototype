# Measured accuracy — test_1.mp4

Every number here is `eval/validate.py` output on the real pipeline
(`cpcs_poc.py`), not a replay and not an estimate. Reproduce with the commands
at the bottom.

## The clip

`test_1.mp4` — 1283 frames, 402×300, 30 fps (≈43 s), overhead view.
Ten people cross the mid-line: **7 boardings, 3 alightings**. No staff, no
re-crossings, no approach-and-retreat. Labels: `eval/labels/test_1.json`.

Ground truth was read off a frame-by-frame review of the clip; each event
frame is where the person's centroid crosses y=150. Scoring tolerance is the
protocol default, 45 frames.

## Results

| Configuration | P | R | F1 | tp | fp | fn |
|---|---|---|---|---|---|---|
| `yolov8s` @ 960 (accuracy mode) | 1.000 | 1.000 | 1.000 | 10 | 0 | 0 |
| `yolov8s` @ 960, `--no-coast`   | 1.000 | 1.000 | 1.000 | 10 | 0 | 0 |
| `yolov8n` @ 640 (fast mode)     | 1.000 | 0.900 | 0.947 |  9 | 0 | 1 |

Method mix in accuracy mode: 7 `live`, 3 `coast`. With `--no-coast` all ten
are `live` — the recovery layers change *which* layer fires, not the totals,
on this clip.

The fast-mode miss is the person at frame 750. `yolov8n` @ 640 never detects
them below y=46, so no counting logic can recover it; it is a detector limit,
not a counter limit.

## What this does and does not establish

**Does not**: this is one clip with ten events. Ten events cannot support a
claim like "95% recall" — the confidence interval on 10/10 is far too wide,
and a single overhead clip of a clean pavement is easier than a crowded bus
doorway. Treat this as a regression baseline, not a validation result. The
week-4 10-trip validation is what produces a quotable fleet number.

**Does**: it pins the pipeline end to end, and it caught a real counting bug
(below).

## Bug this measurement found

The frame-750 boarding was being lost inside `DoorCounter`, not by the
detector. `_try_stitch` adopted a *spent* track — one that had already fired
its event — as the donor for a newly-born track. The newcomer inherited
`counted=True` and was muted for the rest of its life, so a genuine crossing
by a different person was never recorded.

Two independent gates were wrong, and either one alone lets it through:

- an already-counted track could be a stitch donor. Adopting one can never
  recover a count but can destroy one.
- the raw-distance gate `STITCH_BASE_D + STITCH_PER_FR * gap` grew 8 px per
  frame with no ceiling. At an 18-frame gap that is a 174 px allowance on a
  402 px-wide frame, so two people at opposite ends were stitchable.

Fixed by skipping counted donors and capping the gate at `STITCH_MAX_D = 90`.
Effect on tp: `s960` 9→10, `s1280` 8→9. Regression tests in
`tests/test_counter_scenarios.py` pin each gate separately, plus one test that
asserts the scenario still *fails* with both gates disabled so the suite
cannot go quietly vacuous.

## Sensitivity

`python eval/tune.py --dumps "eval/dumps/test_1_*.csv" --labels
eval/labels/test_1.json` sweeps every counter tunable. After the fix the score
is flat across the full range of all eleven knobs, except `stale` < 30 and
`stitch_max_gap` < 25, which lose one event. The result is therefore not a
knife-edge parameter fit — no tunable was tuned to obtain it.

## Reproduce

```bash
python cpcs_poc.py --source test_1.mp4 --model yolov8s.pt --imgsz 960 \
                   --headless --db /tmp/eval.db --events-csv /tmp/events.csv
python eval/validate.py --events /tmp/events.csv --labels eval/labels/test_1.json
```

Counter-only (no inference, milliseconds, no torch needed):

```bash
python eval/replay.py --dump eval/dumps/test_1_s960.csv --labels eval/labels/test_1.json
```
