# CPCS ground-truth annotation protocol

## What counts as an event
One event = one person fully crossing the doorway threshold in one direction.
- direction `boarding` = outside to inside; `alighting` = inside to outside
- record the frame number at which the person's HEAD crosses the physical threshold
- a person who steps back before fully crossing is NOT an event
- a person who crosses, returns, and crosses again = multiple events (each recorded)
- staff (driver/conductor) crossings ARE recorded, tagged `staff: true`
  (lets us measure staff-suppression separately)

## Label file format (one JSON per clip)
```json
{
  "clip": "cam0_trip3.mp4",
  "fps": 30,
  "annotator": "ram",
  "events": [
    {"frame": 412, "direction": "boarding", "staff": false},
    {"frame": 655, "direction": "alighting", "staff": false}
  ]
}
```

## Matching rule for scoring
A produced event matches a ground-truth event if same direction and
|frame_produced - frame_truth| <= tolerance (default 45 frames = 1.5 s at 30fps).
Matching is one-to-one, greedy by smallest frame distance.
- matched pair -> true positive
- unmatched produced event -> false positive
- unmatched ground-truth event -> false negative

## Reporting rules
- report precision, recall, F1 overall AND per method tag (live/coast/...)
- report per clip; aggregate with per-clip bootstrap CIs, never pool events
- README accuracy numbers may only come from eval/validate.py output
