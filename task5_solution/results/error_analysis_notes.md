# Qualitative error analysis notes

Model: logistic {'C': 1.0, 'class_weight': None}, threshold=0.3, median_window=3.
Ground truth = whole-second majority vote over annotators (from the aligned .npz annotation tensor). Visual is a *precomputed mel feature representation* (segment means), not a raw waveform (no .wav available).

## success — 003224.wav  (figure: error_case_1.png)
- Segment support (GT positives): 31 over 21 seconds
- Correctly detected classes: keyboard_typing, phone_ringing
- Missed classes (present in GT, not detected): none
- False detections (predicted, absent in GT): cutlery_dishes
- Timing shifts: minor / none
- Likely reasons: loud, sustained sources (e.g. running water, vacuum, microwave) give stable spectral features that linear scores separate well; false detections (cutlery_dishes) come from acoustically similar transients and the balanced class weights raising recall at the cost of precision.

## false_positive — 001714.wav  (figure: error_case_2.png)
- Segment support (GT positives): 33 over 30 seconds
- Correctly detected classes: none
- Missed classes (present in GT, not detected): door_open_close, running_water
- False detections (predicted, absent in GT): bell_ringing, cutlery_dishes, footsteps, phone_ringing, wardrobe_drawer_open_close, window_open_close
- Timing shifts: minor / none
- Likely reasons: missed classes are often short or rare (door_open_close, running_water); 1-second resolution and class imbalance hurt recall; false detections (bell_ringing, cutlery_dishes, footsteps) come from acoustically similar transients and the balanced class weights raising recall at the cost of precision.

## missed — 001503.wav  (figure: error_case_3.png)
- Segment support (GT positives): 71 over 28 seconds
- Correctly detected classes: coffee_machine
- Missed classes (present in GT, not detected): microwave, running_water
- False detections (predicted, absent in GT): vacuum_cleaner
- Timing shifts: coffee_machine (onset +1s, duration -9s)
- Likely reasons: missed classes are often short or rare (microwave, running_water); 1-second resolution and class imbalance hurt recall; false detections (vacuum_cleaner) come from acoustically similar transients and the balanced class weights raising recall at the cost of precision.
