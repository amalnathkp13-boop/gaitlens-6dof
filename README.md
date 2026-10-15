# GaitLens-6DOF — virtual bench

End-to-end simulation of the GaitLens-6DOF knee kinematics concept: ground-truth subjects,
sensor models, the full processing pipeline, and a validation study scored against truth.

## Layout
- `gaitlens/kin.py` — Grood-Suntay / ISB knee JCS (rotations Z-X'-Y'', translations on JCS axes)
- `gaitlens/truth.py` — virtual subjects (healthy, medial-OA varus thrust, ACL-deficient), protocol schedule, 6-DOF truth
- `gaitlens/sensors.py` — BNO086-like IMUs (game rotation vector: arbitrary heading, drift, noise, BLE loss, clock offset,
  strap misplacement, soft-tissue artefact) and a MediaPipe-like side camera (latency, depth noise, landmark sliding, occlusion)
- `gaitlens/pipeline.py` — the host pipeline: resampling, two-pass functional calibration, heading alignment,
  gait events, camera sync, drift estimation (standing updates + stride stationarity + camera), crosstalk minimisation,
  filtering, optional joint-constraint shrinkage, translation model
- `gaitlens/evaluate.py` — run a subject and score it
- `run_validation.py` → `analyze.py` → `export_demo.py` → `report.py`

## Run
```
pip install -r requirements.txt
python run_validation.py   # ~6 min on 2 cores, writes out/results.pkl
python analyze.py          # out/summary.json + figures
python export_demo.py      # out/demo.json
python report.py           # out/GaitLens_clinician_report.pdf
```

## Moving to hardware
Replace `sample_imus` / `sample_camera` with readers for the ESP32 BLE stream and the phone landmark stream,
using the same dict layout (`t`, `quat` [x,y,z,w], `gyro` rad/s, `acc` m/s², `lm` hip/knee/ankle in metres).
`Processor(imus, cam, protocol_bounds, leg).run()` is unchanged. Protocol bounds come from the app's
static / knee-bend / walk buttons.

## Results
Pre-computed outputs are in `results/`: `gaitlens_bench.html` (interactive dashboard, open in a browser),
`GaitLens_clinician_report.pdf`, `summary.json` and the figures.

## Key results (64 knees)
Flexion 1.8°, varus 1.8°, IE 2.6° RMSE. OA varus-thrust detection AUC 1.00. Translation is model-derived and
does not detect ACL deficiency (AUC 0.44). Joint-constraint STA shrinkage removes 48–73% of true varus thrust.
