# MuJoCo Table-Tennis Robot Replay

A **MuJoCo** physics simulation in which a robot replays table-tennis strokes. Captured or
generated stroke motion drives a simulated arm and paddle, so shots can be **replayed,
teleoperated, and analysed** in a controlled physics environment. This is the extended
research work that builds on my MSc thesis
[IMU Table-Tennis Coaching System](https://github.com/GitDip008/imu-tabletennis-coaching-system).

## Demo

The robot executing table-tennis shots in the MuJoCo simulation:

https://github.com/user-attachments/assets/fdf379f3-2364-4bb1-8143-8c09ea86fbcf







## What it does

- **Replay** recorded / generated table-tennis strokes on a simulated robot arm + paddle in
  MuJoCo physics.
- **Teleoperate** the arm live from a motion stream (including streaming poses through to the
  Unity visualization).
- **Generate & analyse** stroke data — synthesize training data from simulation, split it by
  stroke class, plot kinematics, and train baseline classifiers.

The goal is to move from just *recognizing* strokes (the coaching-system repo) to *physically
reproducing* them on a robot in simulation, as a step toward robot-assisted training and
richer, physics-grounded stroke data.

## What it uses

- **MuJoCo** for rigid-body physics simulation of the robot, paddle, and scene.
- **Python** (NumPy, PyTorch) for control, data generation, and the shared classification
  pipeline.
- The **coaching pipeline** (model, feature extraction, inference, kinematics, filtering),
  reused here under `shared/` so this repository runs standalone.
- Optional **Unity** streaming bridge to mirror the simulated motion on the 3D avatar.

## Repository structure

```
mujoco_sim/
  models/      MuJoCo model files (robot, paddle, scene)
  scripts/
    mujoco_replay_tt.py        replay a stroke on the robot in simulation
    mujoco_live_teleop_tt.py   live teleoperation of the arm
    mujoco_live_viewer.py      interactive MuJoCo viewer
    generate_tt_data.py        synthesize stroke data from simulation
    split_by_class.py          organize generated data by stroke class
    plot_kinematics.py         plot joint / paddle kinematics
    train_baselines.py         train baseline classifiers
    stream_to_unity.py         forward simulated poses to Unity
live_app_mujoco.py   real-time app wired to the MuJoCo simulation
shared/              shared helpers reused from the coaching pipeline
                     (model, inference, feature extraction, kinematics, filtering, ...)
mujoco_trial_working_11June.md   working notes / setup log
demo/                demo video
```

## Setup

```bash
pip install mujoco numpy torch     # plus the coaching pipeline's requirements
```

## How to run

```bash
# Replay a table-tennis stroke on the robot in simulation
python mujoco_sim/scripts/mujoco_replay_tt.py

# Open the interactive viewer
python mujoco_sim/scripts/mujoco_live_viewer.py

# Live teleoperation of the arm
python mujoco_sim/scripts/mujoco_live_teleop_tt.py
```

## Not included

- `mujoco_sim/output/` — large rendered/exported artifacts (videos, rollouts) are not
  distributed.
- Datasets are large and not distributed — bring your own local copy.

## SiriusCeption controller (`.pyd` / `.so`) — not included

If you use the live sensor / Unity path, it relies on the proprietary
`SiriusCeption_unity_controller` binary, which is **not redistributed here**. To obtain it,
**please contact the author.**

## Relationship to the coaching system

This repository is the **extension** of the MSc thesis coaching system. The classification
and coaching pipeline lives in the
[coaching-system repo](https://github.com/GitDip008/imu-tabletennis-coaching-system); here
the focus is the **MuJoCo robot simulation** and replaying strokes on it. Shared pipeline
code is duplicated under `shared/` so this repo runs standalone.

## License

MIT — see [LICENSE](LICENSE).
