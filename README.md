# MuJoCo Table-Tennis Robot Replay

Extended work building on the [IMU Table-Tennis Coaching System](https://github.com/USERNAME/imu-tabletennis-coaching-system):
a **MuJoCo** physics simulation where a robot replays table-tennis strokes. Captured or
generated stroke motion drives a simulated arm/paddle so shots can be replayed, teleoperated,
and analysed in a controlled physics environment.

## What it does
- **Replay** recorded/generated table-tennis strokes on a simulated robot in MuJoCo.
- **Teleoperate** the arm live from a motion stream.
- **Generate & analyse** stroke data (kinematics plots, per-class splits) for training.

## Repository layout
```
mujoco_sim/
  models/      MuJoCo model files (robot, paddle, scene)
  scripts/     replay, live teleop, viewer, data generation, kinematics plotting,
               baseline training, stream-to-Unity bridge
live_app_mujoco.py   real-time app wired to the MuJoCo simulation
shared/              shared helpers reused from the coaching pipeline
                     (model, inference, feature extraction, kinematics, filtering)
mujoco_trial_working_11June.md   working notes / setup log
```

## Not included (and why)
- **`mujoco_sim/output/`** — large rendered/exported artifacts (videos, rollouts) are not
  distributed.
- **Datasets** — large; bring your own local copy.
- **Proprietary sensor binary** (`.pyd`/`.so`) — not redistributed (see the coaching-system
  repo).

## Quick start
```bash
pip install mujoco numpy torch    # plus the coaching pipeline's requirements
python mujoco_sim/scripts/mujoco_replay_tt.py     # replay a stroke in simulation
python mujoco_sim/scripts/mujoco_live_viewer.py   # interactive viewer
```

## Relationship to the coaching system
This repo is the **extension** of my MSc thesis coaching system. The classification /
coaching pipeline lives in the coaching-system repo; here the focus is the MuJoCo robot
simulation and replaying strokes on it. Shared pipeline code is duplicated under `shared/`
so this repo runs standalone.

## License
MIT — see [LICENSE](LICENSE).
