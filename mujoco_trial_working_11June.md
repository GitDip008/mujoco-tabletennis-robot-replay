# MuJoCo Capture-and-Replay Table Tennis System — Build Report
**Date:** 11 June 2026
**Files:** `tt_coaching_pipeline/mujoco_sim/scripts/mujoco_replay_tt.py`, `tt_coaching_pipeline/mujoco_sim/models/humanoid_tt.xml`

---

## 1. What Was Built

A complete **capture-and-replay table tennis demo** in MuJoCo, extending the live-teleop
system (`mujoco_live_teleop_tt.py`) into a two-phase coaching demonstrator:

1. **LIVE_RECORD** — the player swings with the 3 IMU sensors (7=hand, 8=forearm,
   9=upper arm). The stroke classifier detects and labels each real stroke, and the
   corresponding joint-angle trajectory is automatically captured into a per-stroke
   swing library.
2. **AUTO_PLAY** — the avatar replays a captured swing against a served ball, with the
   serve velocity *solved* so the ball arrives exactly at the swing's contact point at
   the right time. Contact is physics-validated before the rally is run live.

By end of day all three stroke types (FH topspin, BH drive, FH smash) **connect with the
ball reliably**, the live outcome matches the validation prediction exactly, and the
returns are genuine forward strikes.

---

## 2. Scene Rebuild — `humanoid_tt.xml`

The scene file was recreated from scratch (previous version deleted):

- DeepMind `humanoid.xml` base (27-DOF humanoid, kinematically driven).
- **Paddle** welded to `hand_right`: capsule handle + box blade
  (0.085 × 0.005 × 0.075 m), `solref="0.005 0.30"`.
- **ITTF-regulation table** at x = 1.7 m: 2.74 × 1.525 m top at 0.76 m height, net
  (0.1524 m above the surface), centerline, 4 legs (collision-disabled).
- **Ball**: 40 mm / 2.7 g sphere.
- **Bounce physics**: positive-form `solref="0.005 0.45"` on ball and table top.
  The first number is the contact time constant (s), the second a **damping ratio**
  (0 = perfectly elastic, 1 = no bounce). 0.45 ≈ a realistic TT bounce of about half
  the drop height. (Earlier negative-form `solref` values like `-3000 -150` were
  massively over-damped for a 2.7 g ball — critical damping is only ~5 — which is why
  the ball used to roll dead instead of bouncing; `0.15` was then too elastic and shot
  the ball straight up.)
- Ball friction reduced to `0.4` so landing bounces don't convert into skid/roll.
- Contact excludes between table and avatar body parts to prevent intersection glitches.

---

## 3. The Replay System — `mujoco_replay_tt.py`

### 3.1 Stroke detection — same gate as `live_app_789.py`

The first version captured swings on any confident classifier window, which fired on
arbitrary arm motion. Fixed by replicating the exact pipeline that made the GUI's
"Shot Predictions" panel accurate:

- **Dedicated `ClassifierThread`** polls the UDP server at 3× the 100 Hz sensor rate
  and feeds the sliding-window extractor **only when the accel+gyro signature changes**
  — one row per *new* sensor packet, never a duplicated hold. (Duplicate rows create
  step-jump artifacts; this was the original "everything is a smash" bug.)
- Full gate chain: energy gate (idle windows forced to No Stroke) → MLP prediction
  (`model_synthetic.pt`) → `OnlineStrokeCounter` (fires once per motion-energy local
  maximum ≥ 40, refractory of 10 windows, label ≠ 0).
- A swing is captured **only** when the counter fires. Arm-waving below the energy
  threshold never triggers a capture. The thread is paused in AUTO_PLAY mode.
- Detections are queued thread-safely; the main loop drains the queue and snapshots
  the joint buffer.

### 3.2 Swing capture and shaping

- A ring buffer continuously records 1.5 s of IK joint angles
  (shoulder1/shoulder2/elbow at 60 fps).
- **Delayed snapshot** (`CAPTURE_DELAY = 0.35 s`): detection fires ~at the energy peak
  (the hit), so the snapshot waits a beat to include the follow-through.
- **Trim to active segment**: keep the contiguous region around the paddle-speed peak
  where speed ≥ 8% of max, padded 0.4 s per side (typical result: 90 → ~50 frames).
  Without trimming, swings detected late (smash) replayed ~1.3 s of slow drift before
  the actual hit. The first trim attempt (15%, 0.2 s pad) was too aggressive — it cut
  the wind-up and broke contact-frame search; the gentler values restored it.

### 3.3 Contact-point selection (direction-aware)

`contact_candidates()` ranks swing frames as potential ball-contact moments:

- Paddle inside the hittable window (x 0.30–1.00, y ±0.75, z 0.85–1.85 m, via FK on a
  scratch `MjData`).
- Paddle speed ≥ 30% of the swing max **and moving dominantly forward**
  (+x component ≥ 50% of speed) — ranked by forward velocity.
- Fallback tier accepts any forward motion if no dominantly-forward frame exists.

This was the key fix for return quality: speed-only ranking chose frames where the
hand whipped sideways/downward, producing dribbled or backward returns.

### 3.4 Serve solver (shooting method)

`solve_serve()` aims the ball at a chosen contact point using the **real physics**
(scratch-`MjData` rollouts with paddle contacts disabled, table bounce included):

- Per-axis secant updates on serve-velocity y/z until the simulated ball passes within
  2 cm of the target as it crosses the contact x-plane; the x-velocity is scaled up if
  the ball never reaches the plane.
- Converges in 2–10 iterations in practice; returns the velocity and time-of-flight.

### 3.5 Coupled validation + joint search

`coupled_trial()` replicates the live AUTO_PLAY loop *exactly* (swing replay + ball,
paddle contacts ON, identical substep interpolation) and reports: hit frame, outgoing
ball velocity, closest ball–blade approach, and **where the return first lands on the
table**.

`trigger_replay()` performs a **joint search over (contact candidate × timing offset)**:

- Up to 4 candidates; for each, the serve is solved and validated across ±8 frames of
  swing-timing offset.
- Scoring tiers: return lands on the opponent court > any contact; within a tier,
  prefer the straightest forward return. Early-exit on the first on-table result.
- The chosen schedule is then executed live, all in **render-frame counts** (sim time),
  so paddle and ball stay in sync regardless of wall-clock jitter.

### 3.6 Live execution details

- **Substep sweep interpolation**: the arm pose is interpolated across the 12 physics
  substeps per render frame, with qvel set to the true sweep rate — without this the
  paddle teleports ~10 cm/frame near the swing peak and the ball tunnels through it.
- **Post-swing hold + blend**: after the replay ends the avatar holds the
  follow-through 0.4 s, then eases back to ready stance over 1 s (smoothstep). An
  instant snap-back used to sweep the paddle through the ball's return path and hit it
  a second time.
- **Real outcome reporting**: actual MuJoCo contact pairs decide HIT/MISS (the earlier
  blade-center distance metric falsely reported corner hits as misses), plus the return
  landing position and IN/OUT verdict.

### 3.7 Controls (Ctrl-chorded)

MuJoCo's viewer has built-in plain-key shortcuts (`1` hides geom group 1 — the whole
skeleton; `S` toggles shadows) that the user callback can't suppress. All commands are
now **Ctrl-chorded**: the callback receives the Ctrl key-down (GLFW 341/345) and accepts
an action key within a 3 s window; while Ctrl is held the viewer's own bindings don't
fire.

| Keys | Action |
|---|---|
| `Ctrl+L` / `Ctrl+P` | LIVE_RECORD / AUTO_PLAY mode |
| `Ctrl+1` / `Ctrl+2` / `Ctrl+3` | select FH topspin / BH drive / FH smash |
| `SPACE` | serve + replay selected stroke (AUTO_PLAY) |
| `Ctrl+C` | manual capture (LIVE_RECORD) |
| `Ctrl+R` | reset ball |
| `Ctrl+S` / `Ctrl+O` | save / load swing library |

Library saves are timestamped (`swing_library_YYYYMMDD_HHMMSS.pkl` in
`mujoco_sim/output/`); `Ctrl+O` loads the most recent one.

---

## 4. Issues Found and Fixed Today (chronological)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Ball rolled dead on landing, no bounce | Negative-form `solref` damping (−150, then −60) was 10–30× critical damping for a 2.7 g ball | Positive-form `solref="0.005 …"` (damping *ratio*) on ball + table |
| 2 | Ball rocketed straight up after bounce | Damping ratio 0.15 too elastic | Ratio 0.45 ≈ realistic TT bounce |
| 3 | Swings captured during any arm motion | Classifier fed once per render frame with duplicated packets | `ClassifierThread` with signature dedup + energy-peak `OnlineStrokeCounter` (live_app_789 pipeline) |
| 4 | Smash replay swung long before ball arrived | 1.5 s buffer = ~1.4 s dead lead-in; peak at frame 87/90 | Delayed snapshot (+0.35 s) + speed-based trim |
| 5 | First trim too aggressive (90→25, peak at frame 1–2; smash lost its hittable frames) | 15% threshold, 0.2 s pad cut the wind-up | 8% threshold, 0.4 s pad (90→~50) |
| 6 | Bat hit the ball a second time after the swing | Instant snap back to ready pose swept through the return path | 0.4 s hold + 1 s smoothstep blend |
| 7 | "MISS 9.3 cm" reported on visible hits | Verdict used blade-*center* distance (corner is ~11 cm out) | Real contact-pair detection |
| 8 | Returns dribbled off the paddle or shot backward | Contact frames chosen by speed only, ignoring direction | Direction-aware candidates (dominantly +x), ranked by forward velocity |
| 9 | Poor fallback contact frame when first solver attempt failed | First-converged candidate locked in | Joint candidate × offset search with on-table-first scoring |
| 10 | `1` hid the skeleton, `S` toggled shadows while also saving | Viewer built-in plain-key shortcuts collide with app keys | Ctrl-chorded commands |
| 11 | `Ctrl+O` couldn't find previous sessions after saves became timestamped | Load used the current run's timestamp | Load newest `swing_library_*.pkl` |

---

## 5. End-of-Day Verified State

From the final test session (real sensors, live):

- All three stroke types captured automatically with correct labels
  (confidences 0.54–1.00, energies 200–1100).
- Trims behave as designed (90 → 29–56 frames, peak mid-sequence).
- **Every replay produced a real paddle–ball HIT**, for all three stroke types.
- Validation and live execution agree exactly (predicted landing x/y matched live to
  the centimeter in earlier sessions; deterministic frame-counted scheduling).
- Returns are genuine forward strikes (e.g. `v=[4.35, −0.25, 2.93]`,
  `[6.68, −3.27, 1.02]`, `[8.13, 0.31, 6.2]` m/s) — currently flying long past the far
  edge rather than landing in the opponent court.

## 6. Known Limitations / Next Steps

1. **Returns land long** — struck hard with upward vz; the scoring's on-table tier is
   rarely achievable with the current recorded strokes. Options: prefer offsets with
   lower return vz, or treat it as coaching signal ("this stroke sails long").
2. **LLM coach integration** (recommended next): per-rally outcomes (hit, return
   velocity, landing IN/OUT) → session summary → existing `coaching.py` / LM Studio
   pipeline. Turns the replay into grounded, physics-based coaching feedback.
3. **On-screen scoreboard** for demo legibility (serves / hits / returns IN).
4. Paddle face angle is fixed by the recorded hand orientation — strokes that
   geometrically can't clear the net should be re-recorded with a flatter forward brush.

## 7. How to Run

```
E:\thesis_work\imu_to_unity\.py11_venv\Scripts\python.exe ^
  E:\thesis_work\TT_thesis\tt_coaching_pipeline\mujoco_sim\scripts\mujoco_replay_tt.py
```

1. Power the 3 sensors, stand in T-pose, wait for the 3 s calibration.
2. LIVE_RECORD starts automatically — play each stroke type a few times; watch for
   `[stroke] …` / `[capture] …` lines.
3. `Ctrl+S` to save the library.
4. `Ctrl+P` → `Ctrl+1/2/3` → `SPACE` to serve and watch the avatar return the ball.
5. Console prints the full decision trail: candidate frames, solver convergence,
   validation verdict, serve schedule, and the live HIT/landing outcome.
