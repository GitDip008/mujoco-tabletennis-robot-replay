"""Headless reproduction of mujoco_replay_tt.py's AUTO_PLAY loop.
Synthetic swing -> candidates -> serve solve -> live-loop replay.
Prints ball/blade/target positions at the contact iteration to find the
systematic offset behind the deterministic MISS results.
"""
import pathlib
import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(HERE.parent / "models" / "humanoid_tt.xml"))
data    = mujoco.MjData(model)
scratch = mujoco.MjData(model)
fk_data = mujoco.MjData(model)

def jaddr(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

s1_q, s1_v = jaddr("shoulder1_right")
s2_q, s2_v = jaddr("shoulder2_right")
el_q, el_v = jaddr("elbow_right")
ball_q, ball_v = jaddr("ball_root")
gid_blade  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")
gid_handle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_handle")
gid_ball   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")

data.qpos[2] = 1.282
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)
stand_qpos = data.qpos.copy()

READY      = np.array([0.0, 0.0, -0.5])
TARGET_FPS = 60
dt_frame   = 1.0 / TARGET_FPS
dt_phys    = model.opt.timestep
phys_steps = max(1, int(round(dt_frame / dt_phys)))
SERVE_LEAD = 0.3
TOL        = 0.02

def paddle_pos_at(qarm):
    fk_data.qpos[:] = stand_qpos
    fk_data.qpos[s1_q], fk_data.qpos[s2_q], fk_data.qpos[el_q] = qarm
    mujoco.mj_forward(model, fk_data)
    return fk_data.geom_xpos[gid_blade].copy()

def simulate_ball(S, v, t_max=2.5):
    sv = (model.geom_contype[gid_blade],  model.geom_conaffinity[gid_blade],
          model.geom_contype[gid_handle], model.geom_conaffinity[gid_handle])
    model.geom_contype[gid_blade]  = 0; model.geom_conaffinity[gid_blade]  = 0
    model.geom_contype[gid_handle] = 0; model.geom_conaffinity[gid_handle] = 0
    try:
        d = scratch
        d.qpos[:] = stand_qpos
        d.qpos[s1_q], d.qpos[s2_q], d.qpos[el_q] = READY
        d.qpos[ball_q:ball_q+3]   = S
        d.qpos[ball_q+3:ball_q+7] = [1, 0, 0, 0]
        d.qvel[:] = 0
        d.qvel[ball_v:ball_v+3] = v
        traj = []
        for k in range(int(t_max / dt_phys)):
            bp = d.qpos[ball_q:ball_q+7].copy()
            bv = d.qvel[ball_v:ball_v+6].copy()
            d.qpos[:] = stand_qpos
            d.qpos[s1_q], d.qpos[s2_q], d.qpos[el_q] = READY
            d.qpos[ball_q:ball_q+7] = bp
            d.qvel[:] = 0
            d.qvel[ball_v:ball_v+6] = bv
            mujoco.mj_step(model, d)
            traj.append(((k + 1) * dt_phys, d.qpos[ball_q:ball_q+3].copy()))
            if d.qpos[ball_q + 2] < 0.05:
                break
        return traj
    finally:
        (model.geom_contype[gid_blade],  model.geom_conaffinity[gid_blade],
         model.geom_contype[gid_handle], model.geom_conaffinity[gid_handle]) = sv

def crossing(traj, p_star):
    for (t1, p1), (t2, p2) in zip(traj[:-1], traj[1:]):
        if p1[0] >= p_star[0] >= p2[0]:
            a = (p1[0] - p_star[0]) / max(p1[0] - p2[0], 1e-9)
            return t1 + a * (t2 - t1), p1 + a * (p2 - p1)
    return None, None

def solve_serve(S, v0, p_star, iters=25):
    v = v0.astype(float).copy()
    prev_y = prev_z = None
    for it in range(iters):
        t_star, p_sim = crossing(simulate_ball(S, v), p_star)
        if t_star is None:
            v[0] *= 1.2
            continue
        ey = p_sim[1] - p_star[1]
        ez = p_sim[2] - p_star[2]
        if abs(ey) < TOL and abs(ez) < TOL:
            return v, t_star
        sy = ((ey - prev_y[1]) / (v[1] - prev_y[0])
              if prev_y and abs(v[1]-prev_y[0]) > 1e-6 and abs(ey-prev_y[1]) > 1e-9
              else t_star)
        sz = ((ez - prev_z[1]) / (v[2] - prev_z[0])
              if prev_z and abs(v[2]-prev_z[0]) > 1e-6 and abs(ez-prev_z[1]) > 1e-9
              else t_star)
        prev_y = (v[1], ey)
        prev_z = (v[2], ez)
        v[1] -= float(np.clip(ey / sy, -1.5, 1.5))
        v[2] -= float(np.clip(ez / sz, -1.5, 1.5))
    return None, None

# ── Synthetic swing: READY → forehand sweep, bell-speed profile ────────
n_frames = 90
t = np.linspace(0, 1, n_frames)
ease = 0.5 - 0.5 * np.cos(np.pi * np.clip((t - 0.2) / 0.6, 0, 1))   # 0→1 sweep
q_seq = np.zeros((n_frames, 3))
q_seq[:, 0] = 0.0 + ease * 0.5      # shoulder1
q_seq[:, 1] = 0.0 - ease * 0.7      # shoulder2
q_seq[:, 2] = -0.5 + ease * 0.4     # elbow

# Candidates
P = np.array([paddle_pos_at(q) for q in q_seq])
speed = np.linalg.norm(np.diff(P, axis=0), axis=1)
smax = float(speed.max())
HIT_X = (0.35, 0.95); HIT_Y = (-0.70, 0.70); HIT_Z = (0.88, 1.70)
cands = [i for i in range(1, n_frames - 1)
         if speed[i-1] >= 0.40 * smax
         and HIT_X[0] <= P[i][0] <= HIT_X[1]
         and HIT_Y[0] <= P[i][1] <= HIT_Y[1]
         and HIT_Z[0] <= P[i][2] <= HIT_Z[1]]
cands.sort(key=lambda i: -speed[i-1])
print(f"candidates: {cands[:8]}")
assert cands, "synthetic swing never enters hit window — adjust sweep"

S  = np.array([3.00, -0.20, 1.10])
v0 = np.array([-3.5,  0.00, 1.8])
peak_idx, p_star, vel, t_star = None, None, None, None
for idx in cands[:6]:
    v_try, t_try = solve_serve(S, v0, P[idx])
    if v_try is not None:
        peak_idx, p_star, vel, t_star = idx, P[idx], v_try, t_try
        break
assert vel is not None, "solver failed on synthetic swing"
print(f"contact frame {peak_idx} at {np.round(p_star,3)}, vel {np.round(vel,2)}, t*={t_star:.3f}s")

# ── Coupled validation over swing-timing offsets (mirrors the script) ──
trial_data = mujoco.MjData(model)

def coupled_trial(serve_rel, replay_rel, n_iters):
    d = trial_data
    mujoco.mj_resetData(model, d)
    local_stand = stand_qpos.copy()
    local_stand[ball_q:ball_q+3]   = [0.0, 0.0, -5.0]
    local_stand[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[:] = local_stand
    d.qvel[:] = 0.0
    qw = q_seq[0].copy(); qp = qw.copy()
    served = False; hit_frame = None; out_vel = None; mind = 1e9
    for k in range(n_iters):
        if not served and k >= serve_rel:
            d.qpos[ball_q:ball_q+3]   = S
            d.qpos[ball_q+3:ball_q+7] = [1, 0, 0, 0]
            d.qvel[ball_v:ball_v+3]   = vel
            d.qvel[ball_v+3:ball_v+6] = 0
            local_stand[ball_q:ball_q+3] = S
            served = True
        fi = k - replay_rel
        qw = q_seq[0] if fi < 0 else (q_seq[fi] if fi < n_frames else q_seq[-1])
        rate = (qw - qp) / dt_frame
        for kk in range(phys_steps):
            alpha = (kk + 1) / phys_steps
            qs = qp + (qw - qp) * alpha
            bp = d.qpos[ball_q:ball_q+7].copy(); bv = d.qvel[ball_v:ball_v+6].copy()
            d.qpos[:] = local_stand
            d.qpos[s1_q], d.qpos[s2_q], d.qpos[el_q] = qs
            d.qpos[ball_q:ball_q+7] = bp
            d.qvel[:] = 0
            d.qvel[s1_v], d.qvel[s2_v], d.qvel[el_v] = rate
            d.qvel[ball_v:ball_v+6] = bv
            mujoco.mj_step(model, d)
            if served:
                dd = float(np.linalg.norm(d.geom_xpos[gid_ball] - d.geom_xpos[gid_blade]))
                mind = min(mind, dd)
            if hit_frame is None:
                for ci in range(d.ncon):
                    g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                    if gid_ball in (g1, g2) and (gid_blade in (g1, g2) or gid_handle in (g1, g2)):
                        hit_frame = k
                        break
            elif out_vel is None and k >= hit_frame + 3:
                out_vel = d.qvel[ball_v:ball_v+3].copy()
        qp = qw.copy()
    if hit_frame is not None and out_vel is None:
        out_vel = d.qvel[ball_v:ball_v+3].copy()
    return hit_frame, out_vel, mind

margin    = 8
n_flight  = int(round(t_star / dt_frame))
serve_rel = int(round(SERVE_LEAD / dt_frame)) + margin
replay_rel0 = serve_rel - 1 + n_flight - peak_idx
if replay_rel0 < margin:
    extra = margin - replay_rel0
    serve_rel += extra; replay_rel0 += extra
n_iters = max(serve_rel + n_flight, replay_rel0 + margin + n_frames) + 30

best = None
for delta in range(-margin, margin + 1):
    hf, ov, md = coupled_trial(serve_rel, replay_rel0 + delta, n_iters)
    tag = f"hit@{hf} out_vx={ov[0]:+.2f}" if hf is not None and ov is not None else f"miss ({md*100:.1f}cm)"
    print(f"  delta {delta:+d}: {tag}")
    if hf is not None and ov is not None and (best is None or ov[0] > best[0]):
        best = (float(ov[0]), delta, hf, ov, md)
assert best is not None, "validation found no hit"
_, delta, hf, ov, md = best
print(f"CHOSEN offset {delta:+d}: hit at frame {hf}, outgoing v={np.round(ov,2)}")

serve_k   = serve_rel
replay_k0 = replay_rel0 + delta
contact_k = hf
print(f"serve_k={serve_k} replay_k0={replay_k0} contact_k={contact_k} n_flight={n_flight}")

# park ball
data.qpos[ball_q:ball_q+3]   = [0, 0, -5]
data.qpos[ball_q+3:ball_q+7] = [1, 0, 0, 0]
data.qvel[ball_v:ball_v+6]   = 0
stand_local = stand_qpos.copy()
stand_local[ball_q:ball_q+3] = [0, 0, -5]

q_warm     = READY.copy()
q_prev_arm = q_warm.copy()
min_dist, min_k = 1e9, -1
served = False

for k in range(0, contact_k + 60):
    # serve
    if not served and k >= serve_k:
        data.qpos[ball_q:ball_q+3]   = S
        data.qpos[ball_q+3:ball_q+7] = [1, 0, 0, 0]
        data.qvel[ball_v:ball_v+3]   = vel
        data.qvel[ball_v+3:ball_v+6] = 0
        stand_local[ball_q:ball_q+3] = S
        served = True

    # replay frame
    fi = k - replay_k0
    if fi < 0:
        q_warm = q_seq[0]
    elif fi < n_frames:
        q_warm = q_seq[fi]
    else:
        q_warm = q_seq[-1]

    arm_rate = (q_warm - q_prev_arm) / dt_frame
    for kk in range(phys_steps):
        alpha = (kk + 1) / phys_steps
        q_sub = q_prev_arm + (q_warm - q_prev_arm) * alpha
        bp = data.qpos[ball_q:ball_q+7].copy()
        bv = data.qvel[ball_v:ball_v+6].copy()
        data.qpos[:] = stand_local
        data.qpos[s1_q], data.qpos[s2_q], data.qpos[el_q] = q_sub
        data.qpos[ball_q:ball_q+7] = bp
        data.qvel[:] = 0
        data.qvel[s1_v], data.qvel[s2_v], data.qvel[el_v] = arm_rate
        data.qvel[ball_v:ball_v+6] = bv
        mujoco.mj_step(model, data)
        d = float(np.linalg.norm(data.geom_xpos[gid_ball] - data.geom_xpos[gid_blade]))
        if d < min_dist:
            min_dist, min_k = d, k
    q_prev_arm = q_warm.copy()

    if k == contact_k:
        bpos = data.qpos[ball_q:ball_q+3].copy()
        blad = data.geom_xpos[gid_blade].copy()
        print(f"\nAT CONTACT ITERATION k={k} (end of frame):")
        print(f"  target p*   : {np.round(p_star, 3)}")
        print(f"  ball center : {np.round(bpos, 3)}   (off by {np.round(bpos - p_star, 3)})")
        print(f"  blade center: {np.round(blad, 3)}   (off by {np.round(blad - p_star, 3)})")

print(f"\nmin ball-blade distance {min_dist*100:.1f} cm at iteration {min_k} "
      f"(contact_k={contact_k})")
