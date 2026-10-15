"""Sensor models: BNO086-like IMUs (game rotation vector) and a MediaPipe-like phone camera."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .kin import Ry, UP, G
from .truth import pose_at, bump

IMU_FS = 200
CAM_FS = 30

# Pod design: sensor x points distally, sensor z points out of the skin (lateral)
R_NOM = R.from_matrix(np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).astype(float))


def default_cfg(**kw):
    cfg = dict(sta_scale=1.0, drift_rate=2.0, drift_rw=0.03, quat_noise=0.15, loss=0.02,
               clock=4e-3, cam_tau=None, cam_xy=(10, 18), cam_z=(30, 50), cam_outlier=0.03,
               cam_far_drop=0.25, cam_fov=2.5, cam_scale=0.05, mount_spin=30, mount_tilt=6, mount_circ=15)
    cfg.update(kw)
    return cfg


def sta_channels(subj, leg, truth, rng, scale):
    """Soft-tissue artefact as sensor-vs-bone rotation vectors (deg) in segment frame, dense time."""
    t = truth["t"]; L = truth["legs"][leg]; p = subj["legs"][leg]; s = p["sta"]
    soft = subj["soft"] * scale
    fr = L["flex"] - p["stand_flex"]
    pct, w = L["pct"], L["w"]
    mus = bump(pct, 0.2, 0.1) * w
    ham = bump(pct, 0.85, 0.08) * w
    hs = L["hs"]
    idx = np.searchsorted(hs, t, "right") - 1
    last = np.where(idx >= 0, hs[np.clip(idx, 0, None)] if len(hs) else -1e9, -1e9)
    d = t - last
    env = np.where(d >= 0, np.exp(-d / 0.05), 0.0)
    ax_t = rng.normal(0, 1, 3) * [1, 1, 0.3]; ax_t /= np.linalg.norm(ax_t)
    ax_s = rng.normal(0, 1, 3) * [1, 1, 0.3]; ax_s /= np.linalg.norm(ax_s)
    wob_t = scale * s["wob_t"] * env * np.sin(2 * np.pi * s["f_t"] * d)
    wob_s = scale * s["wob_s"] * env * np.sin(2 * np.pi * s["f_s"] * d)
    thigh = soft * np.column_stack([s["tx"] * fr + 0.6 * mus, s["ty"] * fr + s["tmus"] * mus + 0.8 * ham,
                                    s["tz"] * fr]) + wob_t[:, None] * ax_t
    shank = soft * 0.5 * np.column_stack([s["sx"] * fr, s["sy"] * fr + 0.5 * mus, s["sz"] * fr]) \
        + wob_s[:, None] * ax_s
    return dict(thigh=thigh, shank=shank)


def _burst_loss(rng, n, rate):
    lost = np.zeros(n, bool)
    nb = int(n * rate / 3)
    for st in rng.integers(0, n, nb):
        lost[st:st + rng.integers(1, 6)] = True
    return lost


def sample_imus(subj, leg, truth, rng, cfg):
    sta = sta_channels(subj, leg, truth, rng, cfg["sta_scale"])
    tn = np.arange(0, truth["total"] - 0.02, 1 / IMU_FS)
    out = {}
    for seg in ("thigh", "shank"):
        eps = rng.uniform(-cfg["clock"], cfg["clock"])
        tt = np.clip(tn + eps, 0, truth["total"] - 0.002)
        P = pose_at(truth, subj, leg, tt)
        Rseg = P["Rf"] if seg == "thigh" else P["Rt"]
        S = R.from_rotvec(np.radians(np.column_stack(
            [np.interp(tt, truth["t"], sta[seg][:, i]) for i in range(3)])))
        Pm = (R.from_euler("z", rng.uniform(-cfg["mount_spin"], cfg["mount_spin"]), degrees=True)
              * R.from_euler("y", rng.normal(0, cfg["mount_circ"]), degrees=True)
              * R.from_rotvec(np.radians(rng.normal(0, cfg["mount_tilt"], 3))))
        Rm = Pm * R_NOM
        Qc = Rseg * S * Rm
        # BNO086 game-rotation-vector: arbitrary initial heading + slow heading drift
        n = len(tn)
        rate = rng.uniform(-cfg["drift_rate"], cfg["drift_rate"]) / 60.0
        rw = np.cumsum(rng.normal(0, cfg["drift_rw"] * np.sqrt(1 / IMU_FS), n))
        drift = np.radians(rate * tn + rw) + rng.uniform(-np.pi, np.pi)
        noise = R.from_rotvec(np.radians(rng.normal(0, cfg["quat_noise"], (n, 3))))
        Q = Ry(drift) * Qc * noise
        # gyro
        rel = Qc[:-1].inv() * Qc[1:]
        om = rel.as_rotvec() * IMU_FS
        om = np.vstack([om, om[-1]])
        gyro = om + np.radians(rng.normal(0, 0.3, 3)) + np.radians(rng.normal(0, 0.5, (n, 3)))
        # accelerometer
        if seg == "thigh":
            pos = P["hipc"] + P["Rf"].apply(np.column_stack([np.zeros(n), -0.5 * subj["L1"] * np.ones(n),
                                                            0.07 * np.ones(n)]))
        else:
            pos = P["tib_o"] + P["Rt"].apply(np.column_stack([np.zeros(n), -0.3 * subj["L2"] * np.ones(n),
                                                             0.05 * np.ones(n)]))
        acc_w = np.gradient(np.gradient(pos, tt, axis=0), tt, axis=0)
        acc = Qc.inv().apply(acc_w + G * UP) + rng.normal(0, 0.03, (n, 3))
        lost = _burst_loss(rng, n, cfg["loss"])
        quat = Q.as_quat(); quat[lost] = np.nan
        gyro[lost] = np.nan; acc[lost] = np.nan
        out[seg] = dict(t=tn, quat=quat, gyro=gyro, acc=acc, _Rm=Rm, _drift=drift, _eps=eps)
    out["_sta"] = sta
    return out


def sample_camera(subj, leg, truth, rng, cfg, tau):
    """Side-view phone camera running a MediaPipe-like pose model: hip/knee/ankle 3-D landmarks.
    Frame stamped t shows the scene at t - tau (unknown latency)."""
    ts = np.arange(0, truth["total"] - 0.05, 1 / CAM_FS)
    tc = np.clip(ts - tau, 0, None)
    P = pose_at(truth, subj, leg, tc)
    n = len(ts)
    L = truth["legs"][leg]
    near = np.interp(tc, truth["t"], L["near"].astype(float)) > 0.5
    flex = P["flex"]
    b_hip = rng.normal(0, [0.012, 0.015, 0.02]); b_knee = rng.normal(0, [0.01, 0.01, 0.02])
    b_ank = rng.normal(0, [0.01, 0.01, 0.02])
    hip = P["hipc"] + P["Rf"].apply(np.tile(b_hip, (n, 1)))
    knee_off = np.tile(b_knee, (n, 1)); knee_off[:, 0] += 0.00015 * flex  # landmark slides with flexion
    knee = P["knee_f"] + P["Rf"].apply(knee_off)
    ank = P["ankle"] + P["Rt"].apply(np.tile(b_ank, (n, 1)))
    lm = np.stack([hip, knee, ank], axis=1)
    centre = np.interp(tc, truth["t"], truth["body"][:, 0])
    sc = 1 + rng.normal(0, cfg["cam_scale"])
    bodyc = np.column_stack([np.interp(tc, truth["t"], truth["body"][:, i]) for i in range(3)])
    lm = bodyc[:, None, :] + sc * (lm - bodyc[:, None, :])
    sxy = np.where(near, cfg["cam_xy"][0], cfg["cam_xy"][1]) / 1000
    sz = np.where(near, cfg["cam_z"][0], cfg["cam_z"][1]) / 1000
    noise = rng.standard_normal((n, 3, 3))
    noise[:, :, 0:2] *= sxy[:, None, None]; noise[:, :, 2] *= sz[:, None]
    out_m = rng.random((n, 3)) < cfg["cam_outlier"]
    noise += out_m[:, :, None] * rng.normal(0, 0.08, (n, 3, 3))
    lm = lm + noise
    t_walk0 = [a for (nm, a, b) in truth["bounds"] if nm == "walk"][0]
    vis = (np.abs(centre) < cfg["cam_fov"]) | (tc < t_walk0)  # calibration done in front of the phone
    drop = np.zeros(n, bool)
    far_idx = np.where(~near)[0]
    if len(far_idx):
        d = _burst_loss(rng, n, 0.0)
        for st in rng.choice(far_idx, int(len(far_idx) * cfg["cam_far_drop"] / 8) + 1):
            d[st:st + rng.integers(4, 12)] = True
        drop |= d & ~near
    drop |= rng.random(n) < 0.03
    vis &= ~drop
    lm[~vis] = np.nan
    return dict(t=ts, lm=lm, _tau=tau)
