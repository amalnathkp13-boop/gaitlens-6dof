"""Ground-truth generator: virtual subjects with full 6-DOF knee kinematics.

Each leg is simulated in its own (sagittally mirrored) world so both legs share one
right-leg convention. World: X = initial walking direction, Y = up, Z = right.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from .kin import knee_rot, trans_to_vec, Ry, smoothstep


def _ps(pts):
    x = np.array([p[0] for p in pts], float) / 100
    y = np.array([p[1] for p in pts], float)
    return CubicSpline(x, y, bc_type="periodic")


# Normative-shaped gait curves (deg) vs gait cycle fraction, heel strike = 0
KNEE = _ps([(0, 4), (6, 11), (14, 18), (25, 13), (40, 5), (50, 6), (58, 22), (66, 48),
            (72, 60), (78, 55), (88, 28), (96, 7), (100, 4)])
HIP = _ps([(0, 28), (12, 24), (30, 10), (50, -9), (56, -10), (64, 0), (75, 22), (86, 32), (100, 28)])
ADD = _ps([(0, 0.5), (8, 1.8), (18, 2.2), (35, 1.2), (55, 0), (70, -2.5), (85, -1.0), (100, 0.5)])
IEX = _ps([(0, 1), (10, 2.5), (25, 1), (45, -1), (60, 0), (72, 3), (88, 2), (100, 1)])
HADD = _ps([(0, 0), (15, 3), (35, 2), (55, -1), (75, -3), (100, 0)])
TO_FRAC = 0.60


def bump(p, c, w):
    d = (p - c + 0.5) % 1 - 0.5
    return np.exp(-0.5 * (d / w) ** 2)


def screw_home(flex, screw, gain):
    return screw * gain * (1 - np.exp(-np.maximum(flex, 0) / 15.0))


def make_subject(rng, kind="healthy", sid=0):
    s = dict(id=sid, kind=kind,
             L1=rng.normal(0.42, 0.025), L2=rng.normal(0.41, 0.025),
             cadence=rng.normal(108, 6), stride_len=rng.normal(1.30, 0.08),
             soft=rng.uniform(0.6, 1.6), legs={})
    for leg in ("R", "L"):
        s["legs"][leg] = dict(
            knee_gain=rng.normal(1, 0.07), stand_flex=rng.normal(3, 1.5),
            hip_gain=rng.normal(1, 0.06),
            add_off=rng.normal(0, 1.0), add_gain=rng.normal(1, 0.3),
            screw=rng.normal(8, 2), ie_gain=rng.normal(1, 0.3), ie_off=rng.normal(0, 1.5),
            k_ap=rng.normal(0.08, 0.03), k_ml=rng.normal(0.25, 0.1), k_pd=rng.normal(-0.02, 0.01),
            thrust=0.0, acl=0.0, acl_ie=0.0, stiff=1.0,
            sta=dict(tx=rng.normal(0.04, 0.02), ty=rng.normal(0.08, 0.04), tz=rng.normal(-0.06, 0.02),
                     tmus=rng.normal(1.5, 0.5), sx=rng.normal(0.01, 0.01), sy=rng.normal(0.02, 0.015),
                     sz=rng.normal(-0.02, 0.01), wob_t=rng.uniform(1.0, 2.0), wob_s=rng.uniform(0.7, 1.5),
                     f_t=rng.uniform(11, 17), f_s=rng.uniform(14, 22)))
    if kind == "oa":  # medial OA, varus thrust, stiff knee - affected leg R
        p = s["legs"]["R"]
        p["thrust"] = rng.uniform(3, 6); p["add_off"] += rng.uniform(2, 4); p["stiff"] = rng.uniform(0.75, 0.88)
        s["cadence"] -= 8; s["stride_len"] -= 0.15
    if kind == "acl":  # ACL deficiency - extra anterior tibial translation in stance, leg R
        p = s["legs"]["R"]
        p["acl"] = rng.uniform(3, 6); p["acl_ie"] = rng.uniform(2, 4)
    return s


def schedule(n_pass=3, walk_s=9.0, calib_s=15.0):
    segs = [("static", 5.0), ("calibR", calib_s), ("calibL", calib_s), ("stand", 2.0)]
    for i in range(n_pass):
        segs.append(("walk", walk_s))
        if i < n_pass - 1:
            segs += [("stand", 2.0), ("turn", 2.0), ("stand", 2.0)]
    segs.append(("stand", 2.0))
    return segs


def bounds_of(segs):
    out, t0 = [], 0.0
    for name, d in segs:
        out.append((name, t0, t0 + d)); t0 += d
    return out


def simulate_truth(subj, rng, segs, fs=1000):
    bnd = bounds_of(segs)
    total = bnd[-1][2]
    t = np.arange(0, total, 1 / fs)
    N = len(t); dt = 1 / fs
    T0 = 120.0 / subj["cadence"]
    v = subj["stride_len"] / T0

    psi = np.zeros(N); w = np.zeros(N); c = np.full(N, np.nan); seg_name = np.empty(N, dtype=object)
    cur, pidx = 0.0, -1
    seg_post = {}
    walk_info = []
    for k, (name, a, b) in enumerate(bnd):
        m = (t >= a) & (t < b)
        seg_name[m] = name
        if name == "walk":
            pidx += 1; cur = pidx * np.pi; psi[m] = cur
            tau = t[m] - a
            w[m] = smoothstep(tau / 0.6) * smoothstep((b - a - tau) / 0.6)
            ns = int((b - a) / T0) + 4
            Tk = T0 * (1 + 0.02 * rng.standard_normal(ns))
            B = np.concatenate([[0], np.cumsum(Tk)])
            kk = np.searchsorted(B, tau, "right") - 1
            c[m] = kk + (tau - B[kk]) / Tk[kk]
            walk_info.append(dict(a=a, b=b, B=B, Tk=Tk, pass_idx=pidx))
        elif name == "turn":
            psi[m] = cur + np.pi * smoothstep((t[m] - a) / (b - a)); cur += np.pi
        else:
            psi[m] = cur
    walking = ~np.isnan(c)

    # body progression
    dirv = np.column_stack([np.cos(psi), np.zeros(N), -np.sin(psi)])
    body = np.cumsum(v * w[:, None] * dirv * dt, axis=0)
    body[:, 0] -= 4.6
    cc = np.nan_to_num(c)
    body[:, 1] = 0.92 + 0.02 * np.cos(4 * np.pi * cc) * w

    legs = {}
    for leg, ph0 in (("R", 0.0), ("L", 0.5)):
        p = subj["legs"][leg]
        # standing posture (re-stance variability per segment; static defines 0)
        of = np.zeros((N, 3))
        for k, (name, a, b) in enumerate(bnd):
            m = (t >= a) & (t < b)
            if name != "static":
                of[m] = rng.normal(0, [1.0, 0.5, 0.8])
        sway = 0.4 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6))
        sf = p["stand_flex"] + of[:, 0] + sway
        s_flex = sf.copy()
        s_hip = 0.6 * p["stand_flex"] + of[:, 0] * 0.5
        s_add = p["add_off"] + of[:, 1]
        # calibration flexions of this leg
        calm = seg_name == f"calib{leg}"
        a_cal = [b0 for (n, b0, b1) in bnd if n == f"calib{leg}"][0]
        tau = t[calm] - a_cal
        s_flex[calm] += 80 * np.sin(np.pi * tau / 3.0) ** 2
        s_hip[calm] += 5.0
        s_add[calm] += 0.02 * (s_flex[calm] - sf[calm])
        s_ie = screw_home(s_flex, p["screw"], p["ie_gain"]) + p["ie_off"] + of[:, 2]

        cyc = cc + ph0
        pct = cyc % 1.0
        # per-stride amplitude variability
        amp = 1 + 0.03 * rng.standard_normal(200)
        a_c = np.interp(cyc, np.arange(200), amp)
        st = p["stiff"]
        w_flex = p["stand_flex"] + (KNEE(pct) - 4) * p["knee_gain"] * a_c * st
        w_add = p["add_off"] + (ADD(pct) - 0.5) * p["add_gain"] + p["thrust"] * bump(pct, 0.15, 0.06)
        w_ie = (screw_home(w_flex, p["screw"], p["ie_gain"]) + (IEX(pct) - 1) * p["ie_gain"] + p["ie_off"]
                + p["acl_ie"] * bump(pct, 0.2, 0.1))
        w_hip = HIP(pct) * p["hip_gain"] * a_c

        ww = np.where(walking, w, 0.0)
        flex = (1 - ww) * s_flex + ww * w_flex
        add = (1 - ww) * s_add + ww * w_add
        ie = (1 - ww) * s_ie + ww * w_ie
        hip = (1 - ww) * s_hip + ww * w_hip
        hadd = HADD(pct) * ww
        pyaw = 4 * np.sin(2 * np.pi * pct) * ww
        stance = bump(pct, 0.2, 0.12) * ww
        q = np.column_stack([p["k_ml"] * add,
                             p["k_ap"] * flex + p["acl"] * bump(pct, 0.2, 0.1) * ww,
                             p["k_pd"] * flex - 1.0 * stance])

        # events (heel strike = integer cycle, toe off = 0.60)
        hs, to = [], []
        for wi in walk_info:
            B, Tk = wi["B"], wi["Tk"]
            for n in range(0, len(Tk) + 1):
                for frac, lst in ((0.0, hs), (TO_FRAC, to)):
                    target = n + frac - ph0
                    if target < 0: continue
                    kk = int(np.floor(target))
                    if kk >= len(Tk): continue
                    te = wi["a"] + B[kk] + (target - kk) * Tk[kk]
                    i = int(round(te * fs))
                    if wi["a"] <= te < wi["b"] and i < N and w[i] > 0.99:
                        lst.append(te)
        right_dir = np.column_stack([np.sin(psi), np.zeros(N), np.cos(psi)])
        near = np.cos(psi) > 0 if leg == "R" else np.cos(psi) < 0
        legs[leg] = dict(flex=flex, add=add, ie=ie, q=q, hip=hip, hadd=hadd, pyaw=pyaw, pct=pct,
                         w=ww, stance=stance, hs=np.array(sorted(hs)), to=np.array(sorted(to)),
                         hipc=body + 0.09 * right_dir, near=near)
    return dict(t=t, fs=fs, psi=psi, body=body, walking=walking, seg=seg_name, bounds=bnd, legs=legs,
                total=total, T0=T0)


def pose_at(truth, subj, leg, times):
    """Interpolate truth channels at arbitrary times and build segment poses/landmarks."""
    t = truth["t"]; L = truth["legs"][leg]
    I = lambda x: np.interp(times, t, x)
    flex, add, ie = I(L["flex"]), I(L["add"]), I(L["ie"])
    q = np.column_stack([I(L["q"][:, i]) for i in range(3)])
    psi, hip, hadd, pyaw = I(truth["psi"]), I(L["hip"]), I(L["hadd"]), I(L["pyaw"])
    hipc = np.column_stack([I(L["hipc"][:, i]) for i in range(3)])
    Rf = Ry(psi + np.radians(pyaw)) * R.from_euler("ZX", np.column_stack([hip, hadd]), degrees=True)
    Rk = knee_rot(flex, add, ie)
    Rt = Rf * Rk
    L1, L2 = subj["L1"], subj["L2"]
    knee_f = hipc + Rf.apply([0, -L1, 0])
    tib_o = knee_f + Rf.apply(trans_to_vec(Rk, q))
    ankle = tib_o + Rt.apply([0, -L2, 0])
    return dict(Rf=Rf, Rt=Rt, Rk=Rk, flex=flex, add=add, ie=ie, q=q, hipc=hipc, knee_f=knee_f,
                tib_o=tib_o, ankle=ankle, psi=psi)
