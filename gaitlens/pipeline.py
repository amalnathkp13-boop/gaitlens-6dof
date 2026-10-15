"""GaitLens-6DOF processing pipeline (what would run on the host laptop/phone).

Stages
 1. Resample all IMU streams to a common 200 Hz grid (slerp over BLE packet loss)
 2. Functional calibration: gyro-only joint-axis estimation (Seel/McGrath style) + gravity at static
 3. Heading alignment of the two sensors' arbitrary game-rotation-vector frames
 4. Gait events from shank sagittal angular velocity
 5. Camera time-sync by cross-correlating knee flexion
 6. Heading-drift estimation (robust least squares over a spline):
      - standing updates (knee returns to calibrated standing posture during pauses)
      - stride stationarity (mean IE per stride constant across a trial)
      - camera segment-heading observations (optional)
 7. Soft-tissue handling: zero-phase low-pass + optional joint-constraint shrinkage (lambda)
 8. Translations: joint model (population or camera-fitted subject coefficient)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import least_squares
from .kin import knee_angles, knee_rot, trans_to_vec, Ry, yaw_of, wrap, UP
from .sensors import R_NOM, IMU_FS

FS = IMU_FS
POP = dict(k_ap=0.08, k_ml=0.25, k_pd=-0.02, k_ap_sd=0.03)


def lpf(x, fc, order=2):
    b, a = butter(order, fc / (FS / 2))
    return filtfilt(b, a, x, axis=0)


def resample(s, tg):
    v = ~np.isnan(s["quat"][:, 0])
    ts = s["t"][v]
    Q = Slerp(ts, R.from_quat(s["quat"][v]))(np.clip(tg, ts[0], ts[-1]))
    gy = np.column_stack([np.interp(tg, ts, s["gyro"][v, i]) for i in range(3)])
    ac = np.column_stack([np.interp(tg, ts, s["acc"][v, i]) for i in range(3)])
    return Q, gy, ac


def _sph(j):
    j = j / np.linalg.norm(j)
    return np.arcsin(np.clip(j[2], -1, 1)), np.arctan2(j[1], j[0])


def _vec(th, ph):
    return np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])


def joint_axes(wt, ws, j0t, j0s):
    """Gyro-only hinge-axis identification: ||w_t x j_t|| = ||w_s x j_s||  (Seel et al. 2012)."""
    x0 = np.array([*_sph(j0t), *_sph(j0s)])

    def res(x):
        jt, js = _vec(x[0], x[1]), _vec(x[2], x[3])
        return np.linalg.norm(np.cross(wt, jt), axis=1) - np.linalg.norm(np.cross(ws, js), axis=1)

    r = least_squares(res, x0, loss="soft_l1", f_scale=0.3)
    jt, js = _vec(r.x[0], r.x[1]), _vec(r.x[2], r.x[3])
    # sign from pod design: axis must point roughly out of the lateral skin
    lat = R_NOM.inv().apply([0, 0, 1])
    if jt @ lat < 0: jt = -jt
    if js @ lat < 0: js = -js
    return jt, js


def joint_axes_fixed(wt, ws, j0t, js):
    """Thigh axis only, with the shank axis already known."""
    nrm_s = np.linalg.norm(np.cross(ws, js), axis=1)

    def res(x):
        return np.linalg.norm(np.cross(wt, _vec(*x)), axis=1) - nrm_s

    r = least_squares(res, np.array(_sph(j0t)), loss="soft_l1", f_scale=0.3)
    jt = _vec(*r.x)
    if jt @ R_NOM.inv().apply([0, 0, 1]) < 0: jt = -jt
    return jt


def anat_frame(j, up):
    z = j / np.linalg.norm(j)
    y = up - (up @ z) * z; y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return R.from_matrix(np.column_stack([x, y, z]))


def windows(bounds, name):
    return [(a, b) for (n, a, b) in bounds if n == name]


def mask_of(tg, wins, pad=0.0):
    m = np.zeros(len(tg), bool)
    for a, b in wins:
        m |= (tg >= a + pad) & (tg < b - pad)
    return m


def detect_events(tg, w_sag, walk_mask, T_hint=1.1):
    """Mid-swing = shank angular-velocity peak; HS = next minimum; TO = minimum before mid-swing."""
    x = np.degrees(lpf(w_sag, 8.0))
    xm = np.where(walk_mask, x, 0)
    pk, _ = find_peaks(xm, height=120, distance=int(0.6 * T_hint * FS), prominence=100)
    hs, to = [], []
    for p in pk:
        a, b = p, min(p + int(0.25 * FS), len(x) - 1)
        if b - a > 3 and walk_mask[a:b].all():
            loc, _ = find_peaks(-x[a:b])
            hs.append(tg[a + (loc[0] if len(loc) else np.argmin(x[a:b]))])
        a = max(p - int(0.35 * FS), 0)
        if walk_mask[a:p].all() and p - a > 3:
            to.append(tg[a + np.argmin(x[a:p])])
    return np.array(hs), np.array(to)


def strides_from(hs, walk_wins, T_hint):
    st = []
    for a, b in walk_wins:
        h = hs[(hs >= a) & (hs < b)]
        for i in range(len(h) - 1):
            if 0.7 * T_hint < h[i + 1] - h[i] < 1.4 * T_hint:
                st.append((h[i], h[i + 1]))
    return st


def cam_flex(lm):
    a = lm[:, 0] - lm[:, 1]; b = lm[:, 2] - lm[:, 1]
    cosv = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return 180 - np.degrees(np.arccos(np.clip(cosv, -1, 1)))


def sync_camera(tg, flex_imu, cam):
    fc = cam_flex(cam["lm"]); ok = ~np.isnan(fc)
    tc, fc = cam["t"][ok], fc[ok]
    lags = np.arange(-0.05, 0.45, 0.002)
    best, bl = -2, 0.0
    for L in lags:
        fi = np.interp(tc - L, tg, flex_imu)
        r = np.corrcoef(fi, fc)[0, 1]
        if r > best: best, bl = r, L
    return bl, best


class Processor:
    def __init__(self, imus, cam, bounds, leg, opts=None):
        self.o = dict(calib="functional", drift="imu+cam", lpf=6.0, lam=0.0, trans="pop",
                      sync=True, knot=2.0)
        self.o.update(opts or {})
        self.bounds, self.leg, self.cam = bounds, leg, cam
        total = bounds[-1][2]
        self.tg = np.arange(0, total - 0.03, 1 / FS)
        self.Qt, self.gt, self.at = resample(imus["thigh"], self.tg)
        self.Qs, self.gs, self.as_ = resample(imus["shank"], self.tg)
        self.static = mask_of(self.tg, windows(bounds, "static"), 0.5)
        self.walk_wins = windows(bounds, "walk")
        self.walk = mask_of(self.tg, self.walk_wins, 0.3)

    # ---------------------------------------------------------------- calibration
    def calibrate(self):
        tg = self.tg
        up_t = np.mean(self.at[self.static], 0); up_s = np.mean(self.as_[self.static], 0)
        lat = R_NOM.inv().apply([0, 0, 1])
        upt, ups = up_t / np.linalg.norm(up_t), up_s / np.linalg.norm(up_s)
        if self.o["calib"] in ("functional", "functional1"):
            m = mask_of(tg, windows(self.bounds, f"calib{self.leg}"), 0.2)
            m |= mask_of(tg, self.walk_wins[:1], 0.5)  # first pass: auto-calibrates during walking
            idx = np.where(m)[0][::4]
            jt, js = joint_axes(self.gt[idx], self.gs[idx], lat, lat)
            self._set_frames(anat_frame(jt, upt), anat_frame(js, ups))
            if self.o["calib"] == "functional":
                # pass 2: screw-home rotation near extension tilts the functional axis by 10-15 deg;
                # refit using only samples with flexion > 35 deg where IE coupling is negligible
                flex = self.angles(np.zeros(len(tg)))[0][:, 0]
                mc = mask_of(tg, windows(self.bounds, f"calib{self.leg}"), 0.2) & (flex > 35)
                idx2 = np.where(m & (flex > 35))[0][::2]
                if len(idx2) > 80:
                    jt, _ = joint_axes(self.gt[idx2], self.gs[idx2], jt, js)
                if mc.sum() > 50:
                    # thigh is still during the standing knee-flexion trial -> shank gyro direction IS the axis
                    js = np.linalg.svd(self.gs[mc], full_matrices=False)[2][0]
                    if js @ lat < 0: js = -js
                # heading offset from flexed postures, where both sensors see the same physical hinge axis
                self._set_frames(anat_frame(jt, upt), anat_frame(js, ups), mask=m & (flex > 35))
        else:  # assume the pod sits exactly as designed
            self._set_frames(R_NOM.inv(), R_NOM.inv())

    def _set_frames(self, Ct, Cs, mask=None):
        self.Ct, self.Cs = Ct, Cs
        # heading alignment of the two arbitrary sensor world frames
        hm = self.static if mask is None or mask.sum() < 50 else mask
        zt = (self.Qt[hm] * self.Ct).apply([0, 0, 1])
        zs = (self.Qs[hm] * self.Cs).apply([0, 0, 1])
        d = wrap(yaw_of(zs) - yaw_of(zt))
        self.d0 = np.arctan2(np.mean(np.sin(d)), np.mean(np.cos(d)))
        self.Rf = self.Qt * self.Ct
        self.Rt0 = Ry(-self.d0) * self.Qs * self.Cs
        k = self.Rf[self.static].inv() * self.Rt0[self.static]
        f, a, i = knee_angles(k)
        self.ref = np.array([f.mean(), a.mean(), i.mean()])

    def angles(self, delta):
        Rt = Ry(-delta) * self.Rt0
        f, a, i = knee_angles(self.Rf.inv() * Rt)
        return np.column_stack([f, a, i]) - self.ref, Rt

    # ---------------------------------------------------------------- events
    def events(self):
        w_sag = self.gs @ self.Cs.apply([0, 0, 1])
        self.T_hint = 1.1
        hs, to = detect_events(self.tg, w_sag, self.walk, self.T_hint)
        if len(hs) > 3:
            self.T_hint = np.median(np.diff(hs)[np.diff(hs) < 2.0])
            hs, to = detect_events(self.tg, w_sag, self.walk, self.T_hint)
        self.hs, self.to = hs, to
        self.strides = strides_from(hs, [(a + 0.6, b - 0.6) for a, b in self.walk_wins], self.T_hint)

    # ---------------------------------------------------------------- drift
    def _standing_windows(self):
        g = np.linalg.norm(self.gt, axis=1) + np.linalg.norm(self.gs, axis=1)
        still = lpf(g, 2.0) < 0.35
        still &= ~mask_of(self.tg, windows(self.bounds, f"calib{self.leg}"))
        wins, i, n = [], 0, len(still)
        while i < n:
            if still[i]:
                j = i
                while j < n and still[j]: j += 1
                if (j - i) / FS > 0.8: wins.append((i, j))
                i = j
            else:
                i += 1
        return wins

    def _cam_obs(self, Rf, Rt):
        cam = self.cam; lm = cam["lm"]; tcs = cam["t"] - self.tau
        ok = ~np.isnan(lm[:, 0, 0]) & (tcs > 0) & (tcs < self.tg[-1])
        if ok.sum() < 20: return []
        ii = np.searchsorted(self.tg, tcs[ok])
        ii = np.clip(ii, 0, len(self.tg) - 1)
        L = lm[ok]
        out = []
        frames = np.where(ok)[0]
        for seg, (p0, p1), Rs in (("f", (0, 1), Rf), ("s", (1, 2), Rt)):
            vc = L[:, p1] - L[:, p0]
            vi = Rs[ii].apply([0, -1, 0])
            incl = np.arccos(np.clip(-vc[:, 1] / np.linalg.norm(vc, axis=1), -1, 1))
            dyaw = wrap(yaw_of(vi) - yaw_of(vc))
            out.append((dyaw, np.sin(incl) ** 2 * (incl > np.radians(15))))
        # group into contiguous windows
        gaps = np.where(np.diff(frames) > 15)[0]
        groups = np.split(np.arange(len(frames)), gaps + 1)
        obs = []
        for g in groups:
            wf, ws = out[0][1][g], out[1][1][g]
            if wf.sum() < 3 or ws.sum() < 3: continue
            mf = np.arctan2(np.sum(wf * np.sin(out[0][0][g])), np.sum(wf * np.cos(out[0][0][g])))
            ms = np.arctan2(np.sum(ws * np.sin(out[1][0][g])), np.sum(ws * np.cos(out[1][0][g])))
            # robust spread
            rf = wrap(out[0][0][g] - mf); rs = wrap(out[1][0][g] - ms)
            sd = np.sqrt(np.average(rf ** 2, weights=wf + 1e-9) / max(wf.sum(), 1)
                         + np.average(rs ** 2, weights=ws + 1e-9) / max(ws.sum(), 1))
            obs.append((np.mean(tcs[ok][g]), wrap(ms - mf), max(sd, np.radians(1.0))))
        return obs

    def drift(self):
        tg = self.tg
        delta = np.zeros(len(tg))
        if self.o["drift"] == "none":
            return delta
        knots = np.arange(0, tg[-1] + self.o["knot"], self.o["knot"])
        K = len(knots)

        def basis(t):
            B = np.zeros((len(t), K))
            k = np.clip(np.searchsorted(knots, t, "right") - 1, 0, K - 2)
            u = (t - knots[k]) / (knots[k + 1] - knots[k])
            B[np.arange(len(t)), k] = 1 - u; B[np.arange(len(t)), k + 1] = u
            return B

        Bfull = basis(tg)
        stand = self._standing_windows()
        coef = np.zeros(K)
        t_static = np.mean(tg[self.static])
        for it in range(4):
            ang, Rt = self.angles(delta)
            ie = np.radians(ang[:, 2])
            rows, y, wts = [], [], []
            # anchor at static calibration
            rows.append(np.r_[basis(np.array([t_static]))[0], 0]); y.append(0.0); wts.append(1 / np.radians(0.1))
            for i, j in stand:
                rows.append(np.r_[basis(np.array([tg[(i + j) // 2]]))[0], 0])
                y.append(np.median(ie[i:j])); wts.append(1 / np.radians(1.0))
            for a, b in self.strides:
                m = (tg >= a) & (tg < b)
                rows.append(np.r_[basis(np.array([(a + b) / 2]))[0], 1])
                y.append(np.mean(ie[m])); wts.append(1 / np.radians(0.8))
            if "cam" in self.o["drift"] and self.cam is not None:
                for tc, val, sd in self._cam_obs(self.Rf, Rt):
                    rows.append(np.r_[basis(np.array([tc]))[0], 0]); y.append(val); wts.append(1 / sd)
            # smoothness (2nd difference) prior
            for k in range(1, K - 1):
                r = np.zeros(K + 1); r[k - 1], r[k], r[k + 1] = 1, -2, 1
                rows.append(r); y.append(-(coef[k - 1] - 2 * coef[k] + coef[k + 1]))
                wts.append(1 / np.radians(0.25))
            r = np.zeros(K + 1); r[K] = 1; rows.append(r); y.append(0.0); wts.append(1e-3)  # weak prior on mu
            A = np.array(rows) * np.array(wts)[:, None]; yv = np.array(y) * np.array(wts)
            sol = np.linalg.lstsq(A, yv, rcond=None)[0]
            coef += sol[:K]
            delta = Bfull @ coef
        return delta

    # ---------------------------------------------------------------- translation
    def fit_translation(self, ang):
        if self.o["trans"] != "cam" or self.cam is None:
            return POP["k_ap"], None
        lm = self.cam["lm"]; tcs = self.cam["t"] - self.tau
        ok = ~np.isnan(lm[:, 0, 0]) & (tcs > 0) & (tcs < self.tg[-1]) & mask_of(tcs, self.walk_wins, 0.3)
        if ok.sum() < 30: return POP["k_ap"], None
        D = np.linalg.norm(lm[ok, 2] - lm[ok, 0], axis=1)
        A = np.column_stack([np.interp(tcs[ok], self.tg, ang[:, i]) for i in range(3)])
        Rk = knee_rot(A[:, 0], A[:, 1], A[:, 2])

        def model(x):
            L1, L2, k = x
            q = np.column_stack([POP["k_ml"] * A[:, 1], k * A[:, 0], POP["k_pd"] * A[:, 0]])
            v = np.array([0, -L1, 0]) + trans_to_vec(Rk, q) + Rk.apply([0, -L2, 0])
            return np.linalg.norm(v, axis=1)

        def res(x):
            r = model(x) - D
            return np.r_[r / 0.02, (x[2] - POP["k_ap"]) / POP["k_ap_sd"]]  # Bayesian prior on k

        r = least_squares(res, [0.42, 0.41, POP["k_ap"]], loss="huber", f_scale=2.0)
        return r.x[2], r.x[:2]

    # ---------------------------------------------------------------- run
    def run(self):
        self.calibrate()
        self.events()
        ang0, _ = self.angles(np.zeros(len(self.tg)))
        if self.cam is not None:
            if self.o["sync"]:
                self.tau, self.sync_r = sync_camera(self.tg, ang0[:, 0], self.cam)
            else:
                self.tau, self.sync_r = 0.0, np.nan
        else:
            self.tau = 0.0
        self.delta = self.drift()
        ang, Rt = self.angles(self.delta)
        if self.o.get("xtalk"):
            # flexion->ab/adduction crosstalk minimisation (Marin 2003 / Baudet 2014 style): STA on the thigh
            # rotates the apparent flexion axis, leaking flexion into varus. Remove the linear add~flex
            # slope fitted over walking strides (also removes any genuine linear coupling - documented caveat)
            mw = self.walk
            X = np.column_stack([np.ones(mw.sum()), ang[mw, 0]])
            beta = np.linalg.lstsq(X, ang[mw, 1], rcond=None)[0]
            self.xtalk_beta = beta[1]
            ang[:, 1] = ang[:, 1] - beta[1] * ang[:, 0]
        self.ang_unfilt = ang.copy()
        if self.o["lpf"]:
            ang = lpf(ang, self.o["lpf"])
        lam = self.o["lam"]
        if lam > 0:
            # joint-constraint shrinkage of secondary rotations toward a flexion-coupling model
            # fitted on the calibration flexion trial (rigid-joint prior; residual treated as STA)
            m = mask_of(self.tg, windows(self.bounds, f"calib{self.leg}"), 0.2)
            X = np.column_stack([np.ones(m.sum()), ang[m, 0], ang[m, 0] ** 2])
            Xa = np.column_stack([np.ones(len(ang)), ang[:, 0], ang[:, 0] ** 2])
            for c in (1, 2):
                beta = np.linalg.lstsq(X, ang[m, c], rcond=None)[0]
                model = Xa @ beta
                ang[:, c] = model + (ang[:, c] - model) / (1 + lam)
        self.ang = ang
        k_ap, self.Ls = self.fit_translation(ang)
        self.k_ap = k_ap
        self.q = np.column_stack([POP["k_ml"] * ang[:, 1], k_ap * ang[:, 0], POP["k_pd"] * ang[:, 0]])
        return self


# --------------------------------------------------------------------------- gait-cycle tools
def normalize_cycles(t, x, strides, n=101):
    out = []
    for a, b in strides:
        tt = np.linspace(a, b, n)
        out.append(np.interp(tt, t, x) if x.ndim == 1 else np.column_stack(
            [np.interp(tt, t, x[:, i]) for i in range(x.shape[1])]))
    return np.array(out)


def clinical_metrics(cyc6, stride_T=None, stance=None):
    """cyc6: (n,101,6) [flex, add, ie, ml, ap, pd] -> dict on mean cycle."""
    m = cyc6.mean(0)
    f, a, i, ap = m[:, 0], m[:, 1], m[:, 2], m[:, 4]
    out = dict(peak_swing_flex=f[50:].max(), load_flex=f[:30].max(), flex_rom=f.max() - f.min(),
               peak_stance_add=a[:40].max(), varus_thrust=a[:40].max() - a[0], ie_rom=i.max() - i.min(),
               ap_exc=ap.max() - ap.min(), peak_stance_ap=ap[:40].max() - ap[0])
    if stride_T is not None: out["stride_T"] = stride_T
    if stance is not None: out["stance_pct"] = stance
    return out
