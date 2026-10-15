"""Full virtual validation study for GaitLens-6DOF. Writes out/results.pkl"""
import numpy as np, pickle, time, sys
from multiprocessing import Pool
from gaitlens.truth import make_subject, schedule
from gaitlens.sensors import default_cfg
from gaitlens.evaluate import run

MODES = {
    "A_raw":        dict(drift="none", lpf=None),
    "B_drift_imu":  dict(drift="imu", lpf=None),
    "C_drift_cam":  dict(drift="imu+cam", lpf=None),
    "C2_lpf":       dict(drift="imu+cam", lpf=6.0),
    "D_gaitlens":   dict(drift="imu+cam", lpf=6.0, xtalk=True),
    "E_lam1":       dict(drift="imu+cam", lpf=6.0, xtalk=True, lam=1.0),
    "F_lam4":       dict(drift="imu+cam", lpf=6.0, xtalk=True, lam=4.0),
    "X_nominal":    dict(calib="nominal", drift="imu+cam", lpf=6.0, xtalk=True),
    "X_onepass":    dict(calib="functional1", drift="imu+cam", lpf=6.0, xtalk=True),
    "T_camfit":     dict(drift="imu+cam", lpf=6.0, xtalk=True, trans="cam"),
}


def cohort(seed=2026):
    rng = np.random.default_rng(seed)
    subs = [make_subject(rng, "healthy", i) for i in range(20)]
    subs += [make_subject(rng, "oa", 100 + i) for i in range(6)]
    subs += [make_subject(rng, "acl", 200 + i) for i in range(6)]
    return subs


def strip(r):
    return {leg: {k: v for k, v in r[leg].items() if k != "P"} for leg in ("R", "L")}


def job(args):
    kind, subj, seed, cfgkw, opts = args
    return kind, subj["id"], seed, strip(run(subj, seed, default_cfg(**cfgkw), opts))


if __name__ == "__main__":
    t0 = time.time()
    subs = cohort()
    jobs = []
    for s in subs:
        for m, o in MODES.items():
            jobs.append((("main", m), s, 1000 + s["id"], {}, o))
    healthy = [s for s in subs if s["kind"] == "healthy"]
    for s in healthy:  # test-retest: second session = re-strapped sensors, new drift/noise, new day
        jobs.append((("retest", "D_gaitlens"), s, 5000 + s["id"], {}, MODES["D_gaitlens"]))
    for sc in (0.0, 0.5, 1.0, 1.5, 2.0):
        for s in healthy[:10]:
            for m in ("A_raw", "D_gaitlens", "E_lam1"):
                jobs.append((("sta", sc, m), s, 1000 + s["id"], dict(sta_scale=sc), MODES[m]))
    for s in healthy[:10]:
        jobs.append((("nosync",), s, 1000 + s["id"], {}, dict(MODES["D_gaitlens"], sync=False)))
    print(len(jobs), "jobs"); sys.stdout.flush()
    with Pool(2) as pool:
        out = pool.map(job, jobs, chunksize=4)
    res = {}
    for key, sid, seed, r in out:
        res.setdefault(key, {})[sid] = r
    subj_info = {s["id"]: s for s in subs}
    # long-session drift test: 20 passes (~6 min) with faster drift
    long = {}
    rng = np.random.default_rng(7)
    s = make_subject(rng, "healthy", 999)
    segs = schedule(n_pass=24)
    for m in ("A_raw", "B_drift_imu", "C_drift_cam"):
        r = run(s, 77, default_cfg(drift_rate=3.0), dict(MODES[m]), segs=segs, keep=True)
        P = r["R"]["P"]; tr = r["truth"]
        from gaitlens.evaluate import truth_on_grid
        X, w = truth_on_grid(tr, "R", P.tg)
        per = []
        for (n, a, b) in tr["bounds"]:
            if n == "walk":
                mm = (P.tg >= a) & (P.tg < b) & (w > 0.99)
                per.append(((a + b) / 2, np.sqrt(np.mean((P.ang[mm] - X[mm, :3]) ** 2, 0))))
        long[m] = per
    # latency of the real-time path (one 0.25 s window of 2 legs)
    from gaitlens.kin import knee_angles
    from scipy.spatial.transform import Rotation as R
    q = R.random(50, random_state=1)
    tt = time.perf_counter()
    for _ in range(200):
        knee_angles(q.inv() * q)
    lat_ms = (time.perf_counter() - tt) / 200 * 1000
    pickle.dump(dict(res=res, subj=subj_info, long=long, lat_ms=lat_ms, modes=MODES), open("out/results.pkl", "wb"))
    print("done", round(time.time() - t0), "s")
