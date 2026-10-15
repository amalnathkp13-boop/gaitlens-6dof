"""Run virtual subjects through sensors + pipeline and score against ground truth."""
import numpy as np
from .truth import make_subject, schedule, simulate_truth
from .sensors import default_cfg, sample_imus, sample_camera
from .pipeline import Processor, normalize_cycles, clinical_metrics, mask_of, windows

DOF = ["flex", "add", "ie", "ml", "ap", "pd"]


def truth_on_grid(truth, leg, tg):
    L = truth["legs"][leg]; t = truth["t"]
    x = np.column_stack([np.interp(tg, t, L[k]) for k in ("flex", "add", "ie")]
                        + [np.interp(tg, t, L["q"][:, i]) for i in range(3)])
    st = mask_of(tg, windows(truth["bounds"], "static"), 0.5)
    return x - x[st].mean(0), np.interp(tg, t, L["w"])


def match_err(det, ref, tol=0.15):
    e = []
    for r in ref:
        if len(det) == 0: break
        j = np.argmin(np.abs(det - r))
        if abs(det[j] - r) < tol: e.append(det[j] - r)
    return np.array(e), len(e) / max(len(ref), 1)


def simulate_session(subj, seed, cfg, segs=None):
    rng = np.random.default_rng(seed)
    segs = segs or schedule()
    tr = simulate_truth(subj, rng, segs)
    tau = cfg["cam_tau"] if cfg["cam_tau"] is not None else rng.uniform(0.05, 0.25)
    data = {}
    for leg in ("R", "L"):
        data[leg] = dict(imu=sample_imus(subj, leg, tr, rng, cfg), cam=sample_camera(subj, leg, tr, rng, cfg, tau))
    return tr, data, tau


def score_leg(tr, P, leg):
    X, w = truth_on_grid(tr, leg, P.tg)
    E = np.column_stack([P.ang, P.q])
    m = w > 0.99
    rmse = np.sqrt(np.mean((E[m] - X[m]) ** 2, 0))
    L = tr["legs"][leg]
    hs_e, hs_rate = match_err(P.hs, L["hs"]); to_e, _ = match_err(P.to, L["to"])
    # truth strides on true HS
    hs_t = L["hs"]
    st_t = [(a, b) for a, b in zip(hs_t[:-1], hs_t[1:]) if b - a < 1.6]
    cyc_t = normalize_cycles(P.tg, X, st_t)
    cyc_e = normalize_cycles(P.tg, E, P.strides) if len(P.strides) else None
    out = dict(rmse=rmse, hs_err=hs_e, to_err=to_e, hs_rate=hs_rate,
               met_t=clinical_metrics(cyc_t), met_e=clinical_metrics(cyc_e) if cyc_e is not None else None,
               cyc_t=cyc_t.mean(0), cyc_e=cyc_e.mean(0) if cyc_e is not None else None,
               cyc_e_sd=cyc_e.std(0) if cyc_e is not None else None,
               tau_err=getattr(P, "tau", 0) - tr.get("_tau", 0), n_strides=len(P.strides), k_ap=P.k_ap)
    return out


def run(subj, seed, cfg=None, opts=None, segs=None, keep=False):
    cfg = cfg or default_cfg()
    tr, data, tau = simulate_session(subj, seed, cfg, segs)
    tr["_tau"] = tau
    res = {}
    for leg in ("R", "L"):
        P = Processor(data[leg]["imu"], data[leg]["cam"], tr["bounds"], leg, opts).run()
        res[leg] = score_leg(tr, P, leg)
        if keep:
            res[leg]["P"] = P
    if keep:
        res["truth"] = tr; res["data"] = data
    return res
