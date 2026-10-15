"""Summarise out/results.pkl -> out/summary.json + figures."""
import numpy as np, pickle, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pickle.load(open("out/results.pkl", "rb"))
res, subj = D["res"], D["subj"]
DOF = ["Flexion-extension (deg)", "Varus-valgus (deg)", "Internal-external rot. (deg)",
       "Medio-lateral transl. (mm)", "Antero-posterior transl. (mm)", "Proximo-distal transl. (mm)"]
MODES = list(D["modes"].keys())
METRICS = ["peak_swing_flex", "load_flex", "flex_rom", "peak_stance_add", "varus_thrust", "ie_rom",
           "peak_stance_ap"]


def legs_of(key, kinds=None):
    out = []
    for sid, r in res[key].items():
        k = subj[sid]["kind"]
        if kinds and k not in kinds: continue
        for leg in ("R", "L"):
            out.append((sid, k, leg, r[leg]))
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def icc21(Y):
    n, k = Y.shape
    gm = Y.mean()
    msr = k * np.sum((Y.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((Y.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


S = {}
# 1. accuracy per mode
acc = {}
for m in MODES:
    for grp, kinds in (("all", None), ("healthy", ["healthy"]), ("patho", ["oa", "acl"])):
        E = np.array([x["rmse"] for *_, x in legs_of(("main", m), kinds)])
        acc.setdefault(m, {})[grp] = dict(mean=E.mean(0).tolist(), sd=E.std(0).tolist(),
                                          p90=np.percentile(E, 90, axis=0).tolist())
S["accuracy"] = acc

# 2. events + sync
L = legs_of(("main", "D_gaitlens"))
hs = np.concatenate([x["hs_err"] for *_, x in L]) * 1000
to = np.concatenate([x["to_err"] for *_, x in L]) * 1000
S["events"] = dict(hs_bias=float(hs.mean()), hs_sd=float(hs.std()), hs_mae=float(np.abs(hs).mean()),
                   to_bias=float(to.mean()), to_sd=float(to.std()), to_mae=float(np.abs(to).mean()),
                   hs_rate=float(np.mean([x["hs_rate"] for *_, x in L])),
                   hs_mae_bias_corrected=float(np.abs(hs - hs.mean()).mean()))
tau = np.array([x["tau_err"] for *_, x in L]) * 1000
S["sync"] = dict(mae_ms=float(np.abs(tau).mean()), max_ms=float(np.abs(tau).max()))
ns = np.array([x["rmse"] for *_, x in legs_of(("nosync",))])
ws = np.array([x["rmse"] for sid, k, leg, x in L if sid in res[("nosync",)]])
S["nosync"] = dict(with_sync=ws.mean(0).tolist(), without=ns.mean(0).tolist())

# 3. agreement of clinical metrics (est - truth)
agr = {}
for met in METRICS:
    d = np.array([x["met_e"][met] - x["met_t"][met] for *_, x in L])
    agr[met] = dict(bias=float(d.mean()), loa=float(1.96 * d.std()),
                    r=float(np.corrcoef([x["met_e"][met] for *_, x in L], [x["met_t"][met] for *_, x in L])[0, 1]))
S["agreement"] = agr

# 4. test-retest ICC (healthy, both legs)
icc = {}
A_ = {(sid, leg): x for sid, k, leg, x in legs_of(("main", "D_gaitlens"), ["healthy"])}
B_ = {(sid, leg): x for sid, k, leg, x in legs_of(("retest", "D_gaitlens"))}
for met in METRICS:
    keys = [k for k in A_ if k in B_]
    Ye = np.array([[A_[k]["met_e"][met], B_[k]["met_e"][met]] for k in keys])
    Yt = np.array([[A_[k]["met_t"][met], B_[k]["met_t"][met]] for k in keys])
    sem = float(np.sqrt(np.mean((Ye[:, 0] - Ye[:, 1]) ** 2) / 2))
    icc[met] = dict(icc=icc21(Ye), icc_truth=icc21(Yt), sem=sem, mdc95=1.96 * np.sqrt(2) * sem)
S["icc"] = icc

# 5. pathology detection
det = {}
for m in ("C2_lpf", "D_gaitlens", "E_lam1", "F_lam4", "X_nominal"):
    pos = [x["met_e"]["varus_thrust"] for sid, k, leg, x in legs_of(("main", m), ["oa"]) if leg == "R"]
    neg = [x["met_e"]["varus_thrust"] for sid, k, leg, x in legs_of(("main", m), ["healthy"])]
    post = [x["met_t"]["varus_thrust"] for sid, k, leg, x in legs_of(("main", m), ["oa"]) if leg == "R"]
    det[m] = dict(auc=auc(pos, neg), oa_est=float(np.mean(pos)), oa_true=float(np.mean(post)),
                  healthy_est=float(np.mean(neg)), gain=float(np.mean(pos) / np.mean(post)))
pos = [x["met_t"]["varus_thrust"] for sid, k, leg, x in legs_of(("main", "D_gaitlens"), ["oa"]) if leg == "R"]
neg = [x["met_t"]["varus_thrust"] for sid, k, leg, x in legs_of(("main", "D_gaitlens"), ["healthy"])]
det["truth"] = dict(auc=auc(pos, neg))
S["oa_detection"] = det
acl = {}
for m in ("D_gaitlens", "T_camfit"):
    pos = [x["met_e"]["peak_stance_ap"] for sid, k, leg, x in legs_of(("main", m), ["acl"]) if leg == "R"]
    neg = [x["met_e"]["peak_stance_ap"] for sid, k, leg, x in legs_of(("main", m), ["healthy"])]
    ap_h = [x["rmse"][4] for sid, k, leg, x in legs_of(("main", m), ["healthy"])]
    ap_a = [x["rmse"][4] for sid, k, leg, x in legs_of(("main", m), ["acl"]) if leg == "R"]
    kerr = [x["k_ap"] - subj[sid]["legs"][leg]["k_ap"] for sid, k, leg, x in legs_of(("main", m), ["healthy"])]
    acl[m] = dict(auc=auc(pos, neg), ap_rmse_healthy=float(np.mean(ap_h)), ap_rmse_acl=float(np.mean(ap_a)),
                  k_err_sd=float(np.std(kerr)), k_err_bias=float(np.mean(kerr)))
pos = [x["met_t"]["peak_stance_ap"] for sid, k, leg, x in legs_of(("main", "D_gaitlens"), ["acl"]) if leg == "R"]
neg = [x["met_t"]["peak_stance_ap"] for sid, k, leg, x in legs_of(("main", "D_gaitlens"), ["healthy"])]
acl["truth"] = dict(auc=auc(pos, neg))
acl["pop_k_sd"] = float(np.std([subj[sid]["legs"][l]["k_ap"] for sid in subj for l in "RL"
                               if subj[sid]["kind"] == "healthy"]))
S["acl_detection"] = acl

# 6. STA sweep
sw = {}
for key in res:
    if key[0] == "sta":
        E = np.array([r[leg]["rmse"] for r in res[key].values() for leg in "RL"])
        sw.setdefault(key[2], {})[key[1]] = E.mean(0).tolist()
S["sta_sweep"] = sw

# 7. long session
S["long"] = {m: [(float(t), v.tolist()) for t, v in per] for m, per in D["long"].items()}
S["lat_ms"] = D["lat_ms"]

# 8. normative bands from healthy cohort (estimated cycles, GaitLens mode)
cyc = np.array([x["cyc_e"] for *_, x in legs_of(("main", "D_gaitlens"), ["healthy"])])
S["norm_band"] = dict(mean=cyc.mean(0).tolist(), sd=cyc.std(0).tolist())
nm = {met: dict(mean=float(np.mean([x["met_e"][met] for *_, x in legs_of(("main", "D_gaitlens"), ["healthy"])])),
                sd=float(np.std([x["met_e"][met] for *_, x in legs_of(("main", "D_gaitlens"), ["healthy"])])))
      for met in METRICS}
S["norm_metrics"] = nm
json.dump(S, open("out/summary.json", "w"), indent=1)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
lab = {"A_raw": "Raw IMU", "B_drift_imu": "+drift (IMU)", "C_drift_cam": "+camera", "C2_lpf": "+6 Hz LPF",
       "D_gaitlens": "+crosstalk min.\n(GaitLens)", "E_lam1": "+joint prior λ=1", "F_lam4": "+joint prior λ=4",
       "X_nominal": "no functional\ncalibration", "X_onepass": "1-pass axis\n(no screw-home fix)"}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ms = [m for m in lab]
for i in range(3):
    v = [acc[m]["all"]["mean"][i] for m in ms]; e = [acc[m]["all"]["sd"][i] for m in ms]
    ax[i].bar(range(len(ms)), v, yerr=e, color=["#8a8f98"] * 5 + ["#c48a2c"] * 2 + ["#b5483f"] * 2, capsize=2)
    ax[i].axhline([4, 5, 5][i], ls="--", c="k", lw=0.8)
    ax[i].set_xticks(range(len(ms))); ax[i].set_xticklabels([lab[m] for m in ms], rotation=60, ha="right", fontsize=7)
    ax[i].set_title(DOF[i]); ax[i].set_ylabel("RMSE (deg)")
plt.tight_layout(); plt.savefig("out/fig_accuracy.png", dpi=160); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for i in range(3):
    for m, c in (("A_raw", "#8a8f98"), ("D_gaitlens", "#2b6cb0"), ("E_lam1", "#c48a2c")):
        xs = sorted(sw[m]); ax[i].plot(xs, [sw[m][x][i] for x in xs], "o-", c=c, label=lab[m].replace("\n", " "))
    ax[i].set_title(DOF[i]); ax[i].set_xlabel("STA magnitude (x typical)"); ax[i].set_ylabel("RMSE (deg)")
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("out/fig_sta.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(5, 3))
for m, c in (("A_raw", "#8a8f98"), ("B_drift_imu", "#2b6cb0"), ("C_drift_cam", "#2f855a")):
    per = D["long"][m]; ax.plot([t / 60 for t, v in per], [v[2] for t, v in per], "o-", ms=3, c=c,
                                label=lab[m])
ax.set_xlabel("Session time (min)"); ax.set_ylabel("IE rotation RMSE (deg)"); ax.legend(fontsize=7)
ax.set_title("Heading drift over a long session (3 deg/min sensors)")
plt.tight_layout(); plt.savefig("out/fig_long.png", dpi=160); plt.close()
print(json.dumps({k: S[k] for k in ("events", "sync", "oa_detection", "acl_detection")}, indent=1))
for m in MODES:
    print(m, np.round(acc[m]["all"]["mean"], 2))
for met in METRICS:
    print(met, {k: round(v, 3) for k, v in icc[met].items()}, {k: round(v, 2) for k, v in agr[met].items()})
print("long", {m: [round(v[2], 1) for t, v in per] for m, per in D["long"].items()})
