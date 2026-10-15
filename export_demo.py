"""Export demo-subject data (healthy + OA) for the dashboard and the clinician report."""
import numpy as np, json
from run_validation import MODES, cohort
from gaitlens.evaluate import run, truth_on_grid
from gaitlens.pipeline import normalize_cycles, clinical_metrics

S = json.load(open("out/summary.json"))
subs = cohort()
demo = {}
for name, s in (("healthy", subs[3]), ("oa", subs[20])):
    d = {}
    for mode, key in (("off", "A_raw"), ("on", "D_gaitlens")):
        r = run(s, 1000 + s["id"], opts=MODES[key], keep=True)
        tr = r["truth"]
        for leg in ("R", "L"):
            P = r[leg]["P"]
            X, w = truth_on_grid(tr, leg, P.tg)
            E = np.column_stack([P.ang, P.q])
            # 6-s window from pass 2 at 50 Hz
            a = [b0 for (n, b0, b1) in tr["bounds"] if n == "walk"][1] + 1.0
            m = (P.tg >= a) & (P.tg < a + 6)
            idx = np.where(m)[0][::4]
            cyc = normalize_cycles(P.tg, E, P.strides)
            met = clinical_metrics(cyc, stride_T=float(np.mean([b - a2 for a2, b in P.strides])))
            ent = d.setdefault(leg, {})
            ent[mode] = dict(ts=E[idx].round(2).tolist(), cyc=cyc.mean(0).round(2).tolist(),
                             cyc_sd=cyc.std(0).round(2).tolist(), met={k: round(float(v), 2) for k, v in met.items()},
                             rmse=r[leg]["rmse"].round(2).tolist())
            ent["truth_ts"] = X[idx].round(2).tolist()
            ent["truth_cyc"] = r[leg]["cyc_t"].round(2).tolist()
            ent["truth_met"] = {k: round(float(v), 2) for k, v in r[leg]["met_t"].items()}
            ent["t"] = (P.tg[idx] - a).round(3).tolist()
            ent["hs"] = [round(h - a, 3) for h in P.hs if a <= h < a + 6]
    demo[name] = d
out = dict(demo=demo, band=S["norm_band"], norm=S["norm_metrics"])
json.dump(out, open("out/demo.json", "w"))
print("ok", len(json.dumps(out)) // 1024, "KB")
