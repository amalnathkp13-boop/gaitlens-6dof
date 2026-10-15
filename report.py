"""One-page clinician report generated from a processed session (demo: OA subject)."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = json.load(open("out/demo.json"))
subj = D["demo"]["oa"]
band_m, band_s = np.array(D["band"]["mean"]), np.array(D["band"]["sd"])
norm = D["norm"]
NAMES = ["Flexion-extension (deg)", "Varus(+)/valgus (deg)", "Internal(+)/external rot. (deg)",
         "Medio-lateral transl. (mm) - model", "Antero-posterior transl. (mm) - model", "Proximo-distal transl. (mm) - model"]
INK, ACC, ACC2, MUT = "#16201e", "#0f6e73", "#b26b14", "#5b6865"
RAG = {"g": ("#2f7d4f", "NORMAL"), "a": ("#b7791f", "WATCH"), "r": ("#b4433a", "FLAG")}


def flag(k, v):
    n = norm[k]; z = abs(v - n["mean"]) / n["sd"]
    return "g" if z < 2 else "a" if z < 3 else "r"


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9aa5a2", "axes.labelcolor": INK})
with PdfPages("out/GaitLens_clinician_report.pdf") as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.965, "GaitLens-6DOF  |  Knee kinematics report", fontsize=15, weight="bold", color=INK)
    fig.text(0.06, 0.948, "Subject: VIRTUAL-OA-100 (synthetic)   Protocol: 3 x 10 m walk, self-selected speed   "
             "Sensors: 4 IMU + side camera   Session date: 15 Oct 2026", fontsize=7.5, color=MUT)
    R, L = subj["R"]["on"]["met"], subj["L"]["on"]["met"]
    sym = 100 * (R["peak_swing_flex"] - L["peak_swing_flex"]) / ((R["peak_swing_flex"] + L["peak_swing_flex"]) / 2)
    lf = [lab for lab, k in (("rotation range", "ie_rom"), ("swing flexion", "peak_swing_flex"), ("varus thrust", "varus_thrust")) if flag(k, L[k]) != "g"]
    LTXT = ("Left knee is within healthy ranges except a borderline " + " and ".join(lf) + ". ") if lf else "Left knee is within healthy ranges. "
    summary = (f"Right knee shows an early-stance varus thrust of {R['varus_thrust']:.1f} deg (healthy "
               f"{norm['varus_thrust']['mean']:.1f} +/- {2*norm['varus_thrust']['sd']:.1f}), consistent with medial-"
               f"compartment loading. Right peak swing flexion is reduced to {R['peak_swing_flex']:.0f} deg "
               f"({sym:+.0f}% vs left), a stiff-knee swing pattern. " + LTXT +
               f"Stride time {R['stride_T']:.2f} s.")
    ax = fig.add_axes([0.06, 0.865, 0.88, 0.07]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 0.006, 1, color=ACC, transform=ax.transAxes))
    import textwrap
    ax.text(0.02, 0.5, "\n".join(textwrap.wrap(summary, 118)), va="center", fontsize=8.6, color=INK, transform=ax.transAxes)

    # metric table with RAG
    rows = [("Peak swing flexion", "peak_swing_flex", "deg"), ("Loading-response flexion", "load_flex", "deg"),
            ("Varus thrust (early stance)", "varus_thrust", "deg"), ("Rotation range (IE)", "ie_rom", "deg")]
    ax = fig.add_axes([0.06, 0.70, 0.88, 0.15]); ax.axis("off")
    cols = [0.0, 0.36, 0.50, 0.64, 0.84]
    for c, h in zip(cols, ["Measure", "Right", "Left", "Healthy range", "Status R / L"]):
        ax.text(c, 0.95, h.upper(), fontsize=6.8, color=MUT, weight="bold")
    for i, (lab, k, u) in enumerate(rows):
        y = 0.75 - i * 0.2
        ax.axhline(y + 0.1, color="#d9e0dd", lw=0.6)
        ax.text(cols[0], y, lab, fontsize=8.2)
        ax.text(cols[1], y, f"{R[k]:.1f} {u}", fontsize=8.2, family="monospace")
        ax.text(cols[2], y, f"{L[k]:.1f} {u}", fontsize=8.2, family="monospace")
        ax.text(cols[3], y, f"{norm[k]['mean']:.1f} +/- {2*norm[k]['sd']:.1f}", fontsize=8.2, family="monospace")
        for j, side in enumerate((R, L)):
            col, word = RAG[flag(k, side[k])]
            ax.text(cols[4] + j * 0.08, y, word, fontsize=6.8, weight="bold", color=col)

    # waveforms
    for i in range(6):
        r, c = divmod(i, 3)
        ax = fig.add_axes([0.07 + c * 0.31, 0.47 - r * 0.21, 0.25, 0.16])
        x = np.arange(101)
        ax.fill_between(x, band_m[:, i] - 2 * band_s[:, i], band_m[:, i] + 2 * band_s[:, i], color=ACC, alpha=0.10, lw=0)
        for side, col, ls in (("R", ACC, "-"), ("L", ACC2, "--")):
            y = np.array(subj[side]["on"]["cyc"])[:, i]
            ax.plot(x, y, color=col, ls=ls, lw=1.4, label="Right" if side == "R" else "Left")
        ax.axvline(60, color="#c8d0cd", lw=0.6, ls=":")
        ax.set_title(NAMES[i], fontsize=7.6, loc="left", color=INK)
        ax.set_xlabel("% gait cycle (stance | swing)", fontsize=6.5)
        if i == 0: ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.text(0.06, 0.195, "Shaded: healthy reference band (mean +/- 2 SD). Dotted line: typical toe-off.",
             fontsize=6.8, color=MUT)

    notes = ("Method: four skin-mounted IMUs (thigh and shank, both legs) with functional two-pass axis calibration, "
             "heading-drift correction from standing pauses and stride stationarity, flexion-varus crosstalk minimisation "
             "and 6 Hz zero-phase filtering. Rotations follow the Grood-Suntay joint coordinate system relative to quiet "
             "standing. Translations are estimated from a tibiofemoral coupling model driven by the measured rotations and "
             "are not independent measurements; do not use them to assess ligament laxity. Expected accuracy (virtual "
             "validation, 64 knees): flexion 1.8 deg, varus 1.8 deg, rotation 2.6 deg RMSE. Peak flexion reads ~4 deg low "
             "due to soft-tissue artefact; compare against this device's own reference band only. "
             "Minimal detectable change (95%): varus thrust 0.7 deg, peak swing flexion 2.5 deg.")
    fig.text(0.06, 0.175, "\n".join(textwrap.wrap(notes, 150)), fontsize=6.6, color=MUT, va="top")
    fig.text(0.06, 0.03, "Research prototype output generated from simulated data. Not a diagnostic device.",
             fontsize=6.6, color="#b4433a")
    pdf.savefig(fig); plt.close(fig)
print("ok")
