"""
Nature-standard visualization suite for the real-world AAPL HINN case study.

Consumes the saved rolling-quantile result CSVs and, when present, the
deterministic prediction export (plot_arrays.npz from export_plot_arrays.py).
Panels whose arrays were never persisted are skipped with an explicit notice
rather than fabricated.

Figures -> ./real_world_fnspid/results/
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import apply_style, save, PALETTE, TEAL, VIOLET, CORAL, GREY  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CHANCE = 1.0 / 3.0
apply_style()


def _load_histories():
    return {
        "HINN (full)": pd.read_csv(os.path.join(RESULTS, "history_hinn_rolling.csv")),
        "Data-only": pd.read_csv(os.path.join(RESULTS, "history_data_only_rolling.csv")),
        "Human-only": pd.read_csv(os.path.join(RESULTS, "history_human_only_rolling.csv")),
    }


def _load_export():
    p = os.path.join(RESULTS, "plot_arrays.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


# ---------------------------------------------------------------------------
def fig1_learning_and_balance(hist, npz):
    """Fig 1: (A) learning curves [from CSV], (B) per-class balance [needs export]."""
    have_B = npz is not None and "perclass_hinn" in npz
    ncols = 2 if have_B else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.2 * ncols, 5.4))
    axA = axes[0] if have_B else axes

    # ---- Panel A: learning curves ----
    for name, h in hist.items():
        c = PALETTE[name]
        axA.plot(h.ep, h.acc_va, color=c, ls="-", lw=2.4, label=f"{name} — Val")
        axA.plot(h.ep, h.acc_tr, color=c, ls="--", lw=1.5, alpha=0.7, label=f"{name} — Train")
    axA.axhline(CHANCE, color=GREY, ls=":", lw=1.6, label="Random baseline (0.333)")
    axA.set_xlabel("Training Epoch ($e$)")
    axA.set_ylabel("Classification Accuracy")
    axA.set_title("(A) Ablation Learning Curves", loc="left")
    axA.set_ylim(0.25, 0.45)
    axA.legend(ncol=2, fontsize=8.5, loc="lower right")

    # ---- Panel B: per-class precision / recall / F1 ----
    if have_B:
        axB = axes[1]
        classes = ["Low", "Medium", "High"]
        pc_h = npz["perclass_hinn"]   # (3 classes, 3 metrics) P,R,F1
        pc_d = npz["perclass_data"]
        x = np.arange(3)
        w = 0.36
        axB.bar(x - w / 2, pc_h[:, 2], w, color=TEAL, label="HINN — F1", edgecolor="white")
        axB.bar(x + w / 2, pc_d[:, 2], w, color=VIOLET, label="Data-only — F1", edgecolor="white")
        # overlay precision (circle) and recall (triangle) as markers
        axB.scatter(x - w / 2, pc_h[:, 0], marker="o", s=42, facecolor="white",
                    edgecolor=TEAL, zorder=5, label="Precision")
        axB.scatter(x - w / 2, pc_h[:, 1], marker="^", s=46, facecolor=TEAL,
                    edgecolor="white", zorder=5, label="Recall")
        axB.scatter(x + w / 2, pc_d[:, 0], marker="o", s=42, facecolor="white",
                    edgecolor=VIOLET, zorder=5)
        axB.scatter(x + w / 2, pc_d[:, 1], marker="^", s=46, facecolor=VIOLET,
                    edgecolor="white", zorder=5)
        axB.set_xticks(x); axB.set_xticklabels(classes)
        axB.set_xlabel("Risk Class")
        axB.set_ylabel("Per-Class Score")
        axB.set_title("(B) Representational Balance", loc="left")
        axB.set_ylim(0, 0.7)
        mf_h = float(npz["macro_f1_hinn"]); mf_d = float(npz["macro_f1_data"])
        axB.text(0.02, 0.96, f"Macro-F1:  HINN {mf_h:.3f}  vs  Data {mf_d:.3f}\n"
                 f"($\\Delta$ = {mf_h-mf_d:+.3f})",
                 transform=axB.transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="#f3f3f3", ec="0.8"))
        axB.legend(ncol=2, fontsize=8.5, loc="upper right")
        save(fig, os.path.join(RESULTS, "fig1_ablation_balance.png"))
    else:
        save(fig, os.path.join(RESULTS, "fig1A_learning_curves.png"))
        print("   [skip] Fig 1 Panel B (per-class P/R/F1): needs plot_arrays.npz "
              "(run export_plot_arrays.py).")


# ---------------------------------------------------------------------------
def fig2_trust_routing(npz):
    """Fig 2: lambda(t) vs availability mask. OOS if exported, else train preview."""
    if npz is not None and "lambda_va" in npz:
        lam = npz["lambda_va"]; m = npz["m_va"]; tag = "Out-of-Sample (Validation)"
        fname = "fig2_trust_routing.png"
    else:
        prof = pd.read_csv(os.path.join(RESULTS, "trust_profile_final_rolling.csv"))
        lam = prof["lambda_final"].values; m = prof["m_train"].values
        tag = "Training Segment (preview — OOS needs export)"
        fname = "fig2_trust_routing_TRAIN_preview.png"

    # pick a 400-step window that contains mask variation
    win = 400
    start = 0
    for s in range(0, len(m) - win, 50):
        seg = m[s:s + win]
        if 0.2 < seg.mean() < 0.8:
            start = s
            break
    sl = slice(start, start + win)
    t = np.arange(start, start + win)
    lam_s, m_s = lam[sl], m[sl]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    # shade fallback (m_t = 0) zones
    ax.fill_between(t, 0, 1, where=(m_s == 0), color=CORAL, alpha=0.12,
                    transform=ax.get_xaxis_transform(), step="mid",
                    label="No supervision ($m_t=0$)")
    ax.plot(t, lam_s, color=TEAL, lw=2.2, label=r"Trust $\lambda(t)$")
    ax.set_xlabel("Time Step ($t$)")
    ax.set_ylabel(r"Trust Routing $\lambda(t)$")
    ax.set_ylim(min(0.45, lam_s.min() - 0.02), 1.02)
    ax.set_title(rf"Trust-Routing Dynamics — {tag}", loc="left")

    ax2 = ax.twinx()
    ax2.step(t, m_s, where="mid", color=GREY, lw=1.0, alpha=0.55)
    ax2.set_ylabel(r"Availability Mask $m_t$")
    ax2.set_ylim(-0.05, 1.6); ax2.set_yticks([0, 1])
    ax2.grid(False)
    ax.legend(loc="lower left", fontsize=9)
    save(fig, os.path.join(RESULTS, fname))
    if "lambda_va" not in (npz or {}):
        print("   [note] Fig 2 shows the TRAIN segment; the requested out-of-sample "
              "lambda(t) needs plot_arrays.npz (run export_plot_arrays.py).")


# ---------------------------------------------------------------------------
def fig3_calibration_uncertainty(npz):
    """Fig 3: (A) uncertainty fan, (B) reliability — both need the export."""
    if npz is None or "yhat_high" not in npz:
        print("   [skip] Fig 3 (uncertainty fan + reliability): needs plot_arrays.npz "
              "(run export_plot_arrays.py).")
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.4, 5.4))

    # ---- Panel A: uncertainty fan ----
    lo, hi = npz["fan_window"]
    t = np.arange(lo, hi)
    axA.fill_between(t, npz["bnn_q05"], npz["bnn_q95"], color=CORAL, alpha=0.25,
                     label=r"BNN $5\%$–$95\%$ posterior band")
    axA.plot(t, npz["bnn_q50"], color=CORAL, lw=1.8, label="BNN posterior median")
    axA.plot(t, npz["yhat_high"], color=TEAL, lw=1.8, ls="--",
             label=r"Machine $P_{\mathrm{m}}(\mathrm{High})$")
    axA.set_xlabel("Time Step ($t$)")
    axA.set_ylabel("Predicted Probability (High Risk)")
    axA.set_title("(A) Posterior-Predictive Uncertainty", loc="left")
    axA.legend(loc="upper right", fontsize=8.5)

    # ---- Panel B: reliability ----
    axB.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1.6, label="Perfect calibration")
    rm_c, rm_a = npz["rel_machine_conf"], npz["rel_machine_acc"]
    rh_c, rh_a = npz["rel_human_conf"], npz["rel_human_acc"]
    axB.plot(rm_c, rm_a, "s-", color=TEAL, lw=2.0, ms=7,
             label=f"Machine (ECE={float(npz['ece_machine']):.3f})")
    axB.plot(rh_c, rh_a, "o-", color=CORAL, lw=2.0, ms=7,
             label=f"Human (ECE={float(npz['ece_human']):.3f})")
    axB.set_xlabel("Expected Confidence")
    axB.set_ylabel("Observed Empirical Accuracy")
    axB.set_title("(B) Reliability Comparison", loc="left")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    axB.legend(loc="upper left", fontsize=9)
    save(fig, os.path.join(RESULTS, "fig3_calibration_uncertainty.png"))


def main():
    print("Generating real-world figure suite ...")
    hist = _load_histories()
    npz = _load_export()
    if npz is None:
        print("   [info] plot_arrays.npz not found -> generating CSV-supported panels only.")
    fig1_learning_and_balance(hist, npz)
    fig2_trust_routing(npz)
    fig3_calibration_uncertainty(npz)
    print("Done.")


if __name__ == "__main__":
    main()
