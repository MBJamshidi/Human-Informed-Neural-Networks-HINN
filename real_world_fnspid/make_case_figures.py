"""
12 real-world AAPL case-study figures, decoupled + title-free, into root figures/.

Consumes the saved rolling-quantile CSVs and the deterministic export
(results/plot_arrays.npz). No canvas titles; identification lives in the
LaTeX caption. 'Volatility states' are the rolling local-quantile target
classes (AAPL has no ground-truth latent regime; this is an explicit proxy).
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGROOT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(HERE, "results")
os.makedirs(FIGROOT, exist_ok=True)
sys.path.insert(0, HERE)
from _plot_style import apply_style, save, TEAL, VIOLET, CORAL, GREY  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
import seaborn as sns  # noqa: E402

apply_style()
F = lambda n: os.path.join(FIGROOT, n)
D = np.load(os.path.join(RESULTS, "plot_arrays.npz"), allow_pickle=True)
HIST = pd.read_csv(os.path.join(RESULTS, "history_hinn_rolling.csv"))


def acf(x, nlags=30):
    x = x - x.mean(); var = (x * x).mean() + 1e-12
    return np.array([1.0] + [(x[:-k] * x[k:]).mean() / var for k in range(1, nlags + 1)])


def main():
    plt.clf(); plt.close("all")   # clear any cached state to prevent file mirroring
    print("Rendering 12 AAPL case figures ...")
    price, Y, m = D["price"], D["target_full"], D["mask_full"]
    t = np.arange(len(price))

    # 1. price + volatility-state proxy shading
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(t, price, color=GREY, lw=1.2, zorder=3)
    cols = {0: TEAL, 1: "#cfcfcf", 2: CORAL}
    labs = {0: "Low-vol state", 1: "Med-vol state", 2: "High-vol state"}
    for k in (0, 1, 2):
        ax.fill_between(t, 0, 1, where=(Y == k), transform=ax.get_xaxis_transform(),
                        color=cols[k], alpha=0.18, step="mid", label=labs[k])
    ax.set_yscale("log"); ax.set_xlabel("Trading Day ($t$)")
    ax.set_ylabel("AAPL Adj. Close (log, USD)")
    ax.set_xlim(0, len(price)); ax.legend(ncol=3, loc="upper left")
    save(fig, F("case_aapl_price_regimes.png"))

    # 2. feature correlation (lower triangle)
    corr = D["feat_corr"]; names = list(D["feat_names"])
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", square=True, xticklabels=names,
                yticklabels=names, annot_kws={"size": 8}, ax=ax,
                cbar_kws={"label": "Pearson Correlation"})
    ax.set_xlabel("Technical Indicator"); ax.set_ylabel("Technical Indicator")
    save(fig, F("case_aapl_feature_correlations.png"))

    # 3 + 4. ACF returns / squared returns
    r = D["r1_full"]; band = 1.96 / np.sqrt(len(r))
    for arr, nm, ylab, c in [(r, "case_aapl_acf_returns.png", r"ACF of $r_t$", TEAL),
                             (r**2, "case_aapl_acf_volatility.png", r"ACF of $r_t^2$", CORAL)]:
        a = acf(arr, 30)
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.bar(range(31), a, color=c, edgecolor=GREY, alpha=0.8)
        ax.axhline(band, color=GREY, ls="--", lw=1.4, label=r"$\pm1.96/\sqrt{T}$")
        ax.axhline(-band, color=GREY, ls="--", lw=1.4)
        ax.set_xlabel("Lag ($k$)"); ax.set_ylabel(ylab); ax.legend()
        save(fig, F(nm))

    # 5. rolling quantile thresholds
    fv, q33, q66 = D["fwd_vol"], D["q33"], D["q66"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(t, fv, color="#bcbcbc", lw=0.7, alpha=0.7, label=r"Forward vol $\sigma_{t+h}$")
    ax.plot(t, q33, color=TEAL, lw=1.8, label=r"Rolling $q_{33}$")
    ax.plot(t, q66, color=CORAL, lw=1.8, label=r"Rolling $q_{66}$")
    ax.set_xlim(0, len(t)); ax.set_xlabel("Trading Day ($t$)")
    ax.set_ylabel("Realized Volatility"); ax.legend(loc="upper right")
    save(fig, F("case_aapl_rolling_quantiles.png"))

    # 6. supervisory mask timeline
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.fill_between(t, 0, m, color=VIOLET, step="mid", alpha=0.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["None", "Avail."])
    ax.set_xlim(0, len(t)); ax.set_xlabel("Trading Day ($t$)")
    ax.set_ylabel(rf"Mask $m_t$ ({m.mean()*100:.1f}%)")
    save(fig, F("case_aapl_supervisory_mask.png"))

    # 7. trust timeline (out-of-sample window)
    lam, mva = D["lambda_va"], D["m_va"]
    win = 400; start = 0
    for s in range(0, len(mva) - win, 50):
        if 0.2 < mva[s:s+win].mean() < 0.8:
            start = s; break
    sl = slice(start, start + win); tt = np.arange(start, start + win)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.fill_between(tt, 0, 1, where=(mva[sl] == 0), transform=ax.get_xaxis_transform(),
                    color=CORAL, alpha=0.12, step="mid", label="No supervision ($m_t=0$)")
    ax.plot(tt, lam[sl], color=TEAL, lw=2.2, label=r"Trust $\lambda(t)$")
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel(r"Trust $\lambda(t)$")
    ax.set_ylim(min(0.45, lam[sl].min() - 0.02), 1.02); ax.legend(loc="lower left")
    save(fig, F("case_aapl_trust_timeline.png"))

    # 8. analytical trust surface
    rho = np.linspace(0, 1, 60); c = np.linspace(0, 1, 60)
    R, C = np.meshgrid(rho, c, indexing="ij")
    Ls = 0.5 + 0.5 * np.exp(-1.5 * R * C)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(R, C, Ls, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel(r"Recency $\rho_t$"); ax.set_ylabel(r"Confidence $c_t$")
    ax.set_zlabel(r"Trust $\lambda(t)$")
    save(fig, F("case_aapl_trust_surface_3d.png"))

    # 9. BNN posterior variance trace
    pv = D["post_var_trace"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(np.arange(1, len(pv) + 1), pv, color=TEAL, lw=2.4)
    ax.fill_between(np.arange(1, len(pv) + 1), pv, color=TEAL, alpha=0.1)
    ax.set_xlabel("Training Epoch ($e$)")
    ax.set_ylabel(r"$\overline{\mathrm{Tr}\,\mathrm{Cov}_q[f_B]}$")
    save(fig, F("case_aapl_bnn_variance_trace.png"))

    # 10. posterior fan chart
    lo, hi = D["fan_window"]; ta = np.arange(lo, hi)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.fill_between(ta, D["bnn_q05"], D["bnn_q95"], color=CORAL, alpha=0.25,
                    label=r"BNN $5\%$–$95\%$ band")
    ax.plot(ta, D["bnn_q50"], color=CORAL, lw=1.8, label="BNN median")
    ax.plot(ta, D["yhat_high"], color=TEAL, lw=1.8, ls="--",
            label=r"Machine $P_{\mathrm{m}}(\mathrm{High})$")
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel("P(High Risk)"); ax.legend(loc="upper right")
    save(fig, F("case_aapl_posterior_fan_chart.png"))

    # 11. reliability comparison
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1.6, label="Perfect calibration")
    ax.plot(D["rel_machine_conf"], D["rel_machine_acc"], "s-", color=TEAL, lw=2.0, ms=8,
            label=f"Machine (ECE={float(D['ece_machine']):.3f})")
    ax.plot(D["rel_human_conf"], D["rel_human_acc"], "o-", color=CORAL, lw=2.0, ms=8,
            label=f"Human (ECE={float(D['ece_human']):.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Expected Confidence"); ax.set_ylabel("Observed Empirical Accuracy")
    ax.legend(loc="upper left")
    save(fig, F("case_aapl_calibration_reliability.png"))

    # 12. normalized confusion matrix
    cm = D["cm_norm"]; labels = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": 13, "weight": "bold"}, ax=ax,
                cbar_kws={"label": "Row-normalized Rate"})
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    save(fig, F("case_aapl_class_confusion_matrix.png"))

    print("12 case figures ->", FIGROOT)


if __name__ == "__main__":
    main()
