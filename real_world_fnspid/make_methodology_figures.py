"""
10 core HINN methodology figures (SYNTHETIC pipeline), decoupled + title-free.

Trains the synthetic HINN once (no ablations / no hp grid) and renders ten
standalone figures in the shared sans-serif style into the root figures/ dir.
The synthetic generation code in src/hinn/simulation.py is imported UNMODIFIED.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGROOT = os.path.join(ROOT, "figures")
os.makedirs(FIGROOT, exist_ok=True)
os.environ.setdefault("HINN_OUTDIR", os.path.join(ROOT, "synthetic_baseline"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from hinn.simulation import (  # noqa: E402
    generate_financial_dataset, make_labels, train_hinn, one_hot, softmax, K,
)
from _plot_style import apply_style, save, TEAL, VIOLET, CORAL, GREY  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

apply_style()
F = lambda n: os.path.join(FIGROOT, n)


def train_synthetic():
    big, prices, regimes = generate_financial_dataset(n_days=1260)
    y_data, y_human, m, _ = make_labels(big, prices)
    cols = ["ret_idx", "ret_a2", "ret_a3", "ret_a4", "ret_a5",
            "ma10", "ma50", "rsi", "vol20", "macd"]
    X = big[cols].values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    s = int(0.7 * len(X))
    res = train_hinn(X[:s], y_data[:s], y_human[:s], m[:s],
                     X[s:], y_data[s:], y_human[s:], m[s:],
                     n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=8, verbose=False)
    return res, X[:s], y_data[:s]


def main():
    print("Training synthetic HINN for methodology figures ...")
    res, X_tr, y_tr = train_synthetic()
    h = res["history"]

    # 1. loss decomposition
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot(h.ep, h.L_total, color=GREY, lw=2.6, label=r"$\mathcal{L}_{\mathrm{HINN}}$")
    ax.plot(h.ep, h.L_data, color=TEAL, ls="--", label=r"$\mathcal{L}_{\mathrm{data}}$")
    ax.plot(h.ep, h.L_human, color=VIOLET, ls="-.", label=r"$\mathcal{L}_{\mathrm{human}}$")
    ax.plot(h.ep, h.L_kl, color=CORAL, alpha=0.8, label=r"$\beta\,\mathrm{KL}$")
    ax.set_xlabel("Training Epoch ($e$)"); ax.set_ylabel("Loss Magnitude"); ax.legend()
    save(fig, F("hinn_loss_decomposition.png"))

    # 2. KL divergence
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot(h.ep, h.kl, color=TEAL, lw=2.4)
    ax.fill_between(h.ep, h.kl, color=TEAL, alpha=0.1)
    ax.set_xlabel("Training Epoch ($e$)"); ax.set_ylabel(r"$\mathrm{KL}(q\,\|\,p)$ (nats)")
    save(fig, F("hinn_kl_divergence.png"))

    # 3. accuracy
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot(h.ep, h.acc_tr, color=TEAL, ls="--", label="Training Accuracy")
    ax.plot(h.ep, h.acc_va, color=VIOLET, lw=2.4, label="Validation Accuracy")
    ax.set_xlabel("Training Epoch ($e$)"); ax.set_ylabel("Classification Accuracy"); ax.legend()
    save(fig, F("hinn_accuracy_evolution.png"))

    # 4. macro-F1
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot(h.ep, h.f1_tr, color=TEAL, ls="--", label="Training Macro-F1")
    ax.plot(h.ep, h.f1_va, color=VIOLET, lw=2.4, label="Validation Macro-F1")
    ax.set_xlabel("Training Epoch ($e$)"); ax.set_ylabel("Macro-F1 Score"); ax.legend()
    save(fig, F("hinn_f1_evolution.png"))

    # 5. gradient norm
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot(h.ep, h.grad_norm, color=TEAL, lw=2.2)
    ax.axhline(5.0, color=GREY, ls="--", lw=1.6, label=r"Clip Threshold $\tau=5$")
    ax.set_yscale("log"); ax.set_xlabel("Training Epoch ($e$)")
    ax.set_ylabel(r"Gradient Norm $\|\nabla_{\Theta_m}\mathcal{L}\|_2$"); ax.legend()
    save(fig, F("hinn_gradient_norm.png"))

    # 6. clip scale
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot(h.ep, h.grad_scale, color=CORAL, lw=2.2)
    ax.set_ylim(0, 1.1); ax.set_xlabel("Training Epoch ($e$)")
    ax.set_ylabel(r"Clip Scale Factor (at $\tau=5$)")
    save(fig, F("hinn_gradient_clip_scale.png"))

    # 7 + 8. loss landscape (3D + contour)
    rnn = res["rnn"]
    bx, bh = rnn.W_xh[0, 0], rnn.W_hh[0, 0]
    g = np.linspace(-0.6, 0.6, 22)
    L = np.zeros((len(g), len(g)))
    Y_oh = one_hot(y_tr, K)
    for i, dx in enumerate(g):
        for j, dh in enumerate(g):
            rnn.W_xh[0, 0] = bx + dx; rnn.W_hh[0, 0] = bh + dh
            Y = rnn.forward(X_tr)["Y"]
            L[i, j] = -np.sum(Y_oh * np.log(Y + 1e-9), axis=1).mean()
    rnn.W_xh[0, 0], rnn.W_hh[0, 0] = bx, bh
    XX, YY = np.meshgrid(g, g, indexing="ij")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, L, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel(r"$\Delta W_{xh}[0,0]$"); ax.set_ylabel(r"$\Delta W_{hh}[0,0]$")
    ax.set_zlabel(r"$\mathcal{L}_{\mathrm{data}}$")
    save(fig, F("hinn_loss_landscape_3d.png"))

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cs = ax.contourf(XX, YY, L, levels=20, cmap="viridis")
    ax.contour(XX, YY, L, levels=10, colors="white", linewidths=0.5, alpha=0.5)
    ax.plot(0, 0, marker="*", color="white", markersize=16, markeredgecolor=GREY)
    ax.set_xlabel(r"$\Delta W_{xh}[0,0]$"); ax.set_ylabel(r"$\Delta W_{hh}[0,0]$")
    fig.colorbar(cs, label="Cross-Entropy Loss")
    save(fig, F("hinn_loss_contour_map.png"))

    # 9. output simplex trajectory
    YH = res["Yhat_history"]
    modes = np.zeros((YH.shape[0], K))
    for e in range(YH.shape[0]):
        am = YH[e].argmax(1)
        modes[e] = [(am == k).mean() for k in range(K)]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.viridis(np.linspace(0.2, 1, len(modes)))
    for i in range(len(modes) - 1):
        ax.plot(modes[i:i+2, 0], modes[i:i+2, 1], modes[i:i+2, 2], "-", color=cmap[i], lw=2.4)
    ax.scatter(*modes[0], s=120, c=[GREY], marker="o", label="Epoch 1", depthshade=False)
    ax.scatter(*modes[-1], s=180, c=[TEAL], marker="*", label=f"Epoch {len(modes)}", depthshade=False)
    ax.set_xlabel("Low-Risk Mass"); ax.set_ylabel("Med-Risk Mass"); ax.set_zlabel("High-Risk Mass")
    ax.legend()
    save(fig, F("hinn_output_simplex_trajectory.png"))

    # 10. summary dashboard (8 panels, title-free; panel id via bold ylabel)
    panels = [("L_total", r"$\mathcal{L}_{\mathrm{HINN}}$", GREY),
              ("acc_va", "Val Accuracy", TEAL),
              ("lam_mean", r"$\overline{\lambda(t)}$", VIOLET),
              ("post_var_mean", r"$\overline{\mathrm{Tr}(\Sigma_q)}$", TEAL),
              ("kl", "KL Divergence", CORAL),
              ("ece_tr", "ECE (Train)", VIOLET),
              ("brier_tr", "Brier (Train)", TEAL),
              ("auc_tr_ovo", "AUC$_{\\mathrm{ovo}}$", GREY)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (col, lbl, c) in zip(axes.ravel(), panels):
        ax.plot(h.ep, h[col], color=c, lw=2.2)
        ax.fill_between(h.ep, h[col], color=c, alpha=0.1)
        ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel(lbl)
    fig.tight_layout()
    save(fig, F("hinn_summary_dashboard.png"))
    print("10 methodology figures ->", FIGROOT)


if __name__ == "__main__":
    main()
