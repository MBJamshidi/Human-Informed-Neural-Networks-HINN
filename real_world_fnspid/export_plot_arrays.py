"""
Deterministic export of the per-step prediction arrays needed for Figs 1B/2/3.

This RE-RUNS the frozen rolling-quantile pipeline with the identical seed and
configuration, so it reproduces the already-reported headline numbers EXACTLY
(it asserts acc_va ~ 0.399). It changes no methodology and no results; it only
persists arrays that the original run did not save. Output: results/plot_arrays.npz
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
os.environ.setdefault("HINN_OUTDIR", RESULTS)
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from hinn.simulation import (  # noqa: E402
    softmax, one_hot, trust_lambda, train_hinn, ablation_data_only,
    expected_calibration_error, K, SEED,
)
from run_hinn_aapl_rolling import (  # noqa: E402
    preprocess, make_human, compute_aux_series, TRAIN_FRAC,
)
from run_hinn_aapl import exact_bursty_mask  # noqa: E402

TRUST_KW = dict(lambda_min=0.5, alpha=1.5, gamma=0.05, kappa=2.0)
S_POST = 200          # posterior samples for fan/reliability
FAN_WINDOW = (50, 250)


def reliability(probs, y, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    cs, accs = [], []
    for i in range(n_bins):
        msk = (conf >= edges[i]) & (conf < edges[i + 1])
        if msk.sum() < 5:
            continue
        cs.append(conf[msk].mean()); accs.append(correct[msk].mean())
    return np.array(cs), np.array(accs)


def main():
    dates, X, Y = preprocess(os.path.join(DATA, "AAPL_prices.csv"))
    T = len(X)
    yh = make_human(Y)
    m = exact_bursty_mask(T, coverage=0.466, seed=SEED + 2)
    split = int(TRAIN_FRAC * T)
    mu, sd = X[:split].mean(0), X[:split].std(0) + 1e-8
    Xz = (X - mu) / sd
    X_tr, y_tr, yh_tr, m_tr = Xz[:split], Y[:split], yh[:split], m[:split]
    X_va, y_va, yh_va, m_va = Xz[split:], Y[split:], yh[split:], m[split:]

    print("Re-running frozen pipeline (deterministic) ...")
    res = train_hinn(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                     n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=8, verbose=False)
    res_d = ablation_data_only(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                               n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=4, verbose=False)

    acc_va = res["history"].acc_va.iloc[-1]
    assert abs(acc_va - 0.399) < 0.01, f"reproduction drift: acc_va={acc_va}"
    print(f"  reproduced HINN val acc = {acc_va:.4f} (matches frozen result)")

    rnn, bnn = res["rnn"], res["bnn"]
    rnn_d = res_d["rnn"]

    # validation machine predictions
    Yhat_va = rnn.forward(X_va)["Y"]
    Yhat_va_d = rnn_d.forward(X_va)["Y"]

    # BNN posterior predictive on validation
    rng = np.random.default_rng(SEED + 99)
    eps = rng.normal(size=(S_POST, bnn.n_params))
    thetas = bnn.mu + bnn.sigma * eps
    bnn_probs = np.zeros((S_POST, X_va.shape[0], K))
    for s in range(S_POST):
        zb, _ = bnn.forward_sample(Yhat_va, thetas[s])
        bnn_probs[s] = softmax(zb, axis=-1)
    bnn_mean = bnn_probs.mean(0)
    cov_trace_va = bnn_probs.var(0).sum(-1)

    # per-class metrics
    pc_h = np.column_stack(precision_recall_fscore_support(
        y_va, Yhat_va.argmax(1), labels=[0, 1, 2], zero_division=0)[:3])
    pc_d = np.column_stack(precision_recall_fscore_support(
        y_va, Yhat_va_d.argmax(1), labels=[0, 1, 2], zero_division=0)[:3])

    # out-of-sample trust profile
    lam_va, rho_va, c_va, _ = trust_lambda(np.arange(len(m_va)), m_va,
                                           cov_trace_va, **TRUST_KW)

    # uncertainty fan (high-risk class)
    lo, hi = FAN_WINDOW
    q05, q50, q95 = np.quantile(bnn_probs[:, lo:hi, 2], [0.05, 0.5, 0.95], axis=0)

    # reliability + ECE
    rm_c, rm_a = reliability(Yhat_va, y_va)
    rh_c, rh_a = reliability(bnn_mean, yh_va)
    ece_m = expected_calibration_error(Yhat_va, y_va)
    ece_h = expected_calibration_error(bnn_mean, yh_va)

    # normalized confusion matrix (machine pathway, validation)
    cm = confusion_matrix(y_va, Yhat_va.argmax(1), labels=[0, 1, 2])
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)

    # 10-feature dependency structure (unscaled, clean range)
    feat_names = ["r1", "r2", "r3", "r4", "r5", "MA10", "MA50", "vol20", "RSI", "MACD"]
    feat_corr = np.corrcoef(X.T)

    # auxiliary price / return / rolling-quantile series (clean range)
    aux = compute_aux_series(os.path.join(DATA, "AAPL_prices.csv"))

    out = os.path.join(RESULTS, "plot_arrays.npz")
    np.savez(
        out,
        perclass_hinn=pc_h, perclass_data=pc_d,
        macro_f1_hinn=pc_h[:, 2].mean(), macro_f1_data=pc_d[:, 2].mean(),
        lambda_va=lam_va, rho_va=rho_va, c_va=c_va, m_va=m_va,
        yhat_high=Yhat_va[lo:hi, 2], bnn_q05=q05, bnn_q50=q50, bnn_q95=q95,
        fan_window=np.array(FAN_WINDOW),
        rel_machine_conf=rm_c, rel_machine_acc=rm_a,
        rel_human_conf=rh_c, rel_human_acc=rh_a,
        ece_machine=ece_m, ece_human=ece_h,
        cm_norm=cm_norm,
        feat_corr=feat_corr, feat_names=np.array(feat_names),
        # full-series (clean range) arrays for case figures
        price=aux["price"], target_full=Y, mask_full=m, r1_full=aux["r1"],
        fwd_vol=aux["fwd_vol"], q33=aux["q33"], q66=aux["q66"],
        post_var_trace=res["history"].post_var_mean.values,
    )
    print(f"Exported prediction arrays -> {out}")


if __name__ == "__main__":
    main()
