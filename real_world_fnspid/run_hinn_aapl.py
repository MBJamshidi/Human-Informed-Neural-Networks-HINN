"""
Real-world HINN validation on AAPL (1980-2023) -- Option 1 (synthetic-from-real).

Real non-stationary price substrate (AAPL OHLCV from FNSPID) + the paper's exact
expert-pathology model for the human pathway. The methodology itself is imported
UNMODIFIED from src/hinn/simulation.py (RNNMachine, BNNHuman, trust_lambda,
train_hinn, make_labels, _rsi). Nothing in the synthetic package is touched.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "real_world_fnspid", "results")
DATA = os.path.join(ROOT, "real_world_fnspid", "data")
os.makedirs(RESULTS, exist_ok=True)

# Keep anything simulation.py creates at import isolated inside results/, headless.
os.environ.setdefault("HINN_OUTDIR", RESULTS)
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(ROOT, "src"))

from hinn.simulation import (  # noqa: E402
    _rsi, make_labels, train_hinn, ablation_data_only, ablation_human_only,
    K, SEED,
)


# ---------------------------------------------------------------------------
def build_features(price_csv):
    """10-dim x_t from real AAPL OHLCV, using the paper's indicator definitions."""
    df = pd.read_csv(price_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_monotonic_increasing

    px = df["adj close"].values.astype(float)

    # 5 log-returns: single-asset analogue using the 5 real price fields
    ret_cols = ["open", "high", "low", "close", "adj close"]
    rets = {}
    for c in ret_cols:
        p = np.clip(df[c].values.astype(float), 1e-9, None)
        rets[c] = np.diff(np.log(p), prepend=np.log(p[0]))

    s = pd.Series(px)
    ma10 = s.rolling(10).mean().bfill().values
    ma50 = s.rolling(50).mean().bfill().values
    rsi = _rsi(px, 14)
    vol20 = pd.Series(rets["adj close"]).rolling(20).std().bfill().values
    macd = ma10 - s.rolling(26).mean().bfill().values

    feat = pd.DataFrame({
        "ret_open":  rets["open"],  "ret_high": rets["high"],
        "ret_low":   rets["low"],   "ret_close": rets["close"],
        "ret_adj":   rets["adj close"],
        "ma10": (ma10 - px) / px,   "ma50": (ma50 - px) / px,
        "rsi":  (rsi - 50.0) / 50.0,
        "vol20": vol20,             "macd": macd / px,
    })
    prices = pd.DataFrame({"price_1": px})   # for make_labels()
    return df["date"], feat, prices


def exact_bursty_mask(T, coverage=0.466, burst_len=10, seed=SEED + 1):
    """Bursty + random availability mask with coverage forced to EXACTLY `coverage`.

    Mirrors make_labels' structure (random base + 10-step bursts) but, because the
    real series is far longer than the synthetic one, calibrates the count so that
    m_t.mean() == round(coverage*T)/T.
    """
    rng = np.random.default_rng(seed)
    target = int(round(coverage * T))
    m = np.zeros(T, dtype=int)
    n_bursts = max(1, round(T * 15 / 1260))          # same burst density as paper
    for start in rng.integers(0, T - burst_len, size=n_bursts):
        m[start:start + burst_len] = 1
    cur = int(m.sum())
    if cur < target:
        idx = rng.choice(np.where(m == 0)[0], size=target - cur, replace=False)
        m[idx] = 1
    elif cur > target:
        idx = rng.choice(np.where(m == 1)[0], size=cur - target, replace=False)
        m[idx] = 0
    return m


def headline(r):
    h = r["history"].iloc[-1]
    return {
        "Acc": h.acc_va, "macro_F1": h.f1_va, "ECE": h.ece_tr,
        "Brier": h.brier_tr, "AUC_ovo": h.auc_tr_ovo,
        "lambda_bar": h.lam_mean, "post_var": h.post_var_mean,
        "KL": h.kl, "loss": h.L_total,
    }


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print(" HINN real-world validation -- AAPL (FNSPID), synthetic-from-real expert")
    print("=" * 78)

    dates, feat, prices = build_features(os.path.join(DATA, "AAPL_prices.csv"))
    T = len(feat)
    print(f"[1] Real AAPL substrate: {T} trading days "
          f"({dates.iloc[0].date()} -> {dates.iloc[-1].date()}), "
          f"{feat.shape[1]} features")

    # paper-faithful labels + expert pathologies (10% noise, policy drift)
    y_data, y_human, _m_unused, fwd_vol = make_labels(None, prices)
    m = exact_bursty_mask(T, coverage=0.466)
    print(f"[2] Labels  low={np.sum(y_data==0)} med={np.sum(y_data==1)} "
          f"high={np.sum(y_data==2)} | human noise=10% + drift | "
          f"mask coverage={m.mean()*100:.1f}% "
          f"| human!=data on masked steps={np.mean(y_human[m==1]!=y_data[m==1])*100:.1f}%")

    # standardize + chronological 70/30 split (identical recipe to synthetic main)
    X = feat.values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    split = int(0.7 * T)
    X_tr, y_tr, yh_tr, m_tr = X[:split], y_data[:split], y_human[:split], m[:split]
    X_va, y_va, yh_va, m_va = X[split:], y_data[split:], y_human[split:], m[split:]
    print(f"[3] Split: train={split} ({m_tr.mean()*100:.1f}% sup.)  "
          f"val={T-split} ({m_va.mean()*100:.1f}% sup.)")

    print("\n[4] Training full HINN (manual NumPy BPTT, Algorithm 1) ...")
    res = train_hinn(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                     n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=8)
    print("[5] Ablation: data-only (lambda=1) ...")
    res_data = ablation_data_only(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                                  n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=4,
                                  verbose=False)
    print("[6] Ablation: human-only (lambda=0) ...")
    res_human = ablation_human_only(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                                    n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=4,
                                    verbose=False)

    # ---- dump convergence traces ----
    res["history"].to_csv(os.path.join(RESULTS, "history_hinn.csv"), index=False)
    res_data["history"].to_csv(os.path.join(RESULTS, "history_data_only.csv"), index=False)
    res_human["history"].to_csv(os.path.join(RESULTS, "history_human_only.csv"), index=False)

    # ---- adaptive trust profiles ----
    trust_trace = res["history"][["ep", "lam_mean", "lam_std", "rho_mean",
                                  "c_mean", "post_var_mean", "kl"]]
    trust_trace.to_csv(os.path.join(RESULTS, "trust_lambda_trace.csv"), index=False)
    final = pd.DataFrame({
        "t": np.arange(res["Lambdas"].shape[1]),
        "lambda_final": res["Lambdas"][-1],
        "rho_final": res["Rhos"][-1],
        "c_final": res["Cs"][-1],
        "cov_trace_final": res["Covs"][-1],
        "m_train": m_tr,
    })
    final.to_csv(os.path.join(RESULTS, "trust_profile_final.csv"), index=False)

    # ---- comparative baseline statistics ----
    table = pd.DataFrame({
        "HINN (full)": headline(res),
        "Data-only (lambda=1)": headline(res_data),
        "Human-only (lambda=0)": headline(res_human),
    }).T
    table.to_csv(os.path.join(RESULTS, "headline_metrics.csv"))

    elapsed = time.time() - t0
    print("\n[7] Comparative baseline statistics (validation split):\n")
    print(table.round(4).to_string())
    print(f"\nDone in {elapsed:.1f}s. Results -> {RESULTS}")

    # lightweight provenance report
    with open(os.path.join(RESULTS, "REPORT_real_world.md"), "w", encoding="utf-8") as f:
        f.write("# HINN Real-World Validation - AAPL (FNSPID)\n\n")
        f.write(f"- Substrate: real AAPL OHLCV, {T} days "
                f"({dates.iloc[0].date()} to {dates.iloc[-1].date()})\n")
        f.write("- Human pathway: paper-exact (10% uniform noise + linear policy "
                f"drift), mask coverage {m.mean()*100:.1f}%\n")
        f.write(f"- Methodology: imported unmodified from src/hinn/simulation.py\n")
        f.write(f"- Runtime: {elapsed:.1f}s\n\n## Comparative baseline\n\n")
        f.write(table.round(4).to_markdown())


if __name__ == "__main__":
    main()
