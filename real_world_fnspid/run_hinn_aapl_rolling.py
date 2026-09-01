"""
Real-world HINN on AAPL -- rolling LOCAL-quantile targets + leak-free scaling.

Corrects the global-quantile baseline (which fell below chance under 43-yr drift)
by binning forward volatility against a trailing 252-day local distribution, so
targets stay locally balanced. Methodology imported UNMODIFIED from
src/hinn/simulation.py. Fixes applied vs the proposed snippet:
  * real RSI / MACD (NOT sin/cos of price)
  * lowercase real column names; adj-close log-returns
  * train-only z-scaling (no validation leakage)
  * seeded bursty mask at exactly 46.6%
  * paper-faithful gradual policy drift (not a blanket +1 on the tail)
  * realized-only quantile window (no look-ahead)
"""
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "real_world_fnspid", "results")
DATA = os.path.join(ROOT, "real_world_fnspid", "data")
os.environ.setdefault("HINN_OUTDIR", RESULTS)
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "real_world_fnspid"))

from hinn.simulation import (  # noqa: E402
    _rsi, train_hinn, ablation_data_only, ablation_human_only, K, SEED,
)
from run_hinn_aapl import exact_bursty_mask, headline  # noqa: E402

LOOKBACK = 252
HORIZON = 5
TRAIN_FRAC = 0.70


def preprocess(price_csv):
    df = pd.read_csv(price_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    px = np.clip(df["adj close"].values.astype(float), 1e-9, None)
    r1 = np.diff(np.log(px), prepend=np.log(px[0]))
    df["r1"] = r1

    # forward HORIZON-day realized vol; entry i known only at i+HORIZON
    df["fwd_vol"] = pd.Series(r1).rolling(HORIZON).std().shift(-HORIZON).values

    # rolling LOCAL-quantile targets (trailing realized-only window)
    Y = np.zeros(len(df), dtype=int)
    fv = df["fwd_vol"].values
    for t in range(LOOKBACK, len(df) - HORIZON):
        window = fv[t - LOOKBACK:t - HORIZON]      # realized as of t
        window = window[~np.isnan(window)]
        if window.size == 0 or np.isnan(fv[t]):
            continue
        q33, q66 = np.percentile(window, [33, 66])
        Y[t] = 1 if fv[t] > q33 else 0
        if fv[t] > q66:
            Y[t] = 2
    df["target"] = Y

    # 5 log-returns = current + 4 lags (single-asset analogue)
    for k in range(2, 6):
        df[f"r{k}"] = pd.Series(r1).shift(k - 1).fillna(0.0).values

    s = pd.Series(px)
    ma10 = s.rolling(10).mean().bfill().values
    ma50 = s.rolling(50).mean().bfill().values
    ma26 = s.rolling(26).mean().bfill().values
    df["MA10"] = (ma10 - px) / px
    df["MA50"] = (ma50 - px) / px
    df["vol20"] = pd.Series(r1).rolling(20).std().bfill().values
    df["RSI"] = (_rsi(px, 14) - 50.0) / 50.0          # real RSI
    df["MACD"] = (ma10 - ma26) / px                    # real MACD proxy (paper def.)

    clean = df.iloc[LOOKBACK:len(df) - HORIZON].copy().reset_index(drop=True)
    feat_cols = ["r1", "r2", "r3", "r4", "r5", "MA10", "MA50", "vol20", "RSI", "MACD"]
    X = clean[feat_cols].values.astype(float)
    Y = clean["target"].values.astype(int)
    return clean["date"], X, Y


def compute_aux_series(price_csv):
    """Auxiliary series over the SAME clean range as preprocess(), for plotting.

    Returns price, r1, forward vol, and the per-step rolling q33/q66 thresholds.
    Additive helper -- does not alter preprocess()'s contract.
    """
    df = pd.read_csv(price_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    px = np.clip(df["adj close"].values.astype(float), 1e-9, None)
    r1 = np.diff(np.log(px), prepend=np.log(px[0]))
    fv = pd.Series(r1).rolling(HORIZON).std().shift(-HORIZON).values

    q33 = np.full(len(df), np.nan)
    q66 = np.full(len(df), np.nan)
    for t in range(LOOKBACK, len(df) - HORIZON):
        w = fv[t - LOOKBACK:t - HORIZON]
        w = w[~np.isnan(w)]
        if w.size:
            q33[t], q66[t] = np.percentile(w, [33, 66])

    sl = slice(LOOKBACK, len(df) - HORIZON)
    return {
        "date": df["date"].values[sl], "price": px[sl], "r1": r1[sl],
        "fwd_vol": fv[sl], "q33": q33[sl], "q66": q66[sl],
    }


def make_human(Y, seed=SEED + 1, drift_strength=0.25):
    """Paper-faithful pathologies: 10% uniform noise + gradual linear-prob drift."""
    rng = np.random.default_rng(seed)
    T = len(Y)
    yh = Y.copy()
    flip = rng.random(T) < 0.10
    yh[flip] = rng.integers(0, K, flip.sum())
    drift = np.linspace(0.0, drift_strength, T)
    bump = rng.random(T) < drift
    yh[bump] = np.minimum(2, yh[bump] + 1)
    return yh


def main():
    t0 = time.time()
    print("=" * 78)
    print(" HINN real-world (AAPL) -- ROLLING local-quantile targets")
    print("=" * 78)

    dates, X, Y = preprocess(os.path.join(DATA, "AAPL_prices.csv"))
    T = len(X)
    yh = make_human(Y)
    m = exact_bursty_mask(T, coverage=0.466, seed=SEED + 2)
    print(f"[1] Substrate: {T} days ({dates.iloc[0].date()} -> {dates.iloc[-1].date()})")
    print(f"[2] Targets (rolling local): low={np.sum(Y==0)} med={np.sum(Y==1)} "
          f"high={np.sum(Y==2)} | mask={m.mean()*100:.1f}% | "
          f"human!=data on mask={np.mean(yh[m==1]!=Y[m==1])*100:.1f}%")

    split = int(TRAIN_FRAC * T)
    # leak-free scaling: fit on TRAIN only
    mu, sd = X[:split].mean(0), X[:split].std(0) + 1e-8
    Xz = (X - mu) / sd
    X_tr, y_tr, yh_tr, m_tr = Xz[:split], Y[:split], yh[:split], m[:split]
    X_va, y_va, yh_va, m_va = Xz[split:], Y[split:], yh[split:], m[split:]
    print(f"[3] Split train={split} val={T-split} | "
          f"val class balance={np.bincount(y_va, minlength=3)/len(y_va)}")

    print("\n[4] Training full HINN ...")
    res = train_hinn(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                     n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=8)
    print("[5] data-only ...")
    res_data = ablation_data_only(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                                  n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=4, verbose=False)
    print("[6] human-only ...")
    res_human = ablation_human_only(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                                    n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=4, verbose=False)

    res["history"].to_csv(os.path.join(RESULTS, "history_hinn_rolling.csv"), index=False)
    res_data["history"].to_csv(os.path.join(RESULTS, "history_data_only_rolling.csv"), index=False)
    res_human["history"].to_csv(os.path.join(RESULTS, "history_human_only_rolling.csv"), index=False)
    res["history"][["ep", "lam_mean", "lam_std", "rho_mean", "c_mean",
                    "post_var_mean", "kl"]].to_csv(
        os.path.join(RESULTS, "trust_lambda_trace_rolling.csv"), index=False)
    pd.DataFrame({
        "t": np.arange(res["Lambdas"].shape[1]), "lambda_final": res["Lambdas"][-1],
        "rho_final": res["Rhos"][-1], "c_final": res["Cs"][-1],
        "cov_trace_final": res["Covs"][-1], "m_train": m_tr,
    }).to_csv(os.path.join(RESULTS, "trust_profile_final_rolling.csv"), index=False)

    table = pd.DataFrame({
        "HINN (full)": headline(res),
        "Data-only (lambda=1)": headline(res_data),
        "Human-only (lambda=0)": headline(res_human),
    }).T
    table.to_csv(os.path.join(RESULTS, "headline_metrics_rolling.csv"))

    elapsed = time.time() - t0
    print("\n[7] Comparative baseline statistics (rolling local-quantile targets):\n")
    print(table.round(4).to_string())
    print(f"\nDone in {elapsed:.1f}s. Results -> {RESULTS}")


if __name__ == "__main__":
    main()
