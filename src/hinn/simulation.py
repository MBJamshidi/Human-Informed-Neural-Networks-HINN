"""
================================================================================
 HINN: Human-Informed Neural Network -- Numerical Simulation
================================================================================
 This file implements EXACTLY the formulation of the HINN paper:

   * Section 3 -- Recurrent machine pathway with CORRECTED BPTT
       Eqs. (1)-(2)  forward dynamics
       Eqs. (8)-(9)  pre/post-activation deltas
       Eqs.(10)-(13) parameter gradients
       Eq. (14)      gradient clipping

   * Section 4 -- Bayesian human pathway
       Eqs.(15)-(16) prior + Gaussian likelihood
       Eq. (18)      mean-field variational family
       Eq. (19)      negative ELBO
       Eq. (20)      reparameterised stochastic gradient
       Eq. (21)      closed-form KL
       Eq. (22)      posterior-expected human loss (mask-gated, MC estimate)

   * Section 5 -- HINN composition
       Eq. (24)      time-sensitive trust function lambda(t)
       Eq. (25)      total HINN objective L_HINN
       Algorithm 1   joint training

 Big data: synthetic but realistic multi-asset financial time series
           (GARCH-like volatility + 3-state regime switching). When run on a
           network-enabled machine the loader can be replaced by a Yahoo /
           FRED pull -- the rest of the pipeline is unchanged.

 Human signal: simulated policy-aligned risk classifications with intermittent
              availability (mask m_t), inter-annotator noise, and policy drift.

 Outputs: ~32 scientific figures + metrics tables + final report text.
================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D registration)
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
SEED = 7
np.random.seed(SEED)

OUTDIR = os.path.abspath(os.environ.get("HINN_OUTDIR", os.getcwd()))
FIGDIR = os.path.join(OUTDIR, "figures")
TBLDIR = os.path.join(OUTDIR, "tables")
DATADIR = os.path.join(OUTDIR, "data")
for d in (FIGDIR, TBLDIR, DATADIR):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.titlesize": 0,  # Effectively remove titles
    "axes.titleweight": "normal",
    "axes.labelsize": 18, # Larger labels
    "axes.labelweight": "bold", # Bold labels
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linestyle": "--",
    "legend.frameon": True, "legend.fontsize": 13,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 2.0,
})
PALETTE = {
    "primary": "#1f4e79",   # Deep Blue
    "secondary": "#5b9bd5", # Light Blue
    "neutral": "#000000",   # Black
    "accent": "#7f7f7f",    # Grey for secondary details
}

FIG_INDEX = []

def save_fig(fig, name, caption):
    path = os.path.join(FIGDIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIG_INDEX.append((f"figures/{name}.png", caption))
    print(f"   [fig] {name:40s}  {caption}")


# ===========================================================================
# 1.  BIG-DATA  --  synthetic-but-realistic multi-asset financial time series
# ===========================================================================

def generate_financial_dataset(n_days=1260, n_assets=5, seed=SEED):
    """
    Generates a multi-asset daily series with:
       * a hidden 3-state Markov regime (bull / sideways / bear),
       * cross-asset correlation,
       * GARCH(1,1)-like volatility clustering.
    Drop-in replacement for a real Yahoo/FRED pull on a network-enabled machine.
    """
    rng = np.random.default_rng(seed)
    P = np.array([[0.985, 0.012, 0.003],
                  [0.020, 0.965, 0.015],
                  [0.005, 0.025, 0.970]])
    mu_regime = np.array([ 0.00060, 0.00010, -0.00080])
    sg_regime = np.array([ 0.0080,  0.0120,   0.0240])
    regimes = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regimes[t] = rng.choice(3, p=P[regimes[t-1]])

    A = rng.normal(size=(n_assets, n_assets))
    Sigma = (A @ A.T) / n_assets + 0.4 * np.eye(n_assets)
    L = np.linalg.cholesky(Sigma)

    # GARCH(1,1) on the *standardised innovations*; the regime multiplier
    # rescales the realised vol but does NOT enter the recursion (so sigma2
    # cannot run away in the bear regime).  We also clip per-step returns to
    # +/- 15% to mimic real-world circuit-breaker behaviour and keep prices
    # strictly positive when we exponentiate log-returns.
    omega, alpha, beta = 1e-6, 0.05, 0.92
    sigma2 = np.full(n_assets, 1e-4)
    log_returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        eps = L @ rng.normal(size=n_assets)
        sig = np.sqrt(sigma2) * (sg_regime[regimes[t]] / sg_regime.mean())
        r_t = mu_regime[regimes[t]] + sig * eps
        r_t = np.clip(r_t, -0.15, 0.15)
        log_returns[t] = r_t
        sigma2 = omega + alpha * (r_t**2) + beta * sigma2
    returns = log_returns

    df = pd.DataFrame(returns, columns=[f"asset_{i+1}" for i in range(n_assets)])
    prices = np.exp(df.cumsum()) * 100.0          # use log-returns -> always > 0
    prices.columns = [f"price_{i+1}" for i in range(n_assets)]

    px = prices["price_1"].values
    ma10  = pd.Series(px).rolling(10).mean().bfill().values
    ma50  = pd.Series(px).rolling(50).mean().bfill().values
    rsi   = _rsi(px, 14)
    vol20 = pd.Series(returns[:, 0]).rolling(20).std().bfill().values
    macd  = ma10 - pd.Series(px).rolling(26).mean().bfill().values

    big = pd.DataFrame({
        "ret_idx": returns[:, 0], "ret_a2": returns[:, 1],
        "ret_a3":  returns[:, 2], "ret_a4": returns[:, 3], "ret_a5": returns[:, 4],
        "ma10":   (ma10 - px) / px, "ma50": (ma50 - px) / px,
        "rsi":    (rsi - 50.0) / 50.0,
        "vol20":   vol20,
        "macd":    macd / px,
        "regime":  regimes,
    })
    return big, prices, regimes


def _rsi(px, n=14):
    px = np.asarray(px, dtype=float)
    d = np.diff(px, prepend=px[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    rs_up = pd.Series(up).rolling(n).mean().bfill().values
    rs_dn = pd.Series(dn).rolling(n).mean().bfill().values
    rs = rs_up / np.maximum(rs_dn, 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


# ===========================================================================
# 2.  TARGETS:  data label y_t  and  human label y^human_t  + mask m_t
# ===========================================================================

K = 3   # number of risk classes  (matches "k" in the paper)

def make_labels(big, prices, horizon=5, p_human=0.40,
                drift_strength=0.25, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    px = prices["price_1"].values
    fwd_vol = pd.Series(np.diff(np.log(px), prepend=np.log(px[0]))) \
                .rolling(horizon).std().shift(-horizon).bfill().ffill().values
    q1, q2 = np.quantile(fwd_vol, [0.33, 0.66])
    y_data = np.where(fwd_vol < q1, 0, np.where(fwd_vol < q2, 1, 2))

    T = len(y_data)
    drift = np.linspace(0.0, drift_strength, T)
    y_human = y_data.copy()
    flip = rng.random(T) < 0.10
    y_human[flip] = rng.integers(0, K, flip.sum())
    bump = rng.random(T) < drift
    y_human[bump] = np.minimum(2, y_human[bump] + 1)

    base = rng.random(T) < p_human
    for start in rng.integers(0, T-15, size=15):
        base[start:start+10] = True
    m = base.astype(int)
    return y_data, y_human, m, fwd_vol


# ===========================================================================
# 3.  THE HINN MODEL  --  pure NumPy, follows paper notation exactly
# ===========================================================================

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def one_hot(y, K):
    out = np.zeros((len(y), K))
    out[np.arange(len(y)), y] = 1.0
    return out

def tanh_grad(a):
    return 1.0 - np.tanh(a)**2


# ----- 3.1  RNN  --  Section 3 of the paper -----------------------------------
class RNNMachine:
    """
    Forward:  a_t = W_xh x_t + W_hh h_{t-1} + b_h
              h_t = tanh(a_t)
              z_t = W_hy h_t + b_y
              y_hat_t = softmax(z_t)

    BPTT uses the CORRECTED pre-activation delta:
         delta_t^h = W_hy^T g_t + W_hh^T delta_{t+1}^a    (Eq. 8)
         delta_t^a = delta_t^h * tanh'(a_t)               (Eq. 9)
         and parameter gradients use delta_t^a exactly once (Eqs. 10-13).
    """

    def __init__(self, n, m, k, seed=SEED):
        rng = np.random.default_rng(seed + 2)
        s = 1.0 / np.sqrt(m)
        self.W_xh = rng.normal(0, s, size=(m, n))
        self.W_hh = rng.normal(0, s, size=(m, m)) * 0.5
        self.b_h  = np.zeros(m)
        self.W_hy = rng.normal(0, s, size=(k, m))
        self.b_y  = np.zeros(k)
        self.n, self.m, self.k = n, m, k

    def forward(self, X):
        T = X.shape[0]
        H = np.zeros((T+1, self.m))    # H[t+1] is h_t in the paper, H[0]=h_0=0
        A = np.zeros((T,  self.m))
        Z = np.zeros((T,  self.k))
        Y = np.zeros((T,  self.k))
        for t in range(T):
            A[t]   = self.W_xh @ X[t] + self.W_hh @ H[t] + self.b_h
            H[t+1] = np.tanh(A[t])
            Z[t]   = self.W_hy @ H[t+1] + self.b_y
            Y[t]   = softmax(Z[t])
        return {"H": H, "A": A, "Z": Z, "Y": Y}

    def bptt(self, X, cache, dZ):
        H, A = cache["H"], cache["A"]
        T = X.shape[0]
        gW_xh = np.zeros_like(self.W_xh)
        gW_hh = np.zeros_like(self.W_hh)
        gb_h  = np.zeros_like(self.b_h)
        gW_hy = np.zeros_like(self.W_hy)
        gb_y  = np.zeros_like(self.b_y)

        delta_a_next = np.zeros(self.m)
        for t in reversed(range(T)):
            g_t       = dZ[t]                                       # eq. 7
            delta_h_t = self.W_hy.T @ g_t + self.W_hh.T @ delta_a_next
            delta_a_t = delta_h_t * tanh_grad(A[t])                 # eq. 9
            gW_xh += np.outer(delta_a_t, X[t])                       # eq.10
            gW_hh += np.outer(delta_a_t, H[t])                       # eq.11
            gb_h  += delta_a_t                                       # eq.12
            gW_hy += np.outer(g_t, H[t+1])                           # eq.13
            gb_y  += g_t
            delta_a_next = delta_a_t
        return {"W_xh": gW_xh, "W_hh": gW_hh, "b_h": gb_h,
                "W_hy": gW_hy, "b_y": gb_y}

    def step(self, grads, lr, tau=5.0):
        # Eq. (14) gradient clipping
        flat = np.concatenate([g.ravel() for g in grads.values()])
        gn = np.linalg.norm(flat)
        scale = tau / max(gn, 1e-9) if gn > tau else 1.0
        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * scale * g)
        return gn, scale


# ----- 3.2  Bayesian Neural Network  --  Section 4 of the paper --------------
class BNNHuman:
    """
    Single-hidden-layer mean-field Bayesian network mapping y_hat (softmax
    probabilities, dim k) to logits over human classes (dim k).

       Prior:        p(theta) = N(0, sig_p^2 I)                     (Eq. 15)
       Posterior q:  N(mu, diag(sigma^2))                            (Eq. 18)
       Sampling:     theta = mu + sigma * eps,   eps ~ N(0,I)
       KL closed:    Eq. 21
    """

    def __init__(self, k, hidden=8, sig_p=1.0, seed=SEED):
        rng = np.random.default_rng(seed + 3)
        self.k = k; self.hidden = hidden; self.sig_p = sig_p
        n_params = k*hidden + hidden + hidden*k + k
        self.n_params = n_params
        self.mu  = rng.normal(0, 0.1, size=n_params)
        # rho parameterises sigma via softplus to keep it positive
        self.rho = np.full(n_params, -3.0)

    @property
    def sigma(self):
        return np.log1p(np.exp(self.rho))

    def _unpack(self, theta):
        k, h = self.k, self.hidden
        i = 0
        W1 = theta[i:i+k*h].reshape(h, k);   i += k*h
        b1 = theta[i:i+h];                   i += h
        W2 = theta[i:i+h*k].reshape(k, h);   i += h*k
        b2 = theta[i:i+k]
        return W1, b1, W2, b2

    def forward_sample(self, y_hat, theta):
        W1, b1, W2, b2 = self._unpack(theta)
        h = np.tanh(y_hat @ W1.T + b1)
        z = h @ W2.T + b2
        return z, h

    def kl_to_prior(self):
        s2 = self.sigma**2; sp2 = self.sig_p**2
        return 0.5 * np.sum((self.mu**2 + s2)/sp2 - 1.0 - np.log(s2/sp2))

    def kl_grads(self):
        s = self.sigma; sp2 = self.sig_p**2
        d_mu  = self.mu / sp2
        d_sig = s / sp2 - 1.0 / s
        d_rho = d_sig * (1.0 / (1.0 + np.exp(-self.rho)))   # softplus chain
        return d_mu, d_rho


# ----- 3.3  Trust function  --  Eq. (24) -------------------------------------
def trust_lambda(t_array, m_array, cov_trace,
                 lambda_min=0.2, alpha=2.5, gamma=0.05, kappa=8.0):
    """
    lambda(t) = lambda_min + (1-lambda_min) exp(-alpha * rho_t * c_t)
       rho_t = exp(-gamma * dt_h)        dt_h = steps since last human signal
       c_t   = exp(-kappa * trace Cov_q)
    """
    T = len(t_array)
    last_h = -1; dt_h = np.full(T, np.inf)
    for t in range(T):
        if m_array[t] == 1:
            last_h = t
        dt_h[t] = (t - last_h) if last_h >= 0 else 1e6
    rho = np.exp(-gamma * dt_h)
    c   = np.exp(-kappa * np.clip(cov_trace, 0, None))
    lam = lambda_min + (1 - lambda_min) * np.exp(-alpha * rho * c)
    return lam, rho, c, dt_h


# ===========================================================================
# 4.  TRAINING LOOP -- Algorithm 1 of the paper, faithful implementation
# ===========================================================================

def expected_calibration_error(P, y, n_bins=10):
    conf = P.max(axis=1); pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.sum() == 0: continue
        ece += (m.sum()/len(y)) * abs(correct[m].mean() - conf[m].mean())
    return ece


def train_hinn(X_tr, y_tr, yh_tr, m_tr,
               X_va, y_va, yh_va, m_va,
               n_epochs=40, lr_m=5e-2, lr_phi=5e-2,
               S_mc=8, beta=1e-3, tau_clip=5.0,
               trust_kw=None, seed=SEED, verbose=True):
    if trust_kw is None:
        trust_kw = dict(lambda_min=0.5, alpha=1.5, gamma=0.05, kappa=2.0)

    n_in  = X_tr.shape[1]; n_hid = 32
    rnn = RNNMachine(n_in, n_hid, K, seed=seed)
    bnn = BNNHuman(K, hidden=8, sig_p=1.0, seed=seed)
    rng = np.random.default_rng(seed + 4)

    history = {k: [] for k in [
        "ep","L_total","L_data","L_human","L_kl",
        "acc_tr","acc_va","f1_tr","f1_va","acc_h_tr","f1_h_va",
        "lam_mean","lam_std","rho_mean","c_mean",
        "kl","post_var_mean","grad_norm","grad_scale",
        "ece_tr","brier_tr","auc_tr_ovo"]}
    Lambdas, Rhos, Cs, Covs, Yhat_history = [], [], [], [], []

    for ep in range(1, n_epochs+1):
        cache = rnn.forward(X_tr); Yhat = cache["Y"]
        T = X_tr.shape[0]

        # MC samples for BNN
        eps = rng.normal(size=(S_mc, bnn.n_params))
        thetas = bnn.mu + bnn.sigma * eps
        bnn_logits = np.zeros((S_mc, T, K))
        bnn_probs  = np.zeros((S_mc, T, K))
        for s in range(S_mc):
            zb, _ = bnn.forward_sample(Yhat, thetas[s])
            bnn_logits[s] = zb
            bnn_probs[s]  = softmax(zb, axis=-1)
        bnn_var   = bnn_probs.var(axis=0)
        cov_trace = bnn_var.sum(axis=-1)

        # Trust function (Eq. 24)
        t_arr = np.arange(T)
        lam, rho, conf, dt_h = trust_lambda(t_arr, m_tr, cov_trace, **trust_kw)
        Lambdas.append(lam.copy()); Rhos.append(rho.copy())
        Cs.append(conf.copy());     Covs.append(cov_trace.copy())

        # Loss components
        Y_oh   = one_hot(y_tr, K)
        Yh_oh  = one_hot(yh_tr, K)
        eps_log = 1e-9
        L_data_t = -np.sum(Y_oh * np.log(Yhat + eps_log), axis=1)
        L_data   = float(np.sum(lam * L_data_t)) / T

        L_human_per_s = np.zeros((S_mc, T))
        for s in range(S_mc):
            L_human_per_s[s] = -np.sum(Yh_oh * np.log(bnn_probs[s] + eps_log), axis=1)
        L_human_t = L_human_per_s.mean(axis=0)
        denom = max(m_tr.sum(), 1)
        L_human = float(np.sum((1 - lam) * m_tr * L_human_t)) / denom

        kl = bnn.kl_to_prior()
        L_kl = beta * float(kl)
        L_total = L_data + L_human + L_kl

        # ---- Backward through RNN ----
        # dL_data/dz_t for softmax+CE
        dZ_data = (Yhat - Y_oh) / T
        # dL_human flowing back into y_hat via the BNN (averaged over MC)
        dYhat_human = np.zeros_like(Yhat)
        for s in range(S_mc):
            W1, b1, W2, b2 = bnn._unpack(thetas[s])
            pb = bnn_probs[s]
            d_zb = (pb - Yh_oh) * m_tr[:, None] / denom
            h_act = np.tanh(Yhat @ W1.T + b1)
            d_h   = d_zb @ W2
            d_pre = d_h * (1 - h_act**2)
            d_yhat = d_pre @ W1
            dYhat_human += d_yhat / S_mc

        # chain through softmax to get dZ_human
        dZ_human = np.zeros_like(Yhat)
        for t in range(T):
            p = Yhat[t]
            J = np.diag(p) - np.outer(p, p)
            dZ_human[t] = J @ dYhat_human[t]

        dZ_total = lam[:, None] * dZ_data + (1 - lam)[:, None] * dZ_human
        grads_m = rnn.bptt(X_tr, cache, dZ_total)
        gn, scale = rnn.step(grads_m, lr_m, tau=tau_clip)

        # ---- Backward through BNN variational params ----
        d_mu_total  = np.zeros_like(bnn.mu)
        d_rho_total = np.zeros_like(bnn.rho)
        sigmoid_rho = 1.0 / (1.0 + np.exp(-bnn.rho))
        for s in range(S_mc):
            W1, b1, W2, b2 = bnn._unpack(thetas[s])
            pb = bnn_probs[s]
            d_zb = (pb - Yh_oh) * (1 - lam)[:, None] * m_tr[:, None] / denom
            h_act = np.tanh(Yhat @ W1.T + b1)
            gW2 = d_zb.T @ h_act
            gb2 = d_zb.sum(axis=0)
            d_h   = d_zb @ W2
            d_pre = d_h * (1 - h_act**2)
            gW1 = d_pre.T @ Yhat
            gb1 = d_pre.sum(axis=0)
            grad_theta = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
            d_mu_total  += grad_theta / S_mc
            d_rho_total += grad_theta * eps[s] * sigmoid_rho / S_mc

        d_mu_kl, d_rho_kl = bnn.kl_grads()
        d_mu_total  += beta * d_mu_kl
        d_rho_total += beta * d_rho_kl

        flat = np.concatenate([d_mu_total, d_rho_total])
        gphi = np.linalg.norm(flat)
        sc_phi = tau_clip / max(gphi, 1e-9) if gphi > tau_clip else 1.0
        bnn.mu  -= lr_phi * sc_phi * d_mu_total
        bnn.rho -= lr_phi * sc_phi * d_rho_total

        # ---- Diagnostics ----
        y_pred_tr = Yhat.argmax(axis=1)
        acc_tr = accuracy_score(y_tr, y_pred_tr)
        f1_tr  = f1_score(y_tr, y_pred_tr, average="macro")
        cache_va = rnn.forward(X_va); Yhat_va = cache_va["Y"]
        bnn_probs_va = np.zeros((S_mc, X_va.shape[0], K))
        for s in range(S_mc):
            zbv, _ = bnn.forward_sample(Yhat_va, thetas[s])
            bnn_probs_va[s] = softmax(zbv, axis=-1)
        bnn_mean_va = bnn_probs_va.mean(axis=0)
        y_pred_va  = Yhat_va.argmax(axis=1)
        yh_pred_va = bnn_mean_va.argmax(axis=1)
        acc_va = accuracy_score(y_va, y_pred_va)
        f1_va  = f1_score(y_va, y_pred_va, average="macro")
        f1_h_va = f1_score(yh_va, yh_pred_va, average="macro")
        bnn_mean_tr = bnn_probs.mean(axis=0)
        acc_h_tr = accuracy_score(yh_tr, bnn_mean_tr.argmax(axis=1))
        try:
            ece = expected_calibration_error(Yhat, y_tr, n_bins=10)
            brier = np.mean(np.sum((Yhat - Y_oh)**2, axis=1)) / 2.0
            auc_ovo = roc_auc_score(y_tr, Yhat, multi_class="ovo")
        except Exception:
            ece, brier, auc_ovo = np.nan, np.nan, np.nan

        history["ep"].append(ep)
        history["L_total"].append(L_total); history["L_data"].append(L_data)
        history["L_human"].append(L_human); history["L_kl"].append(L_kl)
        history["acc_tr"].append(acc_tr);   history["acc_va"].append(acc_va)
        history["f1_tr"].append(f1_tr);     history["f1_va"].append(f1_va)
        history["acc_h_tr"].append(acc_h_tr); history["f1_h_va"].append(f1_h_va)
        history["lam_mean"].append(lam.mean()); history["lam_std"].append(lam.std())
        history["rho_mean"].append(rho.mean()); history["c_mean"].append(conf.mean())
        history["kl"].append(float(kl)); history["post_var_mean"].append(float(bnn_var.mean()))
        history["grad_norm"].append(gn);  history["grad_scale"].append(scale)
        history["ece_tr"].append(ece); history["brier_tr"].append(brier)
        history["auc_tr_ovo"].append(auc_ovo)
        Yhat_history.append(Yhat.copy())

        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  ep {ep:03d} | L={L_total:6.4f}  acc_tr={acc_tr:.3f} "
                  f"acc_va={acc_va:.3f} f1_va={f1_va:.3f}  "
                  f"lam={lam.mean():.3f}±{lam.std():.3f}  KL={kl:.1f}")

    return {
        "rnn": rnn, "bnn": bnn,
        "history": pd.DataFrame(history),
        "Lambdas": np.array(Lambdas), "Rhos": np.array(Rhos),
        "Cs": np.array(Cs), "Covs": np.array(Covs),
        "Yhat_history": np.array(Yhat_history),
    }


# ===========================================================================
# 5.  ABLATIONS
# ===========================================================================

def ablation_data_only(*a, **kw):
    return train_hinn(*a, trust_kw=dict(lambda_min=0.999, alpha=0.001,
                                        gamma=1.0, kappa=1.0), **kw)

def ablation_human_only(*a, **kw):
    return train_hinn(*a, trust_kw=dict(lambda_min=0.001, alpha=8.0,
                                        gamma=0.001, kappa=0.0), **kw)


# ===========================================================================
# 6.  PLOTTING -- the 30+ figures
# ===========================================================================

def plot_dataset_overview(big, prices, regimes):
    # 1. Price series
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices.index, prices["price_1"], color=PALETTE["primary"], lw=2.0, label="Normalized Price Index")
    ax.fill_between(prices.index, prices["price_1"], color=PALETTE["primary"], alpha=0.1)
    ax.set_ylabel("Price (Normalized)")
    ax.set_xlabel("Trading Day ($t$)")
    ax.legend()
    save_fig(fig, "data_price_series", "Synthetic financial 'big-data' index series (price_1)")

    # 2. Regimes
    fig, ax = plt.subplots(figsize=(10, 3.5))
    cmap = [PALETTE["secondary"], "#e1e1e1", PALETTE["primary"]]
    labels = ["Bullish", "Sideways", "Bearish"]
    for r in range(3):
        ax.fill_between(prices.index, 0, 1, where=(regimes == r), step="mid",
                        color=cmap[r], alpha=0.4, label=labels[r])
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_xlabel("Trading Day ($t$)")
    save_fig(fig, "data_regime_process", "Hidden 3-state regime process driving volatility")

    # 3. Features
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices.index, big["vol20"], color=PALETTE["primary"], lw=1.5, label="20-day Realized Volatility")
    ax.plot(prices.index, big["rsi"]*0.05, color=PALETTE["secondary"], ls="--", lw=1.5, label="RSI (Scaled)")
    ax.fill_between(prices.index, big["vol20"], color=PALETTE["primary"], alpha=0.1)
    ax.legend()
    ax.set_xlabel("Trading Day ($t$)")
    ax.set_ylabel("Indicator Value")
    save_fig(fig, "data_engineered_features", "Engineered indicators feeding the RNN input $x_t$")

def plot_label_structure(y_data, y_human, m, fwd_vol):
    # 1. Forward Vol
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fwd_vol, color=PALETTE["primary"], lw=1.5)
    ax.fill_between(range(len(fwd_vol)), fwd_vol, color=PALETTE["primary"], alpha=0.1)
    ax.set_ylabel(r"Forward Volatility $\sigma_{t+h}$")
    ax.set_xlabel("Trading Day ($t$)")
    save_fig(fig, "label_fwd_vol", "Forward realised volatility used to derive $y_t$")

    # 2. Data labels
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(y_data, color=PALETTE["primary"], lw=1.0, label="Data Target ($y_t$)", alpha=0.8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Med", "High"])
    ax.set_xlabel("Trading Day ($t$)")
    ax.set_ylabel("Risk Class")
    ax.legend()
    save_fig(fig, "label_data_targets", "Data labels $y_t$ (Low / Medium / High Risk)")

    # 3. Human labels
    fig, ax = plt.subplots(figsize=(10, 3.5))
    masked = np.where(m == 1, y_human, np.nan)
    ax.scatter(range(len(masked)), masked, color=PALETTE["secondary"], s=10, label="Human Label ($y^h_t$)", alpha=0.7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Med", "High"])
    ax.set_xlabel("Trading Day ($t$)")
    ax.set_ylabel("Risk Class")
    ax.legend()
    save_fig(fig, "label_human_targets", "Human labels $y^h_t$ (Sparse, Drifting Policy)")

    # 4. Mask
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.fill_between(np.arange(len(m)), 0, m, color=PALETTE["primary"], step="mid", alpha=0.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["None", "Available"])
    ax.set_xlabel("Trading Day ($t$)")
    save_fig(fig, "label_mask", f"Availability mask $m_t$ (Coverage = {m.mean()*100:.1f}%)")

def plot_training_curves(hist):
    h = hist
    # 1. Loss
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h.ep, h.L_total, color=PALETTE["neutral"], label=r"$\mathcal{L}_{\mathrm{HINN}}$", lw=2.5)
    ax.plot(h.ep, h.L_data,  color=PALETTE["primary"], label=r"$\mathcal{L}_{\mathrm{data}}$", ls="--")
    ax.plot(h.ep, h.L_human, color=PALETTE["secondary"], label=r"$\mathcal{L}_{\mathrm{human}}$", ls="-.")
    ax.plot(h.ep, h.L_kl,    color=PALETTE["accent"], label=r"$\beta\,\mathrm{KL}$", alpha=0.6)
    ax.fill_between(h.ep, h.L_total, color=PALETTE["neutral"], alpha=0.05)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("Loss Magnitude"); ax.legend()
    save_fig(fig, "train_loss_decomposition", "HINN total loss decomposition")

    # 2. Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h.ep, h.acc_tr, label="Train (Data)", color=PALETTE["primary"], marker="o", ms=4, markevery=5)
    ax.plot(h.ep, h.acc_va, label="Val (Data)", color=PALETTE["secondary"], ls="--", marker="s", ms=4, markevery=5)
    ax.plot(h.ep, h.acc_h_tr, label="Train (Human)", color=PALETTE["neutral"], ls=":", alpha=0.7)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("Classification Accuracy"); ax.legend()
    save_fig(fig, "train_accuracy_evolution", "Accuracy across pathways")

    # 3. F1
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h.ep, h.f1_tr, label="Train (Data)", color=PALETTE["primary"], marker="o", ms=4, markevery=5)
    ax.plot(h.ep, h.f1_va, label="Val (Data)", color=PALETTE["secondary"], ls="--", marker="s", ms=4, markevery=5)
    ax.plot(h.ep, h.f1_h_va, label="Val (Human)", color=PALETTE["neutral"], ls=":", alpha=0.7)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("Macro-F1 Score"); ax.legend()
    save_fig(fig, "train_f1_evolution", "Macro-F1 across pathways")

    # 4. KL
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h.ep, h.kl, color=PALETTE["primary"], label=r"$\mathrm{KL}(q \parallel p)$")
    ax.fill_between(h.ep, h.kl, color=PALETTE["primary"], alpha=0.1)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("KL Divergence (nats)"); ax.legend()
    save_fig(fig, "train_kl_divergence", "Variational KL divergence to prior")


def plot_trust_dynamics(res, m_tr):
    Lambdas = res["Lambdas"]
    h = res["history"]
    # 1. Mean Trust
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h.ep, h.lam_mean, color=PALETTE["primary"], lw=2.5, label=r"$\overline{\lambda(t)}$")
    ax.fill_between(h.ep, h.lam_mean - h.lam_std, h.lam_mean + h.lam_std,
                     color=PALETTE["primary"], alpha=0.15)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel(r"Expected Trust $\mathbb{E}[\lambda(t)]$")
    ax.legend()
    save_fig(fig, "trust_mean_evolution", r"Mean trust $\lambda(t)$ across epochs")

    # 2. Drivers
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h.ep, h.rho_mean, color=PALETTE["primary"], label=r"Recency $\overline{\rho_t}$", marker="o", ms=4, markevery=5)
    ax.plot(h.ep, h.c_mean,   color=PALETTE["secondary"], label=r"Confidence $\overline{c_t}$", ls="--", marker="s", ms=4, markevery=5)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("Driver Value [0, 1]"); ax.legend()
    save_fig(fig, "trust_drivers_evolution", "Recency and confidence drivers over training")

    # 3. Final Lambda
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(Lambdas[-1], color=PALETTE["primary"], label=r"Final $\lambda(t)$", lw=1.5)
    ax.fill_between(range(len(Lambdas[-1])), Lambdas[-1], color=PALETTE["primary"], alpha=0.1)
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel(r"Trust Coefficient $\lambda(t)$")
    ax.legend()
    save_fig(fig, "trust_final_profile", "Final-epoch trust profile vs availability mask")

    # 4. Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Lambdas, aspect="auto", cmap="Blues", origin="lower",
                   extent=[0, Lambdas.shape[1], 1, Lambdas.shape[0]])
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel("Epoch ($e$)")
    fig.colorbar(im, ax=ax, label=r"Trust $\lambda(t)$")
    save_fig(fig, "trust_heatmap_evolution", "Evolution of trust distribution over time and epochs")

def plot_uncertainty(res):
    Covs = res["Covs"]
    h = res["history"]
    # 1. Mean Variance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h.ep, h.post_var_mean, color=PALETTE["primary"], lw=2.5)
    ax.fill_between(h.ep, h.post_var_mean, color=PALETTE["primary"], alpha=0.1)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel(r"Expected Trace $\overline{\mathrm{Tr}(\Sigma_q)}$")
    save_fig(fig, "uncertainty_mean_trace", "Mean BNN posterior predictive variance evolution")

    # 2. Covariance Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Covs, aspect="auto", cmap="Blues", origin="lower",
                   extent=[0, Covs.shape[1], 1, Covs.shape[0]])
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel("Epoch ($e$)")
    fig.colorbar(im, ax=ax, label=r"Trace $\Sigma_q$")
    save_fig(fig, "uncertainty_trace_heatmap", "Posterior covariance trace evolution")

def plot_predictions(rnn, bnn, X, y, yh, m, S_mc=20, seed=SEED, tag="train"):
    rng = np.random.default_rng(seed + 99)
    cache = rnn.forward(X); Yhat = cache["Y"]
    eps = rng.normal(size=(S_mc, bnn.n_params))
    thetas = bnn.mu + bnn.sigma * eps
    bnn_probs = np.zeros((S_mc, X.shape[0], K))
    for s in range(S_mc):
        zb, _ = bnn.forward_sample(Yhat, thetas[s])
        bnn_probs[s] = softmax(zb, axis=-1)
    bnn_mean = bnn_probs.mean(axis=0); bnn_std = bnn_probs.std(axis=0)

    # 1. Probability High Risk
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(Yhat[:, 2], color=PALETTE["primary"], lw=1.5, label=r"$P_{\mathrm{m}}(\text{High Risk})$")
    ax.plot(bnn_mean[:, 2], color=PALETTE["secondary"], lw=1.5, ls="--", label=r"$\mathbb{E}_q[P_{\mathrm{h}}]$")
    ax.fill_between(np.arange(len(bnn_mean)),
                    bnn_mean[:, 2] - 2*bnn_std[:, 2],
                    bnn_mean[:, 2] + 2*bnn_std[:, 2],
                    color=PALETTE["secondary"], alpha=0.15, label=r"$\pm 2\sigma$ Credible Interval")
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel("Probability"); ax.legend()
    save_fig(fig, f"pred_high_risk_prob_{tag}", f"High-risk class probability trajectory ({tag})")

    # 2. Discrete Class Predictions
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y, color=PALETTE["neutral"], lw=1.0, alpha=0.3, label="Ground Truth ($y_t$)")
    masked = np.where(m == 1, yh, np.nan)
    ax.scatter(range(len(masked)), masked, s=15, color=PALETTE["secondary"], alpha=0.5, label="Human Signal ($y^h_t$)")
    ax.plot(Yhat.argmax(axis=1), color=PALETTE["primary"], lw=1.2, label=r"Machine Path ($\hat{y}^m_t$)")
    ax.plot(bnn_mean.argmax(axis=1), color=PALETTE["neutral"], lw=1.2, ls="--", label=r"Human Path ($\hat{y}^h_t$)")
    ax.set_xlabel("Time Step ($t$)"); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Low", "Med", "High"])
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, f"pred_class_comparison_{tag}", f"Predicted vs actual risk classes ({tag})")
    return Yhat, bnn_mean, bnn_std


def plot_confusion(y_true, y_pred, name_base, tag_label="Machine Pathway"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = ["Low Risk", "Medium Risk", "High Risk"]

    # 1. Absolute Counts
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=True,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    save_fig(fig, f"{name_base}_counts", f"Confusion matrix ({tag_label}) — absolute counts")

    # 2. Normalized
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    save_fig(fig, f"{name_base}_normalized", f"Confusion matrix ({tag_label}) — normalized")

def plot_calibration(Yhat, y, name_base, tag_label="Machine Pathway"):
    Yhat = np.nan_to_num(Yhat, nan=1.0/K)
    conf = Yhat.max(axis=1); pred = Yhat.argmax(axis=1)
    correct = (pred == y).astype(int)
    bins = np.linspace(0, 1, 11)
    bin_acc, bin_conf = [], []
    for i in range(10):
        msk = (conf >= bins[i]) & (conf < bins[i+1])
        if msk.sum() < 5: continue
        bin_acc.append(correct[msk].mean()); bin_conf.append(conf[msk].mean())

    # 1. Reliability Diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], ls="--", alpha=0.5, label="Perfect Calibration")
    if bin_conf:
        ax.plot(bin_conf, bin_acc, "s-", color=PALETTE["primary"], lw=2, markersize=8, label=tag_label)
    ax.set_xlabel("Confidence (Max Probability)"); ax.set_ylabel("Empirical Accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    save_fig(fig, f"calibration_reliability_{name_base}", f"Reliability diagram ({tag_label})")

    # 2. Confidence Histogram
    fig, ax = plt.subplots(figsize=(7, 6))
    conf_finite = conf[np.isfinite(conf)]
    if len(conf_finite) > 0:
        ax.hist(conf_finite, bins=15, range=(0, 1), color=PALETTE["secondary"], edgecolor=PALETTE["neutral"], alpha=0.7)
    ax.set_xlabel("Confidence (Max Probability)"); ax.set_ylabel("Sample Count")
    save_fig(fig, f"calibration_histogram_{name_base}", f"Confidence histogram ({tag_label})")

def plot_3d_loss_surface(rnn, X, y):
    base_xh = rnn.W_xh[0, 0]; base_hh = rnn.W_hh[0, 0]
    grid = np.linspace(-0.6, 0.6, 22)
    L = np.zeros((len(grid), len(grid)))
    for i, dx in enumerate(grid):
        for j, dh in enumerate(grid):
            rnn.W_xh[0, 0] = base_xh + dx
            rnn.W_hh[0, 0] = base_hh + dh
            cache = rnn.forward(X)
            Y = cache["Y"]
            Y_oh = one_hot(y, K)
            L[i, j] = -np.sum(Y_oh * np.log(Y + 1e-9), axis=1).mean()
    rnn.W_xh[0, 0] = base_xh; rnn.W_hh[0, 0] = base_hh
    XX, YY = np.meshgrid(grid, grid, indexing="ij")

    # 1. 3D Surface
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, L, cmap="Blues", edgecolor="none", alpha=0.8)
    ax.set_xlabel(r"$\Delta W_{xh}[0,0]$"); ax.set_ylabel(r"$\Delta W_{hh}[0,0]$")
    ax.set_zlabel(r"Loss $\mathcal{L}_{\mathrm{data}}$")
    save_fig(fig, "viz_loss_surface_3d", "3-D slice of data-loss landscape")

    # 2. Contour Map
    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contourf(XX, YY, L, levels=20, cmap="Blues")
    ax.contour(XX, YY, L, levels=10, colors="white", linewidths=0.5, alpha=0.5)
    ax.plot(0, 0, marker="*", color=PALETTE["neutral"], markersize=15, label=r"Optimal $\Theta$")
    ax.set_xlabel(r"$\Delta W_{xh}[0,0]$"); ax.set_ylabel(r"$\Delta W_{hh}[0,0]$")
    ax.legend(); fig.colorbar(cs, label="Cross-Entropy Loss")
    save_fig(fig, "viz_loss_contour", "Contour map of the loss landscape")

def plot_3d_trust_surface():
    rho = np.linspace(0, 1, 50); c = np.linspace(0, 1, 50)
    R, C = np.meshgrid(rho, c, indexing="ij")
    lam_min, alpha = 0.5, 1.5
    L = lam_min + (1 - lam_min) * np.exp(-alpha * R * C)

    # 1. 3D Surface
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(R, C, L, cmap="Blues", edgecolor="none", alpha=0.8)
    ax.set_xlabel(r"Recency $\rho_t$"); ax.set_ylabel(r"Confidence $c_t$")
    ax.set_zlabel(r"Trust $\lambda(t)$")
    save_fig(fig, "viz_trust_surface_3d", "Analytical 3-D shape of the trust function")

    # 2. Contour Map
    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contourf(R, C, L, levels=14, cmap="Blues")
    ax.contour(R, C, L, levels=8, colors="white", linewidths=0.5, alpha=0.5)
    ax.set_xlabel(r"Recency $\rho_t$"); ax.set_ylabel(r"Confidence $c_t$")
    fig.colorbar(cs, label=r"Trust $\lambda(t)$")
    save_fig(fig, "viz_trust_contour", "Contour map of the trust function surface")


def plot_3d_yhat_trajectory(Yhat_history):
    Y_modes = np.zeros((Yhat_history.shape[0], K))
    for e in range(Yhat_history.shape[0]):
        am = Yhat_history[e].argmax(axis=1)
        for k in range(K):
            Y_modes[e, k] = (am == k).mean()
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    cm = plt.cm.Blues(np.linspace(0.3, 1, len(Y_modes)))
    for i in range(len(Y_modes) - 1):
        ax.plot(Y_modes[i:i+2, 0], Y_modes[i:i+2, 1], Y_modes[i:i+2, 2],
                "-", color=cm[i], lw=2.5)
    ax.scatter(Y_modes[0, 0], Y_modes[0, 1], Y_modes[0, 2],
               s=120, c=PALETTE["neutral"], marker="o", label="Epoch 1", depthshade=False)
    ax.scatter(Y_modes[-1, 0], Y_modes[-1, 1], Y_modes[-1, 2],
               s=180, c=PALETTE["primary"], marker="*", label=f"Epoch {len(Y_modes)}",
               depthshade=False)
    ax.set_xlabel("Low Risk Mass"); ax.set_ylabel("Med Risk Mass")
    ax.set_zlabel("High Risk Mass")
    ax.legend(loc="upper left"); save_fig(fig, "viz_yhat_trajectory_3d", "3-D trajectory of class-mode mass through epochs")

def plot_param_distributions(bnn):
    # 1. Means
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bnn.mu, bins=25, color=PALETTE["primary"], alpha=0.7, edgecolor=PALETTE["neutral"])
    ax.set_xlabel(r"Variational Mean $\mu$"); ax.set_ylabel("Frequency")
    save_fig(fig, "viz_bnn_mu_dist", r"Histogram of learned variational means $\mu$")

    # 2. Sigmas
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bnn.sigma, bins=25, color=PALETTE["secondary"], alpha=0.7, edgecolor=PALETTE["neutral"])
    ax.set_xlabel(r"Variational Scale $\sigma$"); ax.set_ylabel("Frequency")
    save_fig(fig, "viz_bnn_sigma_dist", r"Histogram of learned variational scales $\sigma$")

def plot_grad_dynamics(hist):
    # 1. Norms
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(hist.ep, hist.grad_norm, color=PALETTE["primary"], lw=2.0)
    ax.fill_between(hist.ep, hist.grad_norm, color=PALETTE["primary"], alpha=0.1)
    ax.axhline(5.0, color=PALETTE["neutral"], ls="--", alpha=0.6, label=r"Clip Threshold $\tau=5$")
    ax.set_yscale("log"); ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel(r"Gradient Norm $\|\nabla\Theta_m\|_2$")
    ax.legend(); save_fig(fig, "viz_grad_norm_evolution", "RNN gradient norm (BPTT) evolution")

    # 2. Scale
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(hist.ep, hist.grad_scale, color=PALETTE["secondary"], lw=2.0)
    ax.fill_between(hist.ep, hist.grad_scale, color=PALETTE["secondary"], alpha=0.1)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("Clipping Scale Factor")
    save_fig(fig, "viz_grad_scale_evolution", "Gradient clipping scale factor over training")

def plot_metric_table_image(metrics_df):
    fig, ax = plt.subplots(figsize=(14, 0.8 + 0.6*len(metrics_df)))
    ax.axis("off")
    tbl = ax.table(cellText=np.round(metrics_df.values, 4),
                   rowLabels=metrics_df.index,
                   colLabels=[c.replace("_", " ").title() for c in metrics_df.columns],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.0)
    for (i, j), c in tbl.get_celld().items():
        if i == 0:
            c.set_facecolor(PALETTE["primary"]); c.set_text_props(color="white", weight="bold")
        elif j == -1:
            c.set_facecolor("#f1f1f1"); c.set_text_props(weight="bold")
    save_fig(fig, "metric_summary_table", "Headline metric comparison across HINN and ablations")


def plot_per_class(y_true, Yhat, name, tag_label="HINN-Machine"):
    p, r, f, _ = precision_recall_fscore_support(y_true, Yhat.argmax(1),
                                                 labels=[0, 1, 2], zero_division=0)
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    df = pd.DataFrame({"Precision": p, "Recall": r, "F1 Score": f}, index=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind="bar", ax=ax, color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["neutral"]], edgecolor="white", alpha=0.8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Metric Value"); ax.legend(loc="lower right")
    ax.set_xticklabels(labels, rotation=0)
    save_fig(fig, name, f"Per-class performance metrics ({tag_label})")
    return df

def plot_ablation_compare(hist_main, hist_data, hist_human):
    # 1. Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    for h, lbl, c, ls in [(hist_main, "HINN (Full)", PALETTE["neutral"], "-"),
                         (hist_data, r"Data-Only ($\lambda=1$)", PALETTE["primary"], "--"),
                         (hist_human, r"Human-Only ($\lambda=0$)", PALETTE["secondary"], "-.")]:
        ax.plot(h.ep, h.acc_va, label=lbl, color=c, lw=2.5, ls=ls)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("Validation Accuracy"); ax.legend()
    save_fig(fig, "ablation_accuracy_compare", "Validation accuracy comparison across ablations")

    # 2. F1
    fig, ax = plt.subplots(figsize=(8, 6))
    for h, lbl, c, ls in [(hist_main, "HINN (Full)", PALETTE["neutral"], "-"),
                         (hist_data, r"Data-Only ($\lambda=1$)", PALETTE["primary"], "--"),
                         (hist_human, r"Human-Only ($\lambda=0$)", PALETTE["secondary"], "-.")]:
        ax.plot(h.ep, h.f1_va, label=lbl, color=c, lw=2.5, ls=ls)
    ax.set_xlabel("Epoch ($e$)"); ax.set_ylabel("Validation Macro-F1"); ax.legend()
    save_fig(fig, "ablation_f1_compare", "Validation macro-F1 comparison across ablations")

def plot_lambda_vs_metrics(hist):
    # 1. Trust vs Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(hist.lam_mean, hist.acc_va, c=hist.ep, cmap="Blues", s=50, alpha=0.8, edgecolor="0.3")
    ax.set_xlabel(r"Mean Trust $\overline{\lambda(t)}$"); ax.set_ylabel("Validation Accuracy")
    fig.colorbar(sc, label="Epoch ($e$)")
    save_fig(fig, "analysis_trust_vs_accuracy", "Empirical relationship between trust and accuracy")

    # 2. Trust vs Uncertainty
    fig, ax = plt.subplots(figsize=(8, 6))
    sc2 = ax.scatter(hist.post_var_mean, hist.lam_mean, c=hist.ep, cmap="Blues", s=50, alpha=0.8, edgecolor="0.3")
    ax.set_xlabel(r"Posterior Uncertainty $\overline{\mathrm{Tr}(\Sigma_q)}$"); ax.set_ylabel(r"Mean Trust $\overline{\lambda(t)}$")
    fig.colorbar(sc2, label="Epoch ($e$)")
    save_fig(fig, "analysis_trust_vs_uncertainty", "Empirical relationship between trust and BNN uncertainty")

def plot_robustness_curve(rnn, X_va, y_va):
    rng = np.random.default_rng(SEED)
    sigmas = np.linspace(0, 0.6, 11); accs = []
    for s in sigmas:
        Xn = X_va + rng.normal(0, s, size=X_va.shape)
        cache = rnn.forward(Xn)
        accs.append(accuracy_score(y_va, cache["Y"].argmax(1)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sigmas, accs, "-s", color=PALETTE["primary"], lw=2.5, markersize=10, markerfacecolor="white")
    ax.set_xlabel(r"Gaussian Noise Intensity $\sigma_{\eta}$"); ax.set_ylabel("Validation Accuracy")
    save_fig(fig, "analysis_robustness_noise", "Model robustness to input feature noise")


def plot_human_disagreement(Yhat_m, Yhat_h):
    diff = (Yhat_m.argmax(1) != Yhat_h.argmax(1)).astype(int)
    rate = pd.Series(diff).rolling(40).mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(diff, color=PALETTE["accent"], alpha=0.2, lw=0.5, label="Raw Disagreement")
    ax.plot(rate, color=PALETTE["primary"], lw=2.5, label="Rolling Mean (W=40)")
    ax.fill_between(range(len(rate)), rate, color=PALETTE["primary"], alpha=0.1)
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel("Disagreement Rate"); ax.legend()
    save_fig(fig, "analysis_pathway_disagreement", "Disagreement between machine and human pathways")

def plot_correlation_heatmap(hist):
    cols = ["L_total", "L_data", "L_human", "L_kl", "acc_va", "f1_va",
            "lam_mean", "rho_mean", "c_mean", "post_var_mean", "kl"]
    display_cols = [c.replace("_", " ").title() for c in cols]
    M = hist[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="Blues", center=0, ax=ax,
                xticklabels=display_cols, yticklabels=display_cols,
                annot_kws={"size": 11, "weight": "bold"})
    save_fig(fig, "analysis_metric_correlation", "Pearson correlation between training metrics")

def plot_summary_dashboard(hist):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)
    metrics = [
        ("L_total", r"$\mathcal{L}_{\mathrm{HINN}}$", PALETTE["neutral"]),
        ("acc_va", "Val Accuracy", PALETTE["primary"]),
        ("lam_mean", r"$\overline{\lambda(t)}$", PALETTE["secondary"]),
        ("post_var_mean", r"$\overline{\mathrm{Tr}(\Sigma_q)}$", PALETTE["primary"]),
        ("kl", "KL Divergence", PALETTE["accent"]),
        ("ece_tr", "ECE (Train)", PALETTE["secondary"]),
        ("brier_tr", "Brier (Train)", PALETTE["primary"]),
        ("auc_tr_ovo", "AUC (Train)", PALETTE["neutral"])
    ]
    for i, (col, lbl, c) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(hist.ep, hist[col], color=c, lw=2.5)
        ax.fill_between(hist.ep, hist[col], color=c, alpha=0.1)
        ax.set_title(lbl, fontsize=16, weight="bold"); ax.set_xlabel("Epoch ($e$)")
    fig.suptitle("HINN Training Summary Dashboard", fontsize=20, weight="bold")
    save_fig(fig, "summary_dashboard_grid", "Comprehensive training overview dashboard")

def plot_correlation_assets(big):
    cols = ["ret_idx", "ret_a2", "ret_a3", "ret_a4", "ret_a5"]
    display_labels = ["Asset 1 (Index)", "Asset 2", "Asset 3", "Asset 4", "Asset 5"]
    M = big[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="Blues", center=0,
                vmin=-1, vmax=1, ax=ax, square=True,
                xticklabels=display_labels, yticklabels=display_labels,
                annot_kws={"size": 12, "weight": "bold"})
    save_fig(fig, "data_asset_correlation", "Cross-asset return correlation matrix")


def plot_acf(big):
    def _acf(x, nlags):
        x = x - x.mean()
        var = (x*x).mean() + 1e-12
        out = np.empty(nlags + 1)
        out[0] = 1.0
        for k in range(1, nlags + 1):
            out[k] = (x[:-k] * x[k:]).mean() / var
        return out
    r  = big["ret_idx"].values
    r2 = r**2
    lags = 30
    a_r  = _acf(r,  lags)
    a_r2 = _acf(r2, lags)
    # 1. Returns ACF
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(lags+1), a_r, color=PALETTE["primary"], edgecolor=PALETTE["neutral"], alpha=0.7)
    ax.axhline(0, color="black", lw=1.0)
    conf = 1.96/np.sqrt(len(r))
    ax.axhline( conf, color=PALETTE["secondary"], ls="--", alpha=0.8, label="95% CI")
    ax.axhline(-conf, color=PALETTE["secondary"], ls="--", alpha=0.8)
    ax.set_xlabel("Lag ($k$)"); ax.set_ylabel("Autocorrelation"); ax.legend()
    save_fig(fig, "analysis_acf_returns", "Autocorrelation function of asset returns")

    # 2. Squared Returns ACF
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(lags+1), a_r2, color=PALETTE["secondary"], edgecolor=PALETTE["neutral"], alpha=0.7)
    ax.axhline(0, color="black", lw=1.0)
    ax.axhline(conf, color=PALETTE["primary"], ls="--", alpha=0.8, label="95% CI")
    ax.set_xlabel("Lag ($k$)"); ax.set_ylabel("Autocorrelation"); ax.legend()
    save_fig(fig, "analysis_acf_volatility", "Autocorrelation of squared returns (volatility clustering)")

def plot_class_by_regime(y_data, regimes):
    df = pd.DataFrame({"class": y_data, "regime": regimes})
    ct = pd.crosstab(df.regime, df["class"], normalize="index")
    ct.columns = ["Low Risk", "Medium Risk", "High Risk"]
    ct.index = ["Bullish", "Sideways", "Bearish"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[PALETTE["primary"], PALETTE["secondary"], "#e1e1e1"],
            edgecolor="white", alpha=0.9)
    ax.set_ylabel("Relative Frequency"); ax.set_xlabel(r"Market State ($\mathcal{S}_t$)"); ax.legend(title="Risk Class")
    ax.set_xticklabels(ct.index, rotation=0)
    save_fig(fig, "analysis_class_regime_dist", "Risk-class distribution across market regimes")

def plot_per_class_human(yh_va, Yh_va_mean):
    p, r, f, _ = precision_recall_fscore_support(yh_va, Yh_va_mean.argmax(1),
                                                 labels=[0, 1, 2], zero_division=0)
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    df = pd.DataFrame({"Precision": p, "Recall": r, "F1 Score": f}, index=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind="bar", ax=ax, color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["neutral"]], edgecolor="white", alpha=0.8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Metric Value"); ax.legend(loc="lower right")
    ax.set_xticklabels(labels, rotation=0)
    save_fig(fig, "metric_per_class_human", "Per-class performance for the human pathway (validation)")

def plot_posterior_kde(rnn, bnn, X):
    rng = np.random.default_rng(SEED + 31)
    cache = rnn.forward(X); Yhat = cache["Y"]
    S = 200
    eps = rng.normal(size=(S, bnn.n_params))
    thetas = bnn.mu + bnn.sigma * eps
    samples = np.zeros((S, X.shape[0], K))
    for s in range(S):
        zb, _ = bnn.forward_sample(Yhat, thetas[s])
        samples[s] = softmax(zb, axis=-1)
    t_idx = X.shape[0] // 2
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["neutral"]]
    for k in range(K):
        sns.kdeplot(samples[:, t_idx, k], fill=True, alpha=0.3,
                    color=colors[k], label=f"P({labels[k]})", ax=ax, lw=2)
    ax.set_xlim(0, 1); ax.set_xlabel(r"Posterior Probability $P(y_t | x_t, \mathcal{D})$"); ax.set_ylabel(r"Density $\pi(p)$"); ax.legend()
    save_fig(fig, "analysis_posterior_density", f"Posterior-predictive density at $t={t_idx}$ ($S=200$)")


def plot_uncertainty_fan(rnn, bnn, X, y, S=100, t_window=(50, 150)):
    rng = np.random.default_rng(SEED + 32)
    cache = rnn.forward(X); Yhat = cache["Y"]
    eps = rng.normal(size=(S, bnn.n_params))
    thetas = bnn.mu + bnn.sigma * eps
    samples = np.zeros((S, X.shape[0], K))
    for s in range(S):
        zb, _ = bnn.forward_sample(Yhat, thetas[s])
        samples[s] = softmax(zb, axis=-1)
    lo, hi = t_window
    qs = np.quantile(samples[:, lo:hi, 2], [0.05, 0.5, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    t_axis = np.arange(lo, hi)
    ax.fill_between(t_axis, qs[0], qs[2], color=PALETTE["secondary"], alpha=0.2, label="5–95% Credible Band")
    ax.plot(t_axis, qs[1], color=PALETTE["secondary"], lw=2.5, label="Posterior Median")
    ax.plot(t_axis, Yhat[lo:hi, 2], color=PALETTE["primary"], lw=1.5, ls="--", label=r"Machine Output $P_{\mathrm{m}}$")
    ax.scatter(t_axis, (y[lo:hi] == 2).astype(float), s=20, color=PALETTE["neutral"], marker="x", alpha=0.6, label="Actual High Risk")
    ax.set_xlabel("Time Step ($t$)"); ax.set_ylabel("Probability"); ax.legend(loc="upper left", ncol=2)
    save_fig(fig, "analysis_uncertainty_fan", "BNN posterior-predictive fan for high-risk class")

def plot_hp_sensitivity(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va):
    lambdas = [0.20, 0.40, 0.60, 0.80]
    alphas  = [0.5, 1.5, 3.0, 6.0]
    grid = np.zeros((len(lambdas), len(alphas)))
    for i, lm in enumerate(lambdas):
        for j, al in enumerate(alphas):
            res = train_hinn(X_tr, y_tr, yh_tr, m_tr, X_va, y_va, yh_va, m_va,
                             n_epochs=12, lr_m=5e-2, lr_phi=3e-2, S_mc=4,
                             trust_kw=dict(lambda_min=lm, alpha=al,
                                           gamma=0.05, kappa=2.0),
                             verbose=False)
            grid[i, j] = res["history"].acc_va.iloc[-3:].mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(grid, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=alphas, yticklabels=lambdas, ax=ax,
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_xlabel(r"Trust Sensitivity $\alpha$"); ax.set_ylabel(r"Minimum Trust $\lambda_{\min}$")
    save_fig(fig, "analysis_hp_sensitivity", "Validation accuracy sensitivity to trust hyper-parameters")

def plot_per_regime_accuracy(rnn, X, y, regimes):
    cache = rnn.forward(X); pred = cache["Y"].argmax(1)
    df = pd.DataFrame({"correct": (pred == y).astype(int), "regime": regimes})
    accs = df.groupby("regime")["correct"].mean()
    counts = df.groupby("regime")["correct"].count()
    labels = ["Bullish", "Sideways", "Bearish"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, accs.values,
                  color=[PALETTE["primary"], PALETTE["secondary"], "#e1e1e1"],
                  edgecolor=PALETTE["neutral"], alpha=0.8)
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.01,
                f"n={c}", ha="center", fontsize=12, weight="bold")
    ax.axhline(1/3, color=PALETTE["neutral"], ls=":", alpha=0.6, label="Random (0.33)")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy"); ax.set_xlabel(r"Market State ($\mathcal{S}_t$)"); ax.legend()
    save_fig(fig, "metric_accuracy_by_regime", "Machine classification accuracy stratified by market regime")



# ===========================================================================
# 7.  MAIN
# ===========================================================================

def main():
    t0 = time.time()
    print("="*78)
    print(" HINN  --  faithful numerical simulation")
    print("="*78)

    big, prices, regimes = generate_financial_dataset(n_days=1260)
    big.to_csv(os.path.join(DATADIR, "big_features.csv"), index=False)
    prices.to_csv(os.path.join(DATADIR, "prices.csv"), index=False)
    print(f"\n[1] Big-data: {big.shape[0]} steps × {big.shape[1]} features")

    y_data, y_human, m, fwd_vol = make_labels(big, prices)
    print(f"[2] y_t classes: low={np.sum(y_data==0)}  med={np.sum(y_data==1)}  "
          f"high={np.sum(y_data==2)} | human coverage = {m.mean()*100:.1f}%")

    feat_cols = ["ret_idx","ret_a2","ret_a3","ret_a4","ret_a5",
                 "ma10","ma50","rsi","vol20","macd"]
    X = big[feat_cols].values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    split = int(0.7 * X.shape[0])
    X_tr, y_tr, yh_tr, m_tr = X[:split], y_data[:split], y_human[:split], m[:split]
    X_va, y_va, yh_va, m_va = X[split:], y_data[split:], y_human[split:], m[split:]

    plot_dataset_overview(big, prices, regimes)
    plot_label_structure(y_data, y_human, m, fwd_vol)

    print("\n[3] Training full HINN ...")
    res = train_hinn(X_tr, y_tr, yh_tr, m_tr,
                     X_va, y_va, yh_va, m_va,
                     n_epochs=40, lr_m=5e-2, lr_phi=3e-2, S_mc=8)

    plot_training_curves(res["history"])
    plot_trust_dynamics(res, m_tr)
    plot_uncertainty(res)
    Yhat_tr, Yh_tr_mean, _ = plot_predictions(
        res["rnn"], res["bnn"], X_tr, y_tr, yh_tr, m_tr, tag="train")
    Yhat_va, Yh_va_mean, _ = plot_predictions(
        res["rnn"], res["bnn"], X_va, y_va, yh_va, m_va, tag="val")

    print("\n[4] Ablation: data-only ...")
    res_data = ablation_data_only(X_tr, y_tr, yh_tr, m_tr,
                                  X_va, y_va, yh_va, m_va,
                                  n_epochs=40, lr_m=5e-2, lr_phi=3e-2,
                                  S_mc=4, verbose=False)
    print("[5] Ablation: human-only ...")
    res_human = ablation_human_only(X_tr, y_tr, yh_tr, m_tr,
                                    X_va, y_va, yh_va, m_va,
                                    n_epochs=40, lr_m=5e-2, lr_phi=3e-2,
                                    S_mc=4, verbose=False)

    plot_confusion(y_va,  Yhat_va.argmax(1),  "metric_cm_machine", "Machine Pathway")
    plot_confusion(yh_va, Yh_va_mean.argmax(1),"metric_cm_human",  "Human Pathway")
    plot_calibration(Yhat_va, y_va, "machine", "Machine Pathway")
    plot_calibration(Yh_va_mean, yh_va, "human", "Human Pathway")
    plot_3d_loss_surface(res["rnn"], X_tr, y_tr)
    plot_3d_trust_surface()
    plot_3d_yhat_trajectory(res["Yhat_history"])
    plot_param_distributions(res["bnn"])
    plot_grad_dynamics(res["history"])
    plot_per_class(y_va, Yhat_va, "metric_per_class_machine", "Machine Pathway (Validation)")
    plot_ablation_compare(res["history"], res_data["history"], res_human["history"])
    plot_lambda_vs_metrics(res["history"])
    plot_robustness_curve(res["rnn"], X_va, y_va)
    plot_human_disagreement(Yhat_va, Yh_va_mean)
    plot_correlation_heatmap(res["history"])
    plot_summary_dashboard(res["history"])

    # extra deep-dive figures
    plot_correlation_assets(big)
    plot_acf(big)
    plot_class_by_regime(y_data, regimes)
    plot_per_class_human(yh_va, Yh_va_mean)
    plot_posterior_kde(res["rnn"], res["bnn"], X_va)
    plot_uncertainty_fan(res["rnn"], res["bnn"], X_va, y_va)
    plot_per_regime_accuracy(res["rnn"], X, y_data, regimes)
    print("\n[7] Hyper-parameter sensitivity grid (4x4) ...")
    plot_hp_sensitivity(X_tr, y_tr, yh_tr, m_tr,
                        X_va, y_va, yh_va, m_va)


    def headline(r):
        h = r["history"].iloc[-1]
        return {"val accuracy": h.acc_va, "val macro-F1": h.f1_va,
                "human-head F1 (val)": h.f1_h_va, "ECE (train)": h.ece_tr,
                "Brier (train)": h.brier_tr, "AUC OVO (train)": h.auc_tr_ovo,
                "λ̄ final": h.lam_mean, "post.var final": h.post_var_mean,
                "KL final": h.kl, "loss final": h.L_total}
    table = pd.DataFrame({"HINN (full)": headline(res),
                           "data-only":  headline(res_data),
                           "human-only": headline(res_human)}).T
    table.to_csv(os.path.join(TBLDIR, "headline_metrics.csv"))
    print("\n[6] Headline metrics:\n", table.round(4))
    plot_metric_table_image(table.round(4))

    res["history"].to_csv(os.path.join(TBLDIR, "history_hinn.csv"), index=False)
    res_data["history"].to_csv(os.path.join(TBLDIR, "history_data_only.csv"), index=False)
    res_human["history"].to_csv(os.path.join(TBLDIR, "history_human_only.csv"), index=False)

    elapsed = time.time() - t0
    write_report(elapsed, table)
    print(f"\nDone in {elapsed:.1f} s. {len(FIG_INDEX)} figures saved to {FIGDIR}")


def write_report(elapsed, table):
    md = []
    md.append("# HINN Numerical Simulation — Report\n")
    md.append(f"- runtime: **{elapsed:.1f} s**")
    md.append(f"- figures generated: **{len(FIG_INDEX)}**\n")
    md.append("## Headline metrics\n")
    try:
        md.append(table.round(4).to_markdown())
    except Exception:
        md.append(table.round(4).to_string())
    md.append("\n## Figure index\n")
    for i, (path, cap) in enumerate(FIG_INDEX, 1):
        md.append(f"{i}. `{path}` — {cap}")
    with open(os.path.join(OUTDIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
