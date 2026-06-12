#!/usr/bin/env python3
"""
wind_gmm_bayes.py

全貝式 GMM 風電預測 pipeline
替代 Di Persio & Ghadiri (2026) 的 Weibull + XGBoost：
  - Log-Normal Mixture (Gibbs, K=2) 替代 Weibull 不變分布
  - Bivariate GMM p(v,P) (Gibbs, K=4) 替代 XGBoost 功率曲線
  - 日尺度 EM 參數 + CRPS 驗證
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import invwishart, multivariate_normal
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA    = os.path.join(os.path.dirname(__file__), "data",    "data_dipersio_kelmarsh_turbine1_jan2021.csv")
OUT     = os.path.join(os.path.dirname(__file__), "results", "wind_gmm_results.png")
OUT_SSM  = os.path.join(os.path.dirname(__file__), "results", "wind_ssm_results.png")
OUT_DIAG = os.path.join(os.path.dirname(__file__), "results", "wind_gmm_diagnostics.png")

# ── 1. Data ───────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['v']    = df['wind_speed_avg_ms'].clip(lower=0.01)   # log transform 需要正值
    df['P']    = df['active_power_avg_kw'].clip(lower=0.0)  # 待機負功率 → 0
    return df


# ── 2. Gibbs：Log-Normal Mixture (Normal-IG 共軛) ─────────────────────────

def gibbs_lnmix(v, K=2, n_iter=3000, burnin=500, seed=42):
    """
    在 log-space 跑 K-component Normal Mixture 的 Gibbs sampler。
    回傳後驗樣本 dict: w (n_iter,K), mu (n_iter,K), sig2 (n_iter,K)
    """
    rng = np.random.default_rng(seed)
    u   = np.log(v)
    N   = len(u)

    # 弱資訊先驗
    alpha0 = np.ones(K)
    kappa0 = 0.1
    mu0    = np.linspace(np.percentile(u, 25), np.percentile(u, 75), K)
    a0     = 2.0
    b0     = np.var(u)

    # EM 初始化
    gm     = GaussianMixture(K, covariance_type='full', random_state=seed).fit(u.reshape(-1, 1))
    z      = gm.predict(u.reshape(-1, 1))
    mu_k   = gm.means_.flatten().copy()
    sig2_k = gm.covariances_[:, 0, 0].copy()
    w      = gm.weights_.copy()

    W_s, MU_s, SIG2_s = [], [], []

    for it in range(n_iter + burnin):
        # z_i ~ Categorical (log-sum-exp 穩定)
        logp = np.array([
            np.log(w[k] + 1e-300) + stats.norm.logpdf(u, mu_k[k], np.sqrt(sig2_k[k]))
            for k in range(K)
        ]).T                                             # (N, K)
        logp -= logp.max(1, keepdims=True)
        p     = np.exp(logp)
        p    /= p.sum(1, keepdims=True)
        cumW  = np.cumsum(p, axis=1)
        r     = rng.random(N)
        z     = (r[:, None] > cumW).sum(1).clip(0, K - 1)

        # w ~ Dirichlet
        nk = np.bincount(z, minlength=K).astype(float)
        w  = rng.dirichlet(alpha0 + nk)

        # mu_k, sig2_k (Normal-IG 共軛)
        for k in range(K):
            uk  = u[z == k]
            nk_ = len(uk)
            if nk_ < 2:
                continue
            ub  = uk.mean()
            Sk  = ((uk - ub) ** 2).sum()
            kn  = kappa0 + nk_
            mn  = (kappa0 * mu0[k] + nk_ * ub) / kn
            an  = a0 + nk_ / 2
            bn  = b0 + Sk / 2 + kappa0 * nk_ * (ub - mu0[k]) ** 2 / (2 * kn)

            sig2_k[k] = 1.0 / rng.gamma(an, 1.0 / bn)
            mu_k[k]   = rng.normal(mn, np.sqrt(sig2_k[k] / kn))

        if it >= burnin:
            W_s.append(w.copy())
            MU_s.append(mu_k.copy())
            SIG2_s.append(sig2_k.copy())

    return {
        'w':    np.array(W_s),
        'mu':   np.array(MU_s),
        'sig2': np.array(SIG2_s),
    }


# ── 3. Gibbs：Bivariate GMM p(v,P) (Normal-IW 共軛) ──────────────────────

def gibbs_bigmm(X, K=4, n_iter=3000, burnin=500, seed=42):
    """
    2D Normal Mixture Gibbs sampler (Normal-Inverse-Wishart 共軛)。
    回傳後驗樣本 dict: w (n_iter,K), mu (n_iter,K,2), Sigma (n_iter,K,2,2)
    """
    rng  = np.random.default_rng(seed)
    N, D = X.shape

    # 弱資訊先驗
    alpha0 = np.ones(K)
    kappa0 = 0.01
    nu0    = D + 2                          # IG 最小自由度
    m0     = X.mean(0)
    Psi0   = np.cov(X.T) / K               # E[Sigma_k] ≈ global_cov / K

    # EM 初始化
    gm      = GaussianMixture(K, covariance_type='full', random_state=seed).fit(X)
    z       = gm.predict(X)
    mu_k    = gm.means_.copy()
    Sigma_k = gm.covariances_.copy()
    w       = gm.weights_.copy()

    W_s, MU_s, SIG_s = [], [], []

    for it in range(n_iter + burnin):
        # z_i ~ Categorical
        logp = np.array([
            np.log(w[k] + 1e-300) + multivariate_normal.logpdf(X, mu_k[k], Sigma_k[k])
            for k in range(K)
        ]).T
        logp -= logp.max(1, keepdims=True)
        p     = np.exp(logp)
        p    /= p.sum(1, keepdims=True)
        cumW  = np.cumsum(p, axis=1)
        r     = rng.random(N)
        z     = (r[:, None] > cumW).sum(1).clip(0, K - 1)

        # w ~ Dirichlet
        nk = np.bincount(z, minlength=K).astype(float)
        w  = rng.dirichlet(alpha0 + nk)

        # mu_k, Sigma_k (Normal-IW 共軛)
        for k in range(K):
            Xk  = X[z == k]
            nk_ = len(Xk)
            if nk_ < D + 2:
                continue
            xb   = Xk.mean(0)
            Sk   = (Xk - xb).T @ (Xk - xb)
            kn   = kappa0 + nk_
            nun  = nu0 + nk_
            mn   = (kappa0 * m0 + nk_ * xb) / kn
            diff = xb - m0
            Psin = Psi0 + Sk + kappa0 * nk_ / kn * np.outer(diff, diff)

            # IW 取樣（整數 seed 確保相容不同 scipy 版本）
            Sigma_k[k] = invwishart.rvs(
                df=nun, scale=Psin,
                random_state=int(rng.integers(2 ** 31 - 1))
            )
            mu_k[k] = rng.multivariate_normal(mn, Sigma_k[k] / kn)

        if it >= burnin:
            W_s.append(w.copy())
            MU_s.append(mu_k.copy())
            SIG_s.append(Sigma_k.copy())

    return {
        'w':     np.array(W_s),
        'mu':    np.array(MU_s),
        'Sigma': np.array(SIG_s),
    }


# ── 4. 條件功率分布 p(P|v) ────────────────────────────────────────────────

def cond_power(v_grid, samp):
    """
    給定 bivariate GMM 後驗樣本，計算每個 v 對應的 E[P|v] 和 std[P|v]。
    使用後驗均值參數（posterior mean estimator）。
    """
    w  = samp['w'].mean(0)     # (K,)
    mu = samp['mu'].mean(0)    # (K, 2)
    Si = samp['Sigma'].mean(0) # (K, 2, 2)
    K  = len(w)

    E_out = np.zeros(len(v_grid))
    V_out = np.zeros(len(v_grid))

    for i, v in enumerate(v_grid):
        # π_k(v) ∝ w_k * N(v; μ_{v,k}, Σ_{vv,k})
        lpi = np.array([
            np.log(w[k] + 1e-300) + stats.norm.logpdf(v, mu[k, 0], np.sqrt(Si[k, 0, 0]))
            for k in range(K)
        ])
        lpi -= lpi.max()
        pi   = np.exp(lpi)
        pi  /= pi.sum()

        # 各分量條件均值 & 條件方差（解析公式）
        Emuk = mu[:, 1] + Si[:, 0, 1] / Si[:, 0, 0] * (v - mu[:, 0])  # (K,)
        Vark = Si[:, 1, 1] - Si[:, 0, 1] ** 2 / Si[:, 0, 0]            # (K,)

        # 全期望 & 全方差定理
        E_out[i] = (pi * Emuk).sum()
        V_out[i] = (pi * (Vark + Emuk ** 2)).sum() - E_out[i] ** 2

    return E_out, np.sqrt(V_out.clip(0))


# ── 5. CRPS（向量化，O(n_draw²) spread 只算一次）───────────────────────────

def crps_monthly(v, samp, n_draw=500, seed=1):
    """
    後驗預測 Log-Normal Mixture 對全月觀測的平均 CRPS。
    使用 posterior predictive sampling：從後驗中取 n_draw 個參數，各抽一個預測值組成 ensemble。
    """
    rng   = np.random.default_rng(seed)
    n_s   = len(samp['w'])
    idx   = rng.choice(n_s, n_draw, replace=True)

    W    = samp['w'][idx]    # (n_draw, K)
    MU   = samp['mu'][idx]   # (n_draw, K)
    S2   = samp['sig2'][idx] # (n_draw, K)
    K    = W.shape[1]

    # 向量化 component 指派
    cumW  = np.cumsum(W, axis=1)                      # (n_draw, K)
    r     = rng.random(n_draw)[:, None]               # (n_draw, 1)
    k_idx = (r > cumW).sum(1).clip(0, K - 1)         # (n_draw,)

    # 從選定的 log-normal 分量取樣
    ii      = np.arange(n_draw)
    mu_sel  = MU[ii, k_idx]                           # (n_draw,)
    s2_sel  = S2[ii, k_idx]                           # (n_draw,)
    v_ens   = np.exp(rng.normal(mu_sel, np.sqrt(s2_sel)))  # (n_draw,)

    # CRPS = E|X-y| - 0.5*E|X-X'|
    # Spread（常數，跟觀測值無關）
    spread = 0.5 * np.abs(v_ens[:, None] - v_ens[None, :]).mean()

    # MAE 對每個觀測（broadcast：(N,1)-(1,n_draw)）
    mae = np.abs(v[:, None] - v_ens[None, :]).mean(1)     # (N,)

    return float((mae - spread).mean())


# ── 6. 日尺度 EM（快速，用於 regime 時序圖）────────────────────────────────

def daily_em_params(df, K=2):
    """對每天的風速資料跑 GaussianMixture EM，回傳日 regime 權重 DataFrame。"""
    rows = []
    for d, g in df.groupby('date'):
        u  = np.log(g['v'].values).reshape(-1, 1)
        gm = GaussianMixture(K, random_state=0).fit(u)
        order = gm.means_.flatten().argsort()   # 按 μ 小→大排序 (calm→windy)
        row = {'date': pd.Timestamp(d)}
        for j, k in enumerate(order):
            row[f'w{j+1}']  = float(gm.weights_[k])
            row[f'mu{j+1}'] = float(gm.means_[k, 0])
        rows.append(row)
    return pd.DataFrame(rows)


# ── 7. Posterior predictive check（PPC）────────────────────────────────────

def ppc_samples(samp, n=5000, seed=2):
    """從後驗預測分布抽取 n 個風速樣本（用於 QQ plot / 重疊密度）。"""
    rng  = np.random.default_rng(seed)
    n_s  = len(samp['w'])
    idx  = rng.choice(n_s, n, replace=True)
    W, MU, S2 = samp['w'][idx], samp['mu'][idx], samp['sig2'][idx]
    K         = W.shape[1]
    cumW      = np.cumsum(W, axis=1)
    r         = rng.random(n)[:, None]
    k_idx     = (r > cumW).sum(1).clip(0, K - 1)
    ii        = np.arange(n)
    return np.exp(rng.normal(MU[ii, k_idx], np.sqrt(S2[ii, k_idx])))


# ── 8. BIC 比較 ──────────────────────────────────────────────────────────

def bic_comparison(v, samp):
    """
    Weibull MLE vs Log-Normal Mixture EM MLE 的 BIC 比較。
    BIC 必須用 MLE，用後驗均值會因 prior shrinkage 低估 LN-Mix 的 LL，對它不公平。
    LN-Mix LL (original space) = N * gm.score(log_v) - sum(log_v)
    """
    n = len(v)
    u = np.log(v)  # log-space data

    # Weibull MLE
    c, _, sc = stats.weibull_min.fit(v, floc=0)
    ll_wbl   = stats.weibull_min.logpdf(v, c, 0, sc).sum()
    bic_wbl  = 2 * np.log(n) * 2 - 2 * ll_wbl     # 2 params

    # LN-Mix MLE via EM (sklearn GaussianMixture on log-space)
    K = samp['w'].shape[1]
    gm_mle = GaussianMixture(K, covariance_type='full', random_state=42).fit(u.reshape(-1, 1))
    # LL in original space = LL in log-space - sum(log v)  [Jacobian of log transform]
    ll_lnm   = n * gm_mle.score(u.reshape(-1, 1)) - u.sum()
    n_params = K * 2 + (K - 1)                     # K means + K vars + (K-1) free weights
    bic_lnm  = 2 * np.log(n) * n_params - 2 * ll_lnm

    return {
        'weibull': {'c': c, 'scale': sc, 'll': ll_wbl, 'bic': bic_wbl},
        'lnmix':   {'K': K, 'll': ll_lnm, 'bic': bic_lnm},
        'delta_bic': bic_lnm - bic_wbl,   # < 0 → LN-Mix wins
    }


# ── 9. 繪圖 ──────────────────────────────────────────────────────────────

def plot_all(df, lns, gmm, daily_df, crps_val, bic_res):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('全貝式 GMM 風電預測 Pipeline — Kelmarsh Turbine 1，2021-01', fontsize=13)
    v = df['v'].values
    P = df['P'].values

    # ── (a) 風速密度：Weibull vs LN-Mixture ──
    ax = axes[0, 0]
    ax.hist(v, bins=60, density=True, alpha=0.4, color='steelblue', label='Data')
    vg = np.linspace(0.05, v.max() * 1.06, 500)

    c, sc = bic_res['weibull']['c'], bic_res['weibull']['scale']
    ax.plot(vg, stats.weibull_min.pdf(vg, c, 0, sc), 'r--', lw=2,
            label=f'Weibull κ={c:.2f} λ={sc:.2f}  (BIC={bic_res["weibull"]["bic"]:.0f})')

    wm, mum, s2m = lns['w'].mean(0), lns['mu'].mean(0), lns['sig2'].mean(0)
    order_ln = mum.argsort()
    mix_pdf  = sum(wm[k] * stats.lognorm.pdf(vg, s=np.sqrt(s2m[k]), scale=np.exp(mum[k]))
                   for k in range(len(wm)))
    ax.plot(vg, mix_pdf, 'g-', lw=2,
            label=f'LN-Mix K={len(wm)} (Bayes)  (BIC={bic_res["lnmix"]["bic"]:.0f})')

    # Posterior predictive overlay
    v_pp = ppc_samples(lns, n=4000)
    ax.hist(v_pp, bins=60, density=True, alpha=0.15, color='green', label='Posterior predictive')
    ax.set(xlabel='Wind Speed (m/s)', ylabel='Density', title='(a) Wind Speed Distribution + BIC')
    ax.legend(fontsize=7.5)

    # ── (b) 功率曲線：GMM 條件分布 ──
    ax = axes[0, 1]
    ax.scatter(v, P, s=1, alpha=0.2, color='steelblue', label='Data (N=4464)')
    vg2    = np.linspace(0.3, v.max(), 150)
    Emean, Estd = cond_power(vg2, gmm)
    Emean  = Emean.clip(0)
    ax.plot(vg2, Emean, 'r-', lw=2.5, label='E[P|v] — GMM posterior mean')
    ax.fill_between(vg2, (Emean - 2 * Estd).clip(0), Emean + 2 * Estd,
                    alpha=0.2, color='red', label='±2σ (total variance)')
    ax.set(xlabel='Wind Speed (m/s)', ylabel='Power (kW)',
           title='(b) Power Curve：GMM Conditional p(P|v)')
    ax.legend(fontsize=7.5)

    # ── (c) 日 regime 權重（EM） ──
    ax = axes[1, 0]
    dates = daily_df['date']
    ax.bar(dates, daily_df['w1'], color='steelblue', alpha=0.8, label='w₁ calm regime')
    ax.bar(dates, daily_df['w2'], bottom=daily_df['w1'],
           color='tomato', alpha=0.8, label='w₂ windy regime')
    ax.set(xlabel='Date', ylabel='Mixture Weight',
           title='(c) Daily Wind Regime (LN-Mix K=2, EM)')
    ax.legend(fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ── (d) CRPS 摘要 ──
    ax = axes[1, 1]
    delta_bic = bic_res['delta_bic']
    winner    = 'LN-Mix wins ✓' if delta_bic < 0 else 'Weibull wins'
    lines = [
        ('Monthly CRPS  (LN-Mix K=2)', f'{crps_val:.4f} m/s', 'darkgreen'),
        ('Paper baseline (Weibull+SDE)', '1.569 – 1.575 m/s', 'gray'),
        ('', '', 'white'),
        ('ΔBIC  (LN-Mix − Weibull)', f'{delta_bic:+.1f}  →  {winner}', 'navy'),
    ]
    for i, (label, val, color) in enumerate(lines):
        ax.text(0.05, 0.82 - i * 0.18, label, transform=ax.transAxes,
                fontsize=10, color='#333333')
        ax.text(0.95, 0.82 - i * 0.18, val, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=color, ha='right')
    ax.axhline(0.5, color='#ddd', lw=0.5)
    ax.axis('off')
    ax.set_title('(d) Validation Summary')

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure saved: {OUT}")


# ── 8. Kalman Filter + RTS Smoother ──────────────────────────────────────

def kalman_filter(Y, A, Q, R_mat, mu0, P0):
    """
    Linear Gaussian SSM Kalman filter (H = I, i.e. full state observed).
    State eq: θ_t = A θ_{t-1} + ε,  ε ~ N(0,Q)
    Obs   eq: y_t = θ_t + η,        η ~ N(0,R)
    Joseph form covariance update for numerical stability.
    """
    T, d = Y.shape
    mu_f = np.zeros((T, d))
    P_f  = np.zeros((T, d, d))
    mu_p = np.zeros((T, d))   # predicted mean
    P_p  = np.zeros((T, d, d))  # predicted cov

    for t in range(T):
        # Predict
        prev_mu = mu0      if t == 0 else mu_f[t-1]
        prev_P  = P0       if t == 0 else P_f[t-1]
        mu_p[t] = A @ prev_mu
        P_p[t]  = A @ prev_P @ A.T + Q

        # Update (H = I → innovation cov S = P_p + R)
        S       = P_p[t] + R_mat
        K       = np.linalg.solve(S.T, P_p[t].T).T   # K = P_p S^{-1}
        mu_f[t] = mu_p[t] + K @ (Y[t] - mu_p[t])
        IK      = np.eye(d) - K
        P_f[t]  = IK @ P_p[t] @ IK.T + K @ R_mat @ K.T   # Joseph form

    return mu_f, P_f, mu_p, P_p


def rts_smoother(mu_f, P_f, mu_p, P_p, A):
    """
    Rauch-Tung-Striebel backward smoother.
    Returns smoothed means (T,d), smoothed covs (T,d,d),
    and cross-covs P_{t+1,t|T} (T,d,d) needed for EM M-step.
    """
    T, d  = mu_f.shape
    mu_s  = np.zeros((T, d))
    P_s   = np.zeros((T, d, d))
    P_cross = np.zeros((T, d, d))   # P_cross[t+1] = P_{t+1,t|T} = G_t P_{t+1|T}

    mu_s[-1] = mu_f[-1]
    P_s[-1]  = P_f[-1]

    for t in range(T - 2, -1, -1):
        # Smoother gain G_t = P_{t|t} A' P_{t+1|t}^{-1}
        # Solve: P_p[t+1] G_t' = A P_f[t]  (P_p symmetric)
        G       = np.linalg.solve(P_p[t+1], A @ P_f[t]).T
        mu_s[t] = mu_f[t] + G @ (mu_s[t+1] - mu_p[t+1])
        P_s[t]  = P_f[t]  + G @ (P_s[t+1] - P_p[t+1]) @ G.T
        P_cross[t+1] = G @ P_s[t+1]   # P_{t+1, t | T}

    return mu_s, P_s, P_cross


# ── 9. EM for LGSSM (Shumway-Stoffer, diagonal AR(1)) ────────────────────

def em_lgssm_diag(Y, n_iter=100):
    """
    EM for LGSSM with diagonal A (independent AR(1) per dimension).
    State eq: θ_t = diag(a) θ_{t-1} + ε,  ε ~ N(0, diag(q))
    Obs   eq: y_t = θ_t + η,              η ~ N(0, diag(r))

    AR coefficients are constrained to (-0.99, 0.99) for stationarity.
    Returns: a (d,), q (d,), r (d,), smoothed means (T,d), smoothed covs (T,d,d)
    """
    T, d = Y.shape
    dY   = np.diff(Y, axis=0)

    # Initialize
    a   = np.full(d, 0.5)
    q   = np.var(dY, axis=0).clip(1e-4)
    r   = q.copy()
    mu0 = Y[0].copy()
    P0  = np.diag(np.var(Y, axis=0))

    for _ in range(n_iter):
        A     = np.diag(a)
        Q_mat = np.diag(q)
        R_mat = np.diag(r)

        # E-step
        mu_f, P_f, mu_p, P_p = kalman_filter(Y, A, Q_mat, R_mat, mu0, P0)
        mu_s, P_s, P_cross   = rts_smoother(mu_f, P_f, mu_p, P_p, A)

        # Sufficient statistics
        # E[θ_t θ_t' | Y] = P_s[t] + outer(mu_s[t], mu_s[t])
        E_tt     = P_s + np.einsum('ti,tj->tij', mu_s, mu_s)        # (T,d,d)
        # E[θ_{t+1} θ_t' | Y] for t=0..T-2
        E_t1_t   = P_cross[1:] + np.einsum('ti,tj->tij', mu_s[1:], mu_s[:-1])  # (T-1,d,d)

        S00 = E_tt[:-1].sum(0)    # Σ E[θ_{t-1} θ_{t-1}']
        S11 = E_tt[1:].sum(0)     # Σ E[θ_t θ_t']
        S10 = E_t1_t.sum(0)       # Σ E[θ_t θ_{t-1}']

        # M-step: diagonal A only (each dim independent)
        a_new = np.diag(S10) / np.diag(S00).clip(1e-8)
        a_new = np.clip(a_new, -0.99, 0.99)
        A_new = np.diag(a_new)

        # Q (diagonal elements of innovation covariance)
        Q_full = (S11 - A_new @ S10.T) / (T - 1)
        q_new  = np.diag(Q_full).clip(1e-6)

        # R (diagonal: E[(y_t - θ_t)^2 | Y] per dim)
        # = diag( (1/T) * Σ_t [P_s[t] + outer(mu_s[t]-Y[t], mu_s[t]-Y[t])] )
        resid_sq = P_s + np.einsum('ti,tj->tij', mu_s - Y, mu_s - Y)  # (T,d,d)
        r_new    = np.diag(resid_sq.sum(0) / T).clip(1e-6)

        a, q, r = a_new, q_new, r_new
        mu0, P0 = mu_s[0].copy(), P_s[0].copy()

    return a, q, r, mu_s, P_s, mu_p, P_p


# ── 10. Daily SSM wrapper ─────────────────────────────────────────────────

def daily_ssm(daily_df):
    """
    State-Space Model on daily LN-Mix EM parameters.
    State θ_d = [logit(w1_d), mu1_d, mu2_d] (all unconstrained).
    Returns smoothed states + Day-32 one-step-ahead forecast.
    """
    # Transform to unconstrained space
    w1   = daily_df['w1'].values.clip(1e-4, 1 - 1e-4)
    logit_w = np.log(w1 / (1 - w1))
    mu1  = daily_df['mu1'].values
    mu2  = daily_df['mu2'].values
    Y    = np.column_stack([logit_w, mu1, mu2]).astype(float)  # (T, 3)

    a, q, r, mu_s, P_s, mu_p, P_p = em_lgssm_diag(Y, n_iter=150)

    # One-step-ahead forecast: "Day 32" (Feb 1)
    A_hat   = np.diag(a)
    Q_hat   = np.diag(q)
    mu_pred = A_hat @ mu_s[-1]
    P_pred  = A_hat @ P_s[-1] @ A_hat.T + Q_hat

    # Back-transform smoothed states
    w1_s  = 1 / (1 + np.exp(-mu_s[:, 0]))   # sigmoid
    mu1_s = mu_s[:, 1]
    mu2_s = mu_s[:, 2]

    # Back-transform forecast
    w1_pred  = float(1 / (1 + np.exp(-mu_pred[0])))
    mu1_pred = float(mu_pred[1])
    mu2_pred = float(mu_pred[2])

    return {
        'a': a, 'q': q, 'r': r,
        'Y': Y, 'mu_s': mu_s, 'P_s': P_s,
        'w1_s': w1_s, 'mu1_s': mu1_s, 'mu2_s': mu2_s,
        'w1_pred': w1_pred, 'mu1_pred': mu1_pred, 'mu2_pred': mu2_pred,
        'mu_pred': mu_pred, 'P_pred': P_pred,
    }


# ── 11. Plot SSM ──────────────────────────────────────────────────────────

def plot_ssm(daily_df, ssm):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Daily State-Space Model: Kalman Smoother + Day-32 Forecast', fontsize=12)

    dates   = [pd.Timestamp(d) for d in daily_df['date']]
    day32   = pd.Timestamp('2021-02-01')
    labels  = ['logit(w1)  [calm weight]',
               'mu1  [calm log-mean]',
               'mu2  [windy log-mean]']
    units   = ['logit', 'log m/s', 'log m/s']

    for j in range(3):
        ax   = axes[j]
        raw  = ssm['Y'][:, j]
        smth = ssm['mu_s'][:, j]
        std  = np.sqrt(ssm['P_s'][:, j, j])

        ax.scatter(dates, raw, s=18, color='steelblue', alpha=0.75, zorder=3,
                   label='Daily EM (noisy)')
        ax.plot(dates, smth, 'r-', lw=2, label='Kalman smoother')
        ax.fill_between(dates, smth - 2*std, smth + 2*std,
                        alpha=0.18, color='red', label='Smoother +/-2sd')

        # Day-32 forecast
        std_p = np.sqrt(ssm['P_pred'][j, j])
        ax.errorbar(day32, ssm['mu_pred'][j], yerr=2*std_p,
                    fmt='D', ms=8, color='green', capsize=5,
                    label=f'Day-32 forecast')

        ax.set(xlabel='Date', ylabel=units[j], title=f'{labels[j]}\nAR coef = {ssm["a"][j]:.3f}')
        ax.legend(fontsize=7)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha='right')

    plt.tight_layout()
    plt.savefig(OUT_SSM, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] SSM figure saved: {OUT_SSM}")


# ── 12. Posterior Diagnostics ────────────────────────────────────────────

def ess(x):
    """ESS via Geyer's initial positive sequence estimator (single chain)."""
    n  = len(x)
    xc = x - x.mean()
    f   = np.fft.fft(xc, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    acf /= (acf[0] + 1e-15)
    tau = 1.0
    for k in range(1, min(n // 2, 500)):
        pair = acf[2 * k - 1] + acf[2 * k]
        if pair <= 0:
            break
        tau += 2 * pair
    return float(n / tau)


def split_rhat(x):
    """Split-chain R-hat (Gelman-Rubin) for a single chain."""
    n    = len(x) // 2
    a, b = x[:n], x[n:]
    W    = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2
    if W < 1e-12:
        return 1.0
    m_all  = (a.mean() + b.mean()) / 2
    B      = n * ((a.mean() - m_all) ** 2 + (b.mean() - m_all) ** 2)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def run_diagnostics(v, K=2, n_iter=5000, burnin=1000, seed=99):
    """
    Run Gibbs sampler with longer chain, fix label switching,
    compute R-hat / ESS, and produce a 5x3 diagnostic figure.
    Parameters: w1, mu1, mu2, sig2_1, sig2_2 (sorted by mu ascending).
    """
    print(f"    Running Gibbs: K={K}, iter={n_iter}, burnin={burnin}...")
    samp = gibbs_lnmix(v, K=K, n_iter=n_iter, burnin=burnin, seed=seed)

    # Fix label switching: sort components by mu at each iteration
    W   = samp['w']     # (S, K)
    MU  = samp['mu']    # (S, K)
    S2  = samp['sig2']  # (S, K)
    S   = len(W)
    for s in range(S):
        order = MU[s].argsort()
        W[s]  = W[s][order]
        MU[s] = MU[s][order]
        S2[s] = S2[s][order]

    # Extract chains (5 params)
    chains = {
        'w1':     W[:, 0],
        'mu1':    MU[:, 0],
        'mu2':    MU[:, 1],
        'sig2_1': S2[:, 0],
        'sig2_2': S2[:, 1],
    }
    labels = ['w1 (calm weight)', 'mu1 (calm log-mean)',
              'mu2 (windy log-mean)', 'sigma^2_1 (calm)', 'sigma^2_2 (windy)']

    print()
    print(f"    {'Parameter':<14}  {'Mean':>8}  {'SD':>8}  {'2.5%':>8}  {'97.5%':>8}"
          f"  {'R-hat':>7}  {'ESS':>7}")
    print("    " + "-" * 72)
    rhat_vals, ess_vals = {}, {}
    for (key, chain), lab in zip(chains.items(), labels):
        rh = split_rhat(chain)
        es = ess(chain)
        q  = np.quantile(chain, [0.025, 0.975])
        rhat_vals[key] = rh
        ess_vals[key]  = es
        conv = "OK" if rh < 1.05 else "WARN"
        ess_ok = "OK" if es > 100 else "LOW"
        print(f"    {key:<14}  {chain.mean():>8.4f}  {chain.std():>8.4f}"
              f"  {q[0]:>8.4f}  {q[1]:>8.4f}"
              f"  {rh:>6.4f}[{conv}]  {es:>5.0f}[{ess_ok}]")

    all_conv = all(r < 1.05 for r in rhat_vals.values())
    all_ess  = all(e > 100  for e in ess_vals.values())
    if all_conv and all_ess:
        print("\n    CONVERGENCE: All R-hat < 1.05 and ESS > 100  ->  CONVERGED")
    else:
        issues = [k for k, r in rhat_vals.items() if r >= 1.05]
        low_e  = [k for k, e in ess_vals.items()  if e <= 100]
        if issues:
            print(f"\n    WARNING: R-hat >= 1.05 for: {', '.join(issues)}")
        if low_e:
            print(f"    WARNING: ESS <= 100 for: {', '.join(low_e)}")

    # ── Diagnostic figure: 5 params x 3 cols (trace | marginal | ACF) ──
    fig, axes = plt.subplots(5, 3, figsize=(13, 14))
    fig.suptitle(f'Gibbs Posterior Diagnostics  (K={K}, iter={n_iter}, burnin={burnin})',
                 fontsize=12)
    lags_max = 40

    for row, (key, chain) in enumerate(chains.items()):
        lab = labels[row]

        # ── Trace plot ──
        ax = axes[row, 0]
        ax.plot(chain, lw=0.4, alpha=0.8, color='steelblue')
        ax.axhline(chain.mean(), color='red', lw=1.2, ls='--', label='mean')
        ax.set(title=f'{lab}\nTrace (R-hat={rhat_vals[key]:.4f})',
               xlabel='Iteration', ylabel='Value')
        ax.legend(fontsize=7)

        # ── Posterior marginal histogram + KDE ──
        ax = axes[row, 1]
        ax.hist(chain, bins=60, density=True, color='steelblue', alpha=0.5)
        xg  = np.linspace(chain.min(), chain.max(), 300)
        kde = stats.gaussian_kde(chain)
        ax.plot(xg, kde(xg), 'r-', lw=1.8)
        q025, q975 = np.quantile(chain, [0.025, 0.975])
        ax.axvline(q025, color='gray', lw=1, ls=':', label='2.5%/97.5%')
        ax.axvline(q975, color='gray', lw=1, ls=':')
        ax.set(title=f'Posterior  CI=[{q025:.3f}, {q975:.3f}]',
               xlabel='Value', ylabel='Density')
        ax.legend(fontsize=7)

        # ── ACF (lags 0..lags_max) ──
        ax = axes[row, 2]
        n_c = len(chain)
        xc  = chain - chain.mean()
        acf_raw = np.fft.ifft(
            np.fft.fft(xc, n=2 * n_c) * np.conj(np.fft.fft(xc, n=2 * n_c))
        ).real[:lags_max + 1]
        acf_raw /= (acf_raw[0] + 1e-15)
        lags = np.arange(lags_max + 1)
        ci   = 1.96 / np.sqrt(n_c)
        ax.bar(lags, acf_raw, color='steelblue', alpha=0.7, width=0.8)
        ax.axhline(ci,  color='red', lw=1, ls='--')
        ax.axhline(-ci, color='red', lw=1, ls='--')
        ax.axhline(0,   color='black', lw=0.5)
        ax.set(title=f'ACF  (ESS={ess_vals[key]:.0f})',
               xlabel='Lag', ylabel='ACF', ylim=(-0.3, 1.05))

    plt.tight_layout()
    plt.savefig(OUT_DIAG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    [OK] Diagnostic figure saved: {OUT_DIAG}")
    return chains, rhat_vals, ess_vals


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" Full Bayesian GMM Wind Power Pipeline")
    print("=" * 55)

    print("\n[1] Loading data...")
    df    = load_data(DATA)
    v, P  = df['v'].values, df['P'].values
    X     = np.column_stack([v, P])
    print(f"    N={len(df)} | v=[{v.min():.2f}, {v.max():.2f}] m/s"
          f" | P=[{P.min():.0f}, {P.max():.0f}] kW")

    # ── Gibbs：Log-Normal Mixture ──
    print("\n[2] Gibbs: Log-Normal Mixture K=2  (iter=3000, burnin=500)...")
    lns = gibbs_lnmix(v, K=2, n_iter=3000, burnin=500)
    wm, mum, s2m = lns['w'].mean(0), lns['mu'].mean(0), lns['sig2'].mean(0)
    order_ln = mum.argsort()
    print("    Posterior mean (sorted by mu):")
    for j, k in enumerate(order_ln):
        med_v = np.exp(mum[k])
        sd_v  = med_v * (np.exp(s2m[k]) - 1) ** 0.5
        print(f"      k={j}: w={wm[k]:.3f}  mu_log={mum[k]:.3f}"
              f"  (v_median~{med_v:.2f} m/s, sigma_v~{sd_v:.2f})")

    # R-hat 簡易收斂診斷（前半 vs 後半均值差）
    half = len(lns['mu']) // 2
    rhat_approx = np.abs(lns['mu'][:half].mean(0) - lns['mu'][half:].mean(0)).max()
    conv_msg = "OK" if rhat_approx < 0.05 else "WARNING: increase iter"
    print(f"    Convergence check (|mu_first_half - mu_second_half|.max) = {rhat_approx:.5f}  [{conv_msg}]")

    # ── BIC comparison ──
    print("\n[3] BIC comparison (Weibull MLE vs LN-Mix EM MLE)...")
    bic_res = bic_comparison(v, lns)
    w_info  = bic_res['weibull']
    l_info  = bic_res['lnmix']
    print(f"    Weibull:  kappa={w_info['c']:.3f}, lambda={w_info['scale']:.3f}"
          f"  LL={w_info['ll']:.1f}  BIC={w_info['bic']:.1f}")
    print(f"    LN-Mix:   K={l_info['K']}"
          f"  LL={l_info['ll']:.1f}  BIC={l_info['bic']:.1f}")
    delta = bic_res['delta_bic']
    if delta < -10:
        verdict = "LN-Mix wins decisively"
    elif delta < 0:
        verdict = "LN-Mix slightly better"
    else:
        verdict = "Weibull wins"
    print(f"    dBIC = {delta:+.1f}  ->  {verdict}")

    # ── Gibbs：Bivariate GMM ──
    print("\n[4] Gibbs: Bivariate GMM K=4  (iter=3000, burnin=500)...")
    gmm     = gibbs_bigmm(X, K=4, n_iter=3000, burnin=500)
    wg, mug = gmm['w'].mean(0), gmm['mu'].mean(0)
    order_g  = mug[:, 0].argsort()    # 按 v 排序
    print("    Posterior mean components (sorted by v):")
    for j, k in enumerate(order_g):
        print(f"      k={j}: w={wg[k]:.3f}  v={mug[k,0]:.2f} m/s  P={mug[k,1]:.0f} kW")

    # ── CRPS ──
    print("\n[5] Computing CRPS (posterior predictive, n_draw=500)...")
    crps_val = crps_monthly(v, lns, n_draw=500)
    print(f"    Monthly CRPS = {crps_val:.4f} m/s")
    print(f"    Paper (Weibull+SDE) = 1.569-1.575 m/s")
    if crps_val < 1.569:
        print("    -> LN-Mix distribution fit better than Weibull baseline")
    else:
        print("    -> Comparable to baseline (in-sample eval; OOS validation needed)")

    print("\n[6] Daily EM parameter estimation (31 days)...")
    daily_df = daily_em_params(df)
    print("    First 7 days:")
    print(daily_df.head(7)[['date', 'w1', 'w2', 'mu1', 'mu2']].to_string(index=False))
    print(f"    calm  (w1) mean: {daily_df['w1'].mean():.3f} +/- {daily_df['w1'].std():.3f}")
    print(f"    windy (w2) mean: {daily_df['w2'].mean():.3f} +/- {daily_df['w2'].std():.3f}")

    # ── 繪圖 ──
    print("\n[7] Plotting 4-panel figure...")
    plot_all(df, lns, gmm, daily_df, crps_val, bic_res)

    # ── Daily SSM ──
    print("\n[8] Daily SSM: EM-LGSSM (diagonal AR(1), 150 iter)...")
    ssm = daily_ssm(daily_df)
    a, q, r = ssm['a'], ssm['q'], ssm['r']
    print(f"    AR coefficients  a = [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}]")
    print(f"    State noise  sqrt(q) = [{np.sqrt(q[0]):.4f}, {np.sqrt(q[1]):.4f}, {np.sqrt(q[2]):.4f}]")
    print(f"    Obs   noise  sqrt(r) = [{np.sqrt(r[0]):.4f}, {np.sqrt(r[1]):.4f}, {np.sqrt(r[2]):.4f}]")
    print(f"    Day-32 forecast (Feb 1):")
    print(f"      w1_pred   = {ssm['w1_pred']:.3f}  (calm weight)")
    print(f"      mu1_pred  = {ssm['mu1_pred']:.3f}  (v_calm  median ~ {np.exp(ssm['mu1_pred']):.2f} m/s)")
    print(f"      mu2_pred  = {ssm['mu2_pred']:.3f}  (v_windy median ~ {np.exp(ssm['mu2_pred']):.2f} m/s)")
    std_p = np.sqrt(np.diag(ssm['P_pred']))
    print(f"      Forecast 2-sigma (unconstrained): "
          f"+/-{2*std_p[0]:.3f}, +/-{2*std_p[1]:.4f}, +/-{2*std_p[2]:.4f}")

    print("\n[9] Plotting SSM figure...")
    plot_ssm(daily_df, ssm)

    print("\n[10] Posterior diagnostics (K=2, iter=5000, burnin=1000)...")
    run_diagnostics(v, K=2, n_iter=5000, burnin=1000)

    print("\n" + "=" * 55)
    print(" Done!")
    print("=" * 55)


if __name__ == '__main__':
    main()
