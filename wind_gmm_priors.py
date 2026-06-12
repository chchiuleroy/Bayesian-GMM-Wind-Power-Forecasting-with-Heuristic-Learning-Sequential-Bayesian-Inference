#!/usr/bin/env python3
"""
wind_gmm_priors.py
------------------
Prior sensitivity analysis + sequential Bayesian learning for LN-Mix.

Four inference modes on the same LN-Mix model (K=2, log-space Normal Mixture):

  1. Conjugate (Normal-IG, fixed weakly-informative)
     pi(mu_k) = N(mu0, sigma_k^2 / kappa0)
     pi(sigma_k^2) = IG(a0, b0)
     => closed-form Gibbs (existing approach)

  2. Empirical Bayes / Heuristic Learning (啟發式學習)
     Same Normal-IG but hyperparameters mu0, b0 learned from EM
     => data-driven prior; reduces sensitivity to manual tuning

  3. Jeffreys / Uniform (non-informative, non-conjugate in spirit)
     pi(mu_k) propto 1 (flat)
     pi(sigma_k^2) propto 1/sigma_k^2 (Jeffreys)
     => limiting case of Normal-IG with kappa0->0, a0->0, b0->0
     => still closed-form Gibbs (posterior dominated entirely by likelihood)

  4. Half-Cauchy sigma (genuinely non-conjugate via auxiliary variable)
     pi(sigma_k) = Half-Cauchy(0, s=2.5)
     => Equivalent to: sigma_k^2 | xi_k ~ IG(0.5, 1/xi_k)
                       xi_k          ~ IG(0.5, 1/s^2)
     => Two conjugate Gibbs steps (Makalic & Schmidt 2016 auxiliary trick)
     => Heavier-tailed prior on sigma; robust against over-shrinkage

Plus:
  - Sequential Bayesian learning: posterior of month t -> prior of month t+1
  - Comparison figure: posterior marginals + CRPS + ESS across four modes
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

DATA_PILOT = os.path.join(os.path.dirname(__file__), "data",
                          "data_dipersio_kelmarsh_turbine1_jan2021.csv")
DATA_FULL  = os.path.join(os.path.dirname(__file__), "data",
                          "kelmarsh_turbine1_all.parquet")
RES_DIR    = os.path.join(os.path.dirname(__file__), "results")


# ── utilities ────────────────────────────────────────────────────────────────

def _label_sort(W, MU, S2):
    """Sort mixture components by mu (ascending) at each iteration."""
    for s in range(len(W)):
        order  = MU[s].argsort()
        W[s]   = W[s][order]
        MU[s]  = MU[s][order]
        S2[s]  = S2[s][order]
    return W, MU, S2


def ess(x):
    """ESS via Geyer's initial positive sequence."""
    n   = len(x)
    xc  = x - x.mean()
    f   = np.fft.fft(xc, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    acf /= (acf[0] + 1e-15)
    tau = 1.0
    for k in range(1, min(n // 2, 500)):
        if acf[2*k-1] + acf[2*k] <= 0:
            break
        tau += 2 * (acf[2*k-1] + acf[2*k])
    return float(n / tau)


def crps_from_samp(v_obs, samp, n_draw=500, seed=1):
    """Monte Carlo CRPS from posterior samples dict."""
    rng  = np.random.default_rng(seed)
    n_s  = len(samp["w"])
    idx  = rng.choice(n_s, n_draw, replace=True)
    W, MU, S2 = samp["w"][idx], samp["mu"][idx], samp["sig2"][idx]
    K    = W.shape[1]
    cumW = np.cumsum(W, axis=1)
    k_idx = (rng.random(n_draw)[:, None] > cumW).sum(1).clip(0, K - 1)
    ii   = np.arange(n_draw)
    v_ens = np.exp(rng.normal(MU[ii, k_idx], np.sqrt(S2[ii, k_idx])))
    spread = 0.5 * np.abs(v_ens[:, None] - v_ens[None, :]).mean()
    mae   = np.abs(v_obs[:, None] - v_ens[None, :]).mean(1)
    return float((mae - spread).mean())


# ── core Gibbs kernels ────────────────────────────────────────────────────────

def _gibbs_categorical(u, w, mu_k, sig2_k, K, rng):
    """Sample assignment z_i ~ Categorical."""
    logp = np.array([
        np.log(w[k] + 1e-300) + stats.norm.logpdf(u, mu_k[k], np.sqrt(sig2_k[k]))
        for k in range(K)
    ]).T
    logp -= logp.max(1, keepdims=True)
    p    = np.exp(logp)
    p   /= p.sum(1, keepdims=True)
    cumW = np.cumsum(p, axis=1)
    r    = rng.random(len(u))
    return (r[:, None] > cumW).sum(1).clip(0, K - 1)


def _gibbs_weights(z, K, alpha0, rng):
    """w ~ Dirichlet(alpha0 + n_k)."""
    nk = np.bincount(z, minlength=K).astype(float)
    return rng.dirichlet(alpha0 + nk), nk


def _em_init(u, K, seed):
    """EM initialisation for Gibbs."""
    gm    = GaussianMixture(K, covariance_type="full", random_state=seed,
                            n_init=3).fit(u.reshape(-1, 1))
    order = gm.means_.flatten().argsort()
    z     = gm.predict(u.reshape(-1, 1))
    mu_k  = gm.means_.flatten()[order].copy()
    sig2  = gm.covariances_[:, 0, 0][order].copy()
    w     = gm.weights_[order].copy()
    return z, mu_k, sig2, w, gm


# ── 1. Conjugate Normal-IG Gibbs ─────────────────────────────────────────────

def gibbs_conj(v, K=2, n_iter=3000, burnin=500, seed=42,
               kappa0=0.1, a0=2.0, b0_scalar=None,
               mu0=None):
    """
    Conjugate Normal-IG Gibbs sampler.
    Default: weakly-informative fixed priors (existing approach).
    """
    rng    = np.random.default_rng(seed)
    u      = np.log(v)
    N      = len(u)
    alpha0 = np.ones(K)

    z, mu_k, sig2_k, w, _ = _em_init(u, K, seed)
    _b0 = (np.var(u) if b0_scalar is None else b0_scalar)
    _mu0 = (np.linspace(np.percentile(u, 25), np.percentile(u, 75), K)
            if mu0 is None else np.asarray(mu0))

    W_s, MU_s, SIG2_s = [], [], []

    for it in range(n_iter + burnin):
        z       = _gibbs_categorical(u, w, mu_k, sig2_k, K, rng)
        w, nk   = _gibbs_weights(z, K, alpha0, rng)

        for k in range(K):
            uk  = u[z == k]
            nk_ = len(uk)
            if nk_ < 2:
                continue
            ub  = uk.mean()
            Sk  = ((uk - ub) ** 2).sum()
            kn  = kappa0 + nk_
            mn  = (kappa0 * _mu0[k] + nk_ * ub) / kn
            an  = a0 + nk_ / 2
            bn  = _b0 + Sk / 2 + kappa0 * nk_ * (ub - _mu0[k]) ** 2 / (2 * kn)
            sig2_k[k] = 1.0 / rng.gamma(an, 1.0 / bn)
            mu_k[k]   = rng.normal(mn, np.sqrt(sig2_k[k] / kn))

        if it >= burnin:
            W_s.append(w.copy())
            MU_s.append(mu_k.copy())
            SIG2_s.append(sig2_k.copy())

    W_s, MU_s, SIG2_s = map(np.array, (W_s, MU_s, SIG2_s))
    W_s, MU_s, SIG2_s = _label_sort(W_s, MU_s, SIG2_s)
    return {"w": W_s, "mu": MU_s, "sig2": SIG2_s}


# ── 2. Empirical Bayes Gibbs (啟發式學習) ────────────────────────────────────

def gibbs_eb(v, K=2, n_iter=3000, burnin=500, seed=42):
    """
    Empirical Bayes (heuristic learning): priors centered on EM estimates.

    Strategy:
      - Run EM on the data to estimate component means and variances.
      - Set mu0_k  = EM mean_k          (center prior on data structure)
      - Set b0_k   = EM var_k * (a0-1)  (E[sigma^2] = EM variance)
      - kappa0 = 1.0  (moderate prior weight ~ 1 pseudo-observation)
      - a0 = 3.0      (slightly more precise IG than the flat default)

    This is 'heuristic' because the prior hyperparameters are learned from
    the same dataset rather than set independently.  Useful when:
      (a) limited data and you want to avoid degenerate components, or
      (b) you already trust EM estimates as a reasonable starting point.
    """
    rng    = np.random.default_rng(seed)
    u      = np.log(v)
    N      = len(u)
    alpha0 = np.ones(K)

    # ── learn prior from EM ──
    z0, mu_k, sig2_k, w, gm = _em_init(u, K, seed)
    order  = mu_k.argsort()
    mu0    = mu_k[order].copy()
    var_em = sig2_k[order].copy()
    kappa0 = 1.0
    a0     = 3.0
    b0_k   = var_em * (a0 - 1)            # E[sig^2] = b0 / (a0 - 1) = var_em

    # reinitialise from EM
    z = z0.copy()

    W_s, MU_s, SIG2_s = [], [], []

    for it in range(n_iter + burnin):
        z       = _gibbs_categorical(u, w, mu_k, sig2_k, K, rng)
        w, nk   = _gibbs_weights(z, K, alpha0, rng)

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
            bn  = b0_k[k] + Sk / 2 + kappa0 * nk_ * (ub - mu0[k]) ** 2 / (2 * kn)
            sig2_k[k] = 1.0 / rng.gamma(an, 1.0 / bn)
            mu_k[k]   = rng.normal(mn, np.sqrt(sig2_k[k] / kn))

        if it >= burnin:
            W_s.append(w.copy())
            MU_s.append(mu_k.copy())
            SIG2_s.append(sig2_k.copy())

    W_s, MU_s, SIG2_s = map(np.array, (W_s, MU_s, SIG2_s))
    W_s, MU_s, SIG2_s = _label_sort(W_s, MU_s, SIG2_s)
    return {"w": W_s, "mu": MU_s, "sig2": SIG2_s}


# ── 3. Jeffreys / Uniform (non-informative) ───────────────────────────────────

def gibbs_jeffreys(v, K=2, n_iter=3000, burnin=500, seed=42):
    """
    Jeffreys / improper uniform priors (non-informative):

      pi(mu_k)     propto 1              (flat on mu)
      pi(sigma_k^2) propto 1/sigma_k^2  (Jeffreys)
      pi(w)        = Dirichlet(1,...,1)  (uniform on simplex)

    This is the kappa0 -> 0, a0 -> 0, b0 -> 0 limit of Normal-IG.
    Full conditionals remain closed-form:
      mu_k    | sigma_k^2, data  ~ N(u_bar_k, sigma_k^2 / n_k)
      sigma_k^2 | mu_k, data    ~ IG(n_k/2, S_k/2)

    Posterior is dominated entirely by the likelihood (no shrinkage).
    Requires n_k >= 2 for the IG posterior to be proper.
    """
    rng    = np.random.default_rng(seed)
    u      = np.log(v)
    N      = len(u)
    alpha0 = np.ones(K)                 # Dirichlet(1,...,1) = uniform on simplex
    kappa0 = 1e-10                      # flat on mu
    a0     = 1e-6                       # flat on sigma^2 (Jeffreys)
    b0     = 1e-6

    z, mu_k, sig2_k, w, _ = _em_init(u, K, seed)
    mu0 = np.zeros(K)                   # irrelevant when kappa0 ~ 0

    W_s, MU_s, SIG2_s = [], [], []

    for it in range(n_iter + burnin):
        z       = _gibbs_categorical(u, w, mu_k, sig2_k, K, rng)
        w, nk   = _gibbs_weights(z, K, alpha0, rng)

        for k in range(K):
            uk  = u[z == k]
            nk_ = len(uk)
            if nk_ < 2:
                continue
            ub  = uk.mean()
            Sk  = ((uk - ub) ** 2).sum()
            kn  = kappa0 + nk_          # ~ n_k
            mn  = ub                    # posterior mean = data mean (flat prior)
            an  = a0 + nk_ / 2          # ~ n_k / 2
            bn  = b0 + Sk / 2           # no correction term (kappa0 ~ 0)
            sig2_k[k] = 1.0 / rng.gamma(an, 1.0 / bn)
            mu_k[k]   = rng.normal(mn, np.sqrt(sig2_k[k] / kn))

        if it >= burnin:
            W_s.append(w.copy())
            MU_s.append(mu_k.copy())
            SIG2_s.append(sig2_k.copy())

    W_s, MU_s, SIG2_s = map(np.array, (W_s, MU_s, SIG2_s))
    W_s, MU_s, SIG2_s = _label_sort(W_s, MU_s, SIG2_s)
    return {"w": W_s, "mu": MU_s, "sig2": SIG2_s}


# ── 4. Half-Cauchy sigma (auxiliary variable Gibbs) ───────────────────────────

def gibbs_half_cauchy(v, K=2, n_iter=3000, burnin=500, seed=42, s=2.5):
    """
    Half-Cauchy(0, s) prior on sigma_k (non-conjugate), implemented via the
    auxiliary variable trick (Makalic & Schmidt 2016, Gelman 2006):

      sigma_k^2  | xi_k  ~ IG(0.5, 1/xi_k)
      xi_k               ~ IG(0.5, 1/s^2)
      => marginal: sigma_k ~ Half-Cauchy(0, s)

    Full conditional updates:
      sigma_k^2 | xi_k, data ~ IG(n_k/2 + 0.5,  S_k/2 + 1/xi_k)
      xi_k      | sigma_k^2  ~ IG(1,            1/s^2 + 1/sigma_k^2)

    This prior is heavier-tailed than IG: it allows large sigma when the
    data support it, without being improper.  Commonly used in hierarchical
    models and RFL Bayesian inference.

    Prior on mu: flat (kappa0 -> 0), same as Jeffreys version.
    """
    rng    = np.random.default_rng(seed)
    u      = np.log(v)
    N      = len(u)
    alpha0 = np.ones(K)
    kappa0 = 1e-10                      # flat on mu
    s2inv  = 1.0 / s ** 2

    z, mu_k, sig2_k, w, _ = _em_init(u, K, seed)
    xi_k   = np.ones(K)                 # auxiliary variables, one per component

    W_s, MU_s, SIG2_s = [], [], []

    for it in range(n_iter + burnin):
        z       = _gibbs_categorical(u, w, mu_k, sig2_k, K, rng)
        w, _    = _gibbs_weights(z, K, alpha0, rng)

        for k in range(K):
            uk  = u[z == k]
            nk_ = len(uk)
            if nk_ < 2:
                continue
            ub  = uk.mean()
            Sk  = ((uk - ub) ** 2).sum()
            kn  = kappa0 + nk_

            # ── update xi_k (auxiliary) ──────────────────────────────────
            xi_b     = s2inv + 1.0 / (sig2_k[k] + 1e-15)
            xi_k[k]  = 1.0 / rng.gamma(1.0, 1.0 / xi_b)

            # ── update sigma_k^2 | xi_k ──────────────────────────────────
            sig_a    = nk_ / 2 + 0.5
            sig_b    = Sk  / 2 + 1.0 / xi_k[k]
            sig2_k[k] = 1.0 / rng.gamma(sig_a, 1.0 / sig_b)

            # ── update mu_k (flat prior -> posterior = N(u_bar, sigma^2/n)) ─
            mu_k[k]  = rng.normal(ub, np.sqrt(sig2_k[k] / kn))

        if it >= burnin:
            W_s.append(w.copy())
            MU_s.append(mu_k.copy())
            SIG2_s.append(sig2_k.copy())

    W_s, MU_s, SIG2_s = map(np.array, (W_s, MU_s, SIG2_s))
    W_s, MU_s, SIG2_s = _label_sort(W_s, MU_s, SIG2_s)
    return {"w": W_s, "mu": MU_s, "sig2": SIG2_s}


# ── 5. Sequential Bayesian Learning ──────────────────────────────────────────

def _posterior_to_prior(samp, K, kappa_scale=1.0, a0_base=2.0):
    """
    Convert posterior samples to Normal-IG hyperparameters for the next period.
    Uses moment matching:
      mu0   = posterior mean of mu_k
      b0_k  = posterior mean of sig2_k * (a0 - 1)   [so E[sig^2] matches]
      kappa0 proportional to 1/Var[mu_k posterior]
    """
    mu_post   = samp["mu"].mean(0)          # (K,)
    mu_var    = samp["mu"].var(0) + 1e-8    # (K,)
    sig2_post = samp["sig2"].mean(0)        # (K,)
    kappa0_k  = kappa_scale / mu_var        # higher precision -> tighter prior
    kappa0    = kappa0_k.mean()             # scalar (all components same weight)
    a0        = a0_base
    b0_k      = sig2_post * (a0 - 1)
    return {
        "kappa0": float(np.clip(kappa0, 0.1, 50.0)),
        "mu0":    mu_post.copy(),
        "a0":     a0,
        "b0_k":   b0_k.copy(),
    }


def sequential_learning(v_months: list, K: int = 2,
                        n_iter: int = 2000, burnin: int = 500,
                        seed: int = 77) -> dict:
    """
    Sequential Bayesian learning: posterior of month t becomes prior of month t+1.

    v_months : list of np.ndarray, one array per month (wind speed values)
    Returns   : dict with 'samples' (list), 'prior_evolution' (list of prior dicts)

    Learning mechanism:
      - Month 1: flat conjugate prior (kappa0=0.01, a0=1.0)
      - Month t+1: prior = moment-matched Normal-IG from month t's posterior
      - As data accumulate, posterior narrows and prior becomes more precise
    """
    rng_seeds  = np.random.default_rng(seed).integers(0, 10000, len(v_months))

    # Start with weakly informative / flat prior
    u0   = np.log(np.concatenate(v_months))
    prior = {
        "kappa0": 0.01,
        "mu0":    np.linspace(np.percentile(u0, 30), np.percentile(u0, 70), K),
        "a0":     1.0,
        "b0_k":   np.full(K, np.var(u0)),
    }

    all_samps   = []
    prior_evol  = [prior.copy()]

    for t, v_m in enumerate(v_months):
        samp = gibbs_conj(
            v_m, K=K, n_iter=n_iter, burnin=burnin,
            seed=int(rng_seeds[t]),
            kappa0=prior["kappa0"],
            a0=prior["a0"],
            b0_scalar=float(prior["b0_k"].mean()),
            mu0=prior["mu0"],
        )
        all_samps.append(samp)

        # Update prior for next month
        prior = _posterior_to_prior(samp, K)
        prior_evol.append(prior.copy())

        w_m = samp["w"].mean(0)
        print(f"    Month {t+1:02d}: n={len(v_m):4d}"
              f"  mu1={samp['mu'].mean(0)[0]:.3f}+/-{samp['mu'].std(0)[0]:.4f}"
              f"  mu2={samp['mu'].mean(0)[1]:.3f}+/-{samp['mu'].std(0)[1]:.4f}"
              f"  kappa0_next={prior['kappa0']:.2f}")

    return {"samples": all_samps, "prior_evolution": prior_evol}


# ── comparison + plots ────────────────────────────────────────────────────────

PRIOR_LABELS = {
    "conjugate":   "Conjugate (Normal-IG)",
    "empirical_bayes": "Empirical Bayes (Heuristic)",
    "jeffreys":    "Jeffreys / Uniform",
    "half_cauchy": "Half-Cauchy sigma",
}
PRIOR_COLORS = {
    "conjugate":   "#1f77b4",
    "empirical_bayes": "#ff7f0e",
    "jeffreys":    "#2ca02c",
    "half_cauchy": "#d62728",
}


def run_all_priors(v, K=2, n_iter=3000, burnin=500):
    """Run all four samplers and return a dict of results."""
    results = {}
    funcs   = {
        "conjugate":       gibbs_conj,
        "empirical_bayes": gibbs_eb,
        "jeffreys":        gibbs_jeffreys,
        "half_cauchy":     gibbs_half_cauchy,
    }
    for name, fn in funcs.items():
        print(f"    [{name}] running...", end="", flush=True)
        samp = fn(v, K=K, n_iter=n_iter, burnin=burnin)
        cr   = crps_from_samp(v, samp)
        ess_w  = ess(samp["w"][:, 0])
        ess_m1 = ess(samp["mu"][:, 0])
        ess_m2 = ess(samp["mu"][:, 1])
        results[name] = {"samp": samp, "crps": cr,
                         "ess_w": ess_w, "ess_mu1": ess_m1, "ess_mu2": ess_m2}
        print(f"  CRPS={cr:.4f}  ESS_mu1={ess_m1:.0f}", flush=True)
    return results


def plot_prior_comparison(v, results, K=2):
    """
    4-panel comparison figure:
      (a) Posterior density of mu1 (calm log-mean)
      (b) Posterior density of mu2 (windy log-mean)
      (c) Posterior density of sigma^2_1
      (d) CRPS and ESS summary
    """
    params = [
        ("mu1",    r"$\mu_1$ (calm log-mean)",    "mu",   0),
        ("mu2",    r"$\mu_2$ (windy log-mean)",   "mu",   1),
        ("sig2_1", r"$\sigma^2_1$ (calm var)",    "sig2", 0),
        ("sig2_2", r"$\sigma^2_2$ (windy var)",   "sig2", 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Prior Sensitivity: LN-Mix K=2 Posterior Marginals\n"
                 "Kelmarsh Turbine 1, January 2021", fontsize=11)

    for idx, (tag, label, key, kidx) in enumerate(params):
        ax = axes[idx // 2][idx % 2]
        for pname, res in results.items():
            chain  = res["samp"][key][:, kidx]
            xg     = np.linspace(chain.min(), chain.max(), 300)
            kde    = stats.gaussian_kde(chain)
            q025, q975 = np.quantile(chain, [0.025, 0.975])
            ax.plot(xg, kde(xg), lw=2,
                    color=PRIOR_COLORS[pname],
                    label=f"{PRIOR_LABELS[pname]}\n  mean={chain.mean():.4f}"
                          f"  CI=[{q025:.3f},{q975:.3f}]")
        ax.set(title=f"Posterior of {label}", xlabel="Value", ylabel="Density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(RES_DIR, "prior_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Prior comparison figure -> {out}")


def plot_crps_ess_table(results):
    """Bar chart: CRPS and ESS for each prior."""
    names  = list(results.keys())
    colors = [PRIOR_COLORS[n] for n in names]
    labels = [PRIOR_LABELS[n] for n in names]
    crps   = [results[n]["crps"]    for n in names]
    ess_1  = [results[n]["ess_mu1"] for n in names]
    ess_2  = [results[n]["ess_mu2"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Prior Comparison: CRPS and ESS — Kelmarsh Turbine 1, Jan 2021",
                 fontsize=10)

    x = np.arange(len(names))

    # CRPS
    axes[0].bar(x, crps, color=colors, alpha=0.85)
    axes[0].axhline(1.569, color="gray", ls="--", lw=1.2, label="Paper baseline")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="CRPS (m/s)",
                title="(a) CRPS (lower = better)")
    axes[0].legend(fontsize=8)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ESS mu1
    axes[1].bar(x, ess_1, color=colors, alpha=0.85)
    axes[1].axhline(100, color="gray", ls="--", lw=1.2, label="ESS=100 threshold")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="ESS",
                title=r"(b) ESS of $\mu_1$")
    axes[1].legend(fontsize=8)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ESS mu2
    axes[2].bar(x, ess_2, color=colors, alpha=0.85)
    axes[2].axhline(100, color="gray", ls="--", lw=1.2, label="ESS=100 threshold")
    axes[2].set(xticks=x, xticklabels=labels, ylabel="ESS",
                title=r"(c) ESS of $\mu_2$")
    axes[2].legend(fontsize=8)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(RES_DIR, "prior_crps_ess.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] CRPS/ESS figure -> {out}")


def plot_sequential_learning(seq_result, v_months, month_names):
    """
    Show how posterior uncertainty shrinks as months accumulate.
    Three panels: mu1 mean +/- 2sd, mu2 mean +/- 2sd, sigma^2_1 mean +/- 2sd.
    """
    T       = len(seq_result["samples"])
    months  = np.arange(1, T + 1)

    keys    = [("mu", 0, r"$\mu_1$ (calm log-mean)"),
               ("mu", 1, r"$\mu_2$ (windy log-mean)"),
               ("sig2", 0, r"$\sigma^2_1$ (calm var)")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Sequential Bayesian Learning: Posterior Evolution Month-by-Month\n"
                 "Kelmarsh Turbine 1, 2021 (Jan -> Jun)",
                 fontsize=10)

    for idx, (key, kidx, label) in enumerate(keys):
        ax     = axes[idx]
        means  = [s[key][:, kidx].mean() for s in seq_result["samples"]]
        stds   = [s[key][:, kidx].std()  for s in seq_result["samples"]]
        means  = np.array(means)
        stds   = np.array(stds)

        ax.plot(months, means, "o-", color="#1f77b4", lw=2, ms=6,
                label="Posterior mean")
        ax.fill_between(months, means - 2 * stds, means + 2 * stds,
                        alpha=0.2, color="#1f77b4", label="+/-2 SD")
        ax.set(xticks=months, xticklabels=month_names[:T],
               xlabel="Month", ylabel=label, title=f"{label}\nPosterior evolution")
        ax.legend(fontsize=8)

        # annotate how uncertainty shrinks
        ax.annotate(f"SD={stds[0]:.4f}", (1, means[0]),
                    textcoords="offset points", xytext=(5, 10),
                    fontsize=8, color="gray")
        ax.annotate(f"SD={stds[-1]:.4f}", (T, means[-1]),
                    textcoords="offset points", xytext=(5, -15),
                    fontsize=8, color="navy")

    plt.tight_layout()
    out = os.path.join(RES_DIR, "sequential_learning.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Sequential learning figure -> {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print(" Prior Sensitivity + Sequential Learning")
    print(" Kelmarsh Turbine 1 — LN-Mix K=2")
    print("=" * 62)

    # ── Part 1: Prior comparison on Jan 2021 pilot data ──
    print("\n[1] Loading Jan 2021 pilot data...")
    df = pd.read_csv(DATA_PILOT, parse_dates=["timestamp"])
    v  = df["wind_speed_avg_ms"].clip(lower=0.01).values
    print(f"    N={len(v)}, wind=[{v.min():.2f}, {v.max():.2f}] m/s")

    print("\n[2] Running 4 prior configurations (K=2, iter=3000, burnin=500)...")
    results = run_all_priors(v, K=2, n_iter=3000, burnin=500)

    print("\n    Summary:")
    print(f"    {'Prior':<30} {'CRPS':>8}  {'ESS_w1':>8}  {'ESS_mu1':>8}  {'ESS_mu2':>8}")
    print("    " + "-" * 64)
    for pname, res in results.items():
        print(f"    {PRIOR_LABELS[pname]:<30} {res['crps']:>8.4f}"
              f"  {res['ess_w']:>8.0f}  {res['ess_mu1']:>8.0f}  {res['ess_mu2']:>8.0f}")

    print("\n    Posterior mean comparison:")
    print(f"    {'Prior':<30} {'mu1_mean':>10} {'mu1_sd':>8} {'mu2_mean':>10} "
          f"{'mu2_sd':>8} {'sig2_1':>8} {'sig2_2':>8}")
    print("    " + "-" * 78)
    for pname, res in results.items():
        s = res["samp"]
        print(f"    {PRIOR_LABELS[pname]:<30}"
              f" {s['mu'][:,0].mean():>10.4f} {s['mu'][:,0].std():>8.4f}"
              f" {s['mu'][:,1].mean():>10.4f} {s['mu'][:,1].std():>8.4f}"
              f" {s['sig2'][:,0].mean():>8.4f} {s['sig2'][:,1].mean():>8.4f}")

    os.makedirs(RES_DIR, exist_ok=True)
    print("\n[3] Plotting prior comparison figures...")
    plot_prior_comparison(v, results)
    plot_crps_ess_table(results)

    # ── Part 2: Sequential Bayesian learning on 2021 months ──
    print("\n[4] Loading full dataset for sequential learning (2021)...")
    df_full = pd.read_parquet(DATA_FULL)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    df_2021 = df_full[df_full["timestamp"].dt.year == 2021].copy()
    df_2021["month"] = df_2021["timestamp"].dt.month

    v_months    = []
    month_names = []
    months_avail = sorted(df_2021["month"].unique())
    for m in months_avail:
        v_m = df_2021[df_2021["month"] == m]["wind_speed_ms"].values
        v_months.append(v_m)
        month_names.append(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][m - 1])
    print(f"    Months: {month_names}  sizes: {[len(x) for x in v_months]}")

    print("\n[5] Sequential Bayesian learning (conjugate, 2021 Jan->Jun)...")
    seq = sequential_learning(v_months, K=2, n_iter=2000, burnin=500)

    print("\n[6] Plotting sequential learning figure...")
    plot_sequential_learning(seq, v_months, month_names)

    print("\n" + "=" * 62)
    print(" Done!")
    print("=" * 62)


if __name__ == "__main__":
    main()
