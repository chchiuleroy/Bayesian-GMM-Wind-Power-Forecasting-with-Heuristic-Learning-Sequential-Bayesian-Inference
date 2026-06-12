#!/usr/bin/env python3
"""
wind_gmm_bayes_full.py
----------------------
Full Bayesian GMM wind-power forecasting pipeline on the complete
Kelmarsh Turbine-1 dataset (2016-01 to 2021-06, ~281,900 rows).

Extends the Jan-2021 pilot (wind_gmm_bayes.py) with:
  - Seasonal leave-one-year-out CRPS: train on same calendar month from
    prior years, test on each month in 2021.
  - LN-Mix K=2, K=3 vs Weibull BIC and CRPS.
  - Monthly BIC plot: identify which months benefit from bimodal LN-Mix.
  - Seasonal wind-speed violin + model overlay.

Outputs:
  results/full_crps_monthly.png   -- main comparison figure
  results/full_seasonal_violin.png -- seasonal distribution figure
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# Bayesian Gibbs inference (sibling module)
sys.path.insert(0, os.path.dirname(__file__))
from wind_gmm_priors import (gibbs_eb, gibbs_conj, crps_from_samp,
                               _posterior_to_prior)

DATA    = os.path.join(os.path.dirname(__file__), "data", "kelmarsh_turbine1_all.parquet")
RES_DIR = os.path.join(os.path.dirname(__file__), "results")
RATED   = 2050.0   # Senvion MM92 rated power (kW)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── helpers ──────────────────────────────────────────────────────────────────

def crps_from_samples(v_test: np.ndarray, v_ens: np.ndarray) -> float:
    """CRPS = E|X-y| - 0.5*E|X-X'|  (unbiased MC estimator)."""
    spread = 0.5 * np.abs(v_ens[:, None] - v_ens[None, :]).mean()
    mae    = np.abs(v_test[:, None] - v_ens[None, :]).mean(1)
    return float((mae - spread).mean())


def crps_lnmix_em(v_test: np.ndarray, gm: GaussianMixture,
                  n_draw: int = 1000, seed: int = 1) -> float:
    """CRPS for LN-Mix using EM point estimate (fast, no posterior uncertainty)."""
    rng   = np.random.default_rng(seed)
    K     = gm.n_components
    w     = gm.weights_
    mu    = gm.means_.flatten()
    sig   = np.sqrt(gm.covariances_[:, 0, 0])
    cumW  = np.cumsum(w)
    k_idx = np.searchsorted(cumW, rng.random(n_draw)).clip(0, K - 1)
    v_ens = np.exp(rng.normal(mu[k_idx], sig[k_idx]))
    return crps_from_samples(v_test, v_ens)


def crps_weibull(v_test: np.ndarray, c: float, sc: float,
                 n_draw: int = 1000, seed: int = 2) -> float:
    """CRPS for Weibull distribution."""
    rng   = np.random.default_rng(seed)
    v_ens = stats.weibull_min.rvs(
        c, 0, sc, size=n_draw,
        random_state=int(rng.integers(2 ** 31 - 1))
    )
    return crps_from_samples(v_test, v_ens)


def fit_lnmix(v: np.ndarray, K: int = 2, n_init: int = 5) -> GaussianMixture:
    u  = np.log(v).reshape(-1, 1)
    gm = GaussianMixture(K, covariance_type="full",
                         random_state=42, n_init=n_init,
                         max_iter=300)
    gm.fit(u)
    return gm


def bic_lnmix(v: np.ndarray, gm: GaussianMixture) -> float:
    n  = len(v)
    K  = gm.n_components
    u  = np.log(v).reshape(-1, 1)
    ll = n * gm.score(u) - np.log(v).sum()   # Jacobian correction
    n_params = K * 2 + (K - 1)
    return n_params * np.log(n) - 2 * ll


def bic_weibull(v: np.ndarray, c: float, sc: float) -> float:
    n  = len(v)
    ll = stats.weibull_min.logpdf(v, c, 0, sc).sum()
    return 2 * np.log(n) - 2 * ll


# ── seasonal monthly OOS validation ──────────────────────────────────────────

def seasonal_oos_table(df: pd.DataFrame,
                       train_years: tuple = (2016, 2020),
                       test_year:   int   = 2021) -> pd.DataFrame:
    """
    For each month M available in test_year, train LN-Mix K=2,K=3 and Weibull
    on data from the same calendar month in train_years.
    Returns a DataFrame with per-month CRPS and BIC for each model.
    """
    df = df.copy()
    df["year"]  = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    records = []
    for m in range(1, 13):
        mask_train = df["year"].between(*train_years) & (df["month"] == m)
        mask_test  = (df["year"] == test_year)        & (df["month"] == m)

        if not mask_test.any():
            continue

        v_train = df.loc[mask_train, "wind_speed_ms"].values
        v_test  = df.loc[mask_test,  "wind_speed_ms"].values

        if len(v_train) < 200:
            print(f"  Month {m:02d}: insufficient training data ({len(v_train)}), skip")
            continue

        # Weibull
        c, _, sc    = stats.weibull_min.fit(v_train, floc=0)
        crps_wbl    = crps_weibull(v_test, c, sc)
        bic_wbl     = bic_weibull(v_train, c, sc)

        # LN-Mix K=2
        gm2         = fit_lnmix(v_train, K=2)
        crps_ln2    = crps_lnmix_em(v_test, gm2)
        bic_ln2     = bic_lnmix(v_train, gm2)

        # LN-Mix K=3
        gm3         = fit_lnmix(v_train, K=3)
        crps_ln3    = crps_lnmix_em(v_test, gm3)
        bic_ln3     = bic_lnmix(v_train, gm3)

        records.append(dict(
            month       = m,
            month_name  = MONTH_NAMES[m - 1],
            n_train     = int(len(v_train)),
            n_test      = int(len(v_test)),
            crps_weibull= round(crps_wbl, 4),
            crps_lnk2   = round(crps_ln2, 4),
            crps_lnk3   = round(crps_ln3, 4),
            bic_weibull = round(bic_wbl,  1),
            bic_lnk2    = round(bic_ln2,  1),
            bic_lnk3    = round(bic_ln3,  1),
            dbic_k2     = round(bic_ln2 - bic_wbl, 1),  # <0 means LN-Mix wins
            dbic_k3     = round(bic_ln3 - bic_wbl, 1),
        ))

        win2 = "LN2" if bic_ln2 < bic_wbl else "WBL"
        win3 = "LN3" if bic_ln3 < bic_wbl else "WBL"
        print(f"  Month {MONTH_NAMES[m-1]:>3}: "
              f"CRPS Wbl={crps_wbl:.4f}  LN2={crps_ln2:.4f}  LN3={crps_ln3:.4f}"
              f"  BIC winner: {win2}(K=2) {win3}(K=3)")

    return pd.DataFrame(records)


# ── Gibbs OOS: EB + Sequential EB ────────────────────────────────────────────

def gibbs_seasonal_oos_table(df: pd.DataFrame,
                              K: int   = 2,
                              n_iter:  int = 1500,
                              burnin:  int = 300,
                              train_years: tuple = (2016, 2020),
                              test_year:   int   = 2021) -> pd.DataFrame:
    """
    OOS CRPS for two Gibbs-based inference modes:

      EB Gibbs     — fresh Empirical-Bayes prior each month (EM → prior, no carry-over)
      Sequential EB — prior from month M-1 posterior rolls into month M
                       (posterior-to-prior moment matching, accumulated knowledge)

    Both methods train on the same calendar month from train_years and test
    on test_year.  Returns a DataFrame with crps_eb and crps_seq per month.
    """
    df = df.copy()
    df["year"]  = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    # flat starting prior for sequential learning
    u_all = np.log(
        df[df["year"].between(*train_years)]["wind_speed_ms"].clip(lower=0.01).values
    )
    seq_prior = {
        "kappa0": 0.01,
        "mu0":    np.linspace(np.percentile(u_all, 30), np.percentile(u_all, 70), K),
        "a0":     1.0,
        "b0_k":   np.full(K, np.var(u_all)),
    }

    records = []
    for m in range(1, 13):
        mask_train = df["year"].between(*train_years) & (df["month"] == m)
        mask_test  = (df["year"] == test_year)        & (df["month"] == m)
        if not mask_test.any():
            continue

        v_train = df.loc[mask_train, "wind_speed_ms"].values
        v_test  = df.loc[mask_test,  "wind_speed_ms"].values
        if len(v_train) < 200:
            continue

        # ── EB Gibbs: fresh EM prior each month ──────────────────────────────
        samp_eb  = gibbs_eb(v_train, K=K, n_iter=n_iter, burnin=burnin,
                            seed=200 + m)
        crps_eb  = crps_from_samp(v_test, samp_eb)

        # ── Sequential EB: carry prior forward ───────────────────────────────
        samp_seq = gibbs_conj(
            v_train, K=K, n_iter=n_iter, burnin=burnin,
            seed=300 + m,
            kappa0     = seq_prior["kappa0"],
            a0         = seq_prior["a0"],
            b0_scalar  = float(seq_prior["b0_k"].mean()),
            mu0        = seq_prior["mu0"],
        )
        crps_seq = crps_from_samp(v_test, samp_seq)

        # update prior for next month
        seq_prior = _posterior_to_prior(samp_seq, K)

        records.append(dict(
            month      = m,
            month_name = MONTH_NAMES[m - 1],
            crps_eb    = round(crps_eb,  4),
            crps_seq   = round(crps_seq, 4),
        ))
        print(f"  Month {MONTH_NAMES[m-1]:>3}: "
              f"CRPS EB={crps_eb:.4f}  Seq={crps_seq:.4f}  "
              f"(next kappa0={seq_prior['kappa0']:.2f})")

    return pd.DataFrame(records)


# ── seasonal violin + model overlay ──────────────────────────────────────────

def plot_seasonal_violin(df: pd.DataFrame, train_years: tuple = (2016, 2020)):
    df = df.copy()
    df["month"] = df["timestamp"].dt.month
    df["year"]  = df["timestamp"].dt.year
    df_train    = df[df["year"].between(*train_years)]

    months_avail = sorted(df_train["month"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey=True)
    fig.suptitle(
        "Kelmarsh Turbine 1 — Monthly Wind Speed Distribution (2016-2020)\n"
        "with Weibull MLE (red dashed) and LN-Mix K=2/K=3 (green/blue)",
        fontsize=11
    )
    vg = np.linspace(0.1, 25, 400)

    for idx, m in enumerate(range(1, 13)):
        ax   = axes[idx // 4][idx % 4]
        v_m  = df_train[df_train["month"] == m]["wind_speed_ms"].values

        if len(v_m) < 50:
            ax.set_title(MONTH_NAMES[m - 1])
            ax.axis("off")
            continue

        ax.hist(v_m, bins=50, density=True, color="steelblue",
                alpha=0.35, label=f"N={len(v_m):,}")

        # Weibull
        c, _, sc = stats.weibull_min.fit(v_m, floc=0)
        ax.plot(vg, stats.weibull_min.pdf(vg, c, 0, sc),
                "r--", lw=1.8, label=f"Wbl k={c:.2f}")

        # LN-Mix K=2
        gm2 = fit_lnmix(v_m, K=2)
        mu2 = gm2.means_.flatten()
        w2  = gm2.weights_
        s2  = np.sqrt(gm2.covariances_[:, 0, 0])
        pdf2 = sum(w2[k] * stats.lognorm.pdf(vg, s=s2[k], scale=np.exp(mu2[k]))
                   for k in range(2))
        ax.plot(vg, pdf2, "g-", lw=1.8, label="LN K=2")

        # LN-Mix K=3
        gm3 = fit_lnmix(v_m, K=3)
        mu3 = gm3.means_.flatten()
        w3  = gm3.weights_
        s3  = np.sqrt(gm3.covariances_[:, 0, 0])
        pdf3 = sum(w3[k] * stats.lognorm.pdf(vg, s=s3[k], scale=np.exp(mu3[k]))
                   for k in range(3))
        ax.plot(vg, pdf3, "b-", lw=1.2, alpha=0.8, label="LN K=3")

        ax.set_title(f"{MONTH_NAMES[m-1]}  (N={len(v_m):,})", fontsize=9)
        ax.set_xlim(0, 22)
        ax.set_xlabel("m/s", fontsize=7)
        if idx % 4 == 0:
            ax.set_ylabel("Density", fontsize=7)
        if idx == 0:
            ax.legend(fontsize=6)

    plt.tight_layout()
    out = os.path.join(RES_DIR, "full_seasonal_violin.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Seasonal figure -> {out}")


# ── main comparison plot ──────────────────────────────────────────────────────

def plot_crps_comparison(tbl: pd.DataFrame):
    has_gibbs = "crps_eb" in tbl.columns and "crps_seq" in tbl.columns

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Kelmarsh Turbine 1 — OOS Validation 2021\n"
        "(trained on same calendar month from 2016-2020)",
        fontsize=11
    )

    x     = np.arange(len(tbl))
    names = tbl["month_name"].tolist()

    # ── (a) CRPS by month ──
    ax = axes[0]
    if has_gibbs:
        w = 0.15
        ax.bar(x - 2*w, tbl["crps_weibull"], w, label="Weibull",         color="#d62728", alpha=0.85)
        ax.bar(x -   w, tbl["crps_lnk2"],   w, label="LN-Mix K=2 (EM)", color="#2ca02c", alpha=0.85)
        ax.bar(x,       tbl["crps_lnk3"],   w, label="LN-Mix K=3 (EM)", color="#1f77b4", alpha=0.85)
        ax.bar(x +   w, tbl["crps_eb"],     w, label="EB Gibbs",         color="#ff7f0e", alpha=0.85)
        ax.bar(x + 2*w, tbl["crps_seq"],    w, label="Sequential EB",    color="#9467bd", alpha=0.85)
    else:
        w = 0.25
        ax.bar(x - w, tbl["crps_weibull"], w, label="Weibull",    color="#d62728", alpha=0.85)
        ax.bar(x,     tbl["crps_lnk2"],   w, label="LN-Mix K=2", color="#2ca02c", alpha=0.85)
        ax.bar(x + w, tbl["crps_lnk3"],   w, label="LN-Mix K=3", color="#1f77b4", alpha=0.85)

    ax.axhline(1.569, color="gray", ls="--", lw=1.2, label="Paper baseline (1.569)")
    ax.set(xticks=x, xticklabels=names, ylabel="CRPS (m/s)",
           title="(a) Out-of-Sample CRPS per Month")
    ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── (b) delta-BIC by month ──
    ax = axes[1]
    w2 = 0.25
    ax.bar(x - w2/2, tbl["dbic_k2"], w2, color="#2ca02c", alpha=0.85, label="dBIC K=2")
    ax.bar(x + w2/2, tbl["dbic_k3"], w2, color="#1f77b4", alpha=0.85, label="dBIC K=3")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-10, color="gray", ls=":", lw=1, label="dBIC=-10 (decisive win)")
    ax.set(xticks=x, xticklabels=names,
           ylabel="dBIC = BIC(LN-Mix) - BIC(Weibull)",
           title="(b) BIC Advantage of LN-Mix over Weibull\n(< 0 = LN-Mix better)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(RES_DIR, "full_crps_monthly.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] CRPS comparison figure -> {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Full GMM Validation — Kelmarsh Turbine 1  (2016-2021)")
    print("=" * 60)

    print("\n[1] Loading parquet...")
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"    N = {len(df):,}  |  "
          f"years {df['timestamp'].dt.year.min()}-{df['timestamp'].dt.year.max()}")
    print(f"    Wind: [{df['wind_speed_ms'].min():.2f}, {df['wind_speed_ms'].max():.2f}] m/s")
    print(f"    Power: [{df['power_kw'].min():.0f}, {df['power_kw'].max():.0f}] kW")

    print("\n[2] Seasonal monthly OOS validation (test year=2021)...")
    tbl = seasonal_oos_table(df, train_years=(2016, 2020), test_year=2021)

    print("\n  --- Summary Table ---")
    print(f"  {'Month':<5} {'N_train':>8} {'N_test':>7} "
          f"{'CRPS_Wbl':>10} {'CRPS_LN2':>10} {'CRPS_LN3':>10} "
          f"{'dBIC_K2':>9} {'dBIC_K3':>9}")
    print("  " + "-" * 78)
    for _, r in tbl.iterrows():
        win2 = " *" if r["dbic_k2"] < -10 else ""
        win3 = " *" if r["dbic_k3"] < -10 else ""
        print(f"  {r['month_name']:<5} {r['n_train']:>8,} {r['n_test']:>7,} "
              f"{r['crps_weibull']:>10.4f} {r['crps_lnk2']:>10.4f} {r['crps_lnk3']:>10.4f} "
              f"{r['dbic_k2']:>+9.1f}{win2} {r['dbic_k3']:>+9.1f}{win3}")

    avg_wbl  = tbl["crps_weibull"].mean()
    avg_ln2  = tbl["crps_lnk2"].mean()
    avg_ln3  = tbl["crps_lnk3"].mean()
    print(f"\n  Average CRPS:  Weibull={avg_wbl:.4f}  LN-Mix K=2={avg_ln2:.4f}  LN-Mix K=3={avg_ln3:.4f}")

    months_ln2_wins = (tbl["dbic_k2"] < 0).sum()
    months_ln3_wins = (tbl["dbic_k3"] < 0).sum()
    print(f"  BIC winner: LN-Mix K=2 wins in {months_ln2_wins}/{len(tbl)} months, "
          f"LN-Mix K=3 wins in {months_ln3_wins}/{len(tbl)} months")

    os.makedirs(RES_DIR, exist_ok=True)

    print("\n[3] Gibbs OOS: Empirical Bayes + Sequential EB (K=2, iter=1500, burnin=300)...")
    tbl_gibbs = gibbs_seasonal_oos_table(df, K=2, n_iter=1500, burnin=300,
                                          train_years=(2016, 2020), test_year=2021)

    # Merge into main table (inner join on month)
    tbl = tbl.merge(tbl_gibbs[["month", "crps_eb", "crps_seq"]], on="month", how="left")

    avg_eb  = tbl["crps_eb"].mean()
    avg_seq = tbl["crps_seq"].mean()
    print(f"\n  Average CRPS:  EB Gibbs={avg_eb:.4f}  Sequential EB={avg_seq:.4f}")

    print("\n  --- 5-Method CRPS Summary ---")
    print(f"  {'Month':<5} {'Weibull':>9} {'LN2_EM':>9} {'LN3_EM':>9} "
          f"{'EB_Gibbs':>10} {'Seq_EB':>8}")
    print("  " + "-" * 56)
    for _, r in tbl.iterrows():
        print(f"  {r['month_name']:<5} "
              f"{r['crps_weibull']:>9.4f} {r['crps_lnk2']:>9.4f} {r['crps_lnk3']:>9.4f} "
              f"{r['crps_eb']:>10.4f} {r['crps_seq']:>8.4f}")
    print(f"  {'Avg':<5} "
          f"{avg_wbl:>9.4f} {avg_ln2:>9.4f} {avg_ln3:>9.4f} "
          f"{avg_eb:>10.4f} {avg_seq:>8.4f}")

    print("\n[4] Plotting CRPS comparison figure (5 methods)...")
    plot_crps_comparison(tbl)

    print("\n[5] Plotting seasonal wind speed distribution...")
    plot_seasonal_violin(df)

    # Save table to CSV
    csv_out = os.path.join(RES_DIR, "oos_crps_table.csv")
    tbl.to_csv(csv_out, index=False)
    print(f"\n[OK] Results table -> {csv_out}")

    print("\n" + "=" * 60)
    print(" Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
