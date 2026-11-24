"""Final-report analysis, run after analyze.py.

1. Double ML: AIPW / doubly robust score for the ATE (DoubleMLIRM, score="ATE") on the same
   trimmed population as the interacted regression. Nuisances e(X) = P(carry | X) and
   g(d, X) = E[shots | carry = d, X] from LightGBM, cross-fitted with folds grouped by game.
   LightGBM settings are picked once by game-grouped CV loss of the nuisance predictions
   (never by the ATE). A linear-nuisance DML (logistic + OLS) is a sanity check.
   Uncertainty: the same whole-game bootstrap as analyze.py, rerunning propensity trimming and
   the DML fit in every resample.
2. Sensitivity to an unmeasured confounder (Chernozhukov et al. 2022, DoubleML implementation):
   robustness values, a bound contour, and benchmarks from dropping observed covariate groups.
3. Failed carries: break-even carry success rate and the decision-level gap
   s * E[Y | completed carry] + (1 - s) * f - E[Y | dump], using the failed-carry proxies from
   failed_carries.py.

Usage: python analyze_final.py <entries_csv> <failed_carries_csv> <out_dir> [n_boot]
"""
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import doubleml as dml
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import N_FOLDS, NUM, design, lin_ate, pipeline

warnings.filterwarnings("ignore")
BENCH_GROUPS = {
    "Lane (distance from centre)": ["lane"],
    "Manpower (skater diff, even strength)": ["skater_diff", "even_strength"],
    "Shift time": ["shift_time"],
    "Time since faceoff": ["since_faceoff"],
    "Score difference": ["score_diff"],
}
GRID = [dict(n_estimators=n, learning_rate=0.03, num_leaves=l, min_child_samples=30,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
        for n in (50, 100, 300) for l in (2, 4, 8)]


def game_folds(games, seed):
    """(train_idx, test_idx) per fold, folds made of whole original games."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(games)))
    fold_of = dict(zip(uniq, rng.permutation(np.arange(len(uniq)) % N_FOLDS)))
    f = pd.Series(games).map(fold_of).values
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(N_FOLDS) if (f == k).any()]


def clean_cols(X):
    X = X.copy()
    X.columns = [c.replace(":", "_").replace(" ", "_") for c in X.columns]
    return X.loc[:, X.std() > 0]


def trimmed_frame(df, seed=0):
    """Same propensity model + 0.05-0.95 trimming as analyze.py, returns the kept rows."""
    p = pipeline(df, seed=seed, keep_details=True)
    return df[p["keep"]].reset_index(drop=True), p


def lgbm(params, kind):
    cls = LGBMClassifier if kind == "clf" else LGBMRegressor
    return cls(**params, verbose=-1, n_jobs=1)


def pick_lgbm(X, y, d, games):
    """Choose LightGBM settings by game-grouped CV loss of each nuisance prediction."""
    rows = []
    folds = game_folds(games, seed=123)
    for i, prm in enumerate(GRID):
        ll, mse = [], []
        for tr, te in folds:
            m = lgbm(prm, "clf").fit(X.iloc[tr], d[tr])
            ll.append(log_loss(d[te], m.predict_proba(X.iloc[te])[:, 1], labels=[0, 1]))
            Xd = X.assign(_d=d)
            g = lgbm(prm, "reg").fit(Xd.iloc[tr], y[tr])
            mse.append(mean_squared_error(y[te], g.predict(Xd.iloc[te])))
        rows.append(dict(setting=i, **{k: prm[k] for k in ("n_estimators", "num_leaves")},
                         cv_logloss_carry=np.mean(ll), cv_mse_shots=np.mean(mse)))
    t = pd.DataFrame(rows)
    return GRID[int(t.cv_logloss_carry.idxmin())], GRID[int(t.cv_mse_shots.idxmin())], t


def fit_dml(dt, ml_g, ml_m, seeds=(0,), x_drop=()):
    X = clean_cols(design(dt))
    X = X.drop(columns=[c for c in X.columns if c in x_drop])
    data = dml.DoubleMLData(X.assign(y=dt.y.values.astype(float), d=dt.carry.values),
                            y_col="y", d_cols="d", x_cols=list(X.columns))
    obj = dml.DoubleMLIRM(data, ml_g=clone(ml_g), ml_m=clone(ml_m), score="ATE",
                          trimming_threshold=0.01, draw_sample_splitting=False)
    obj.set_sample_splitting([game_folds(dt.orig_game.values, s) for s in seeds])
    obj.fit(store_predictions=True)
    return obj


def decision_gap(mu1, mu0, s, f):
    """Carry-vs-dump gap per *attempted* carry when a share 1-s of carries fail and yield f shots."""
    return s * mu1 + (1 - s) * f - mu0


def break_even(mu1, mu0, f):
    """Carry success rate at which the decision-level gap is zero (None if never)."""
    return (mu0 - f) / (mu1 - f) if mu1 > f and mu0 > f else None


def main(entries_csv, fc_csv, out_dir, B=1000):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(entries_csv)
    df["orig_game"] = df["game"]
    fc = pd.read_csv(fc_csv)
    dt, point = trimmed_frame(df)
    X = clean_cols(design(dt))
    y, d = dt.y.values.astype(float), dt.carry.values

    # ---------- 1. Double ML ----------
    prm_m, prm_g, tune = pick_lgbm(X, y, d, dt.orig_game.values)
    tune.to_csv(f"{out_dir}/dml_lgbm_tuning.csv", index=False)
    ml_m, ml_g = lgbm(prm_m, "clf"), lgbm(prm_g, "reg")
    main_fit = fit_dml(dt, ml_g, ml_m, seeds=range(10))          # 10 game-fold splits, median
    lin_fit = fit_dml(dt, make_pipeline(StandardScaler(), LinearRegression()),
                      make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                      seeds=range(10))
    theta = float(main_fit.coef[0]); theta_lin = float(lin_fit.coef[0])
    e_hat = main_fit.predictions["ml_m"][:, 0, 0]

    # ---------- bootstrap: trimming + regression + DML + decision gap per resample ----------
    rng = np.random.default_rng(11)
    games = df.orig_game.unique()
    fc_n = {defn: fc[fc.definition == defn].groupby("game").size() for defn in ("narrow", "broad")}
    fc_f = {defn: fc[fc.definition == defn].groupby("game").shots_for.sum() for defn in ("narrow", "broad")}
    boot = []
    for b in range(B):
        draw = rng.choice(games, size=len(games), replace=True)
        bdf = pd.concat([df[df.orig_game == g] for g in draw], ignore_index=True)
        try:
            bdt, bp = trimmed_frame(bdf, seed=b)
            bfit = fit_dml(bdt, ml_g, ml_m, seeds=(b,))
        except ValueError:
            continue
        row = dict(lin_ate=bp["lin_ate"], dml_ate=float(bfit.coef[0]))
        mu1 = bdt[bdt.carry == 1].y.mean()
        n_carry = (bdf.carry == 1).sum()
        for defn in ("narrow", "broad"):
            nf = sum(fc_n[defn].get(g, 0) for g in draw)
            ff = sum(fc_f[defn].get(g, 0) for g in draw) / nf if nf else 0.0
            s = n_carry / (n_carry + nf)
            row[f"s_{defn}"] = s
            row[f"gap_dec_{defn}_f0"] = decision_gap(mu1, mu1 - row["dml_ate"], s, 0.0)
            row[f"gap_dec_{defn}_fobs"] = decision_gap(mu1, mu1 - row["dml_ate"], s, ff)
        row["break_even_f0"] = break_even(mu1, mu1 - row["dml_ate"], 0.0)
        boot.append(row)
        if b % 100 == 0:
            print(f"bootstrap {b}/{B}  {time.time() - t0:.0f}s", flush=True)
    boot = pd.DataFrame(boot)
    boot.to_csv(f"{out_dir}/bootstrap_final.csv", index=False)
    ci = lambda c: [float(np.nanpercentile(boot[c], 2.5)), float(np.nanpercentile(boot[c], 97.5))]

    # ---------- 2. Sensitivity analysis ----------
    sens = {}
    main_fit.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    sp = main_fit.sensitivity_params
    sens["rv"], sens["rva"] = float(sp["rv"][0]), float(sp["rva"][0])
    bench = []
    for name, cols in BENCH_GROUPS.items():
        bm = main_fit.sensitivity_benchmark(benchmarking_set=cols)
        cf_y, cf_d, rho = (float(bm.loc["d", k]) for k in ("cf_y", "cf_d", "rho"))
        row = dict(benchmark=name, cf_y=cf_y, cf_d=cf_d, rho=rho,
                   delta_theta=float(bm.loc["d", "delta_theta"]))
        for k in (1, 2, 3):
            main_fit.sensitivity_analysis(cf_y=min(k * cf_y, 0.99), cf_d=min(k * cf_d, 0.99), rho=1.0)
            row[f"lower_bound_{k}x_rho1"] = float(main_fit.sensitivity_params["theta"]["lower"][0])
        bench.append(row)
    bench = pd.DataFrame(bench)
    bench.to_csv(f"{out_dir}/sensitivity_benchmarks.csv", index=False)

    grid = np.linspace(0.001, 0.30, 60)
    lower = np.empty((len(grid), len(grid)))
    for i, cd in enumerate(grid):
        for j, cy in enumerate(grid):
            main_fit.sensitivity_analysis(cf_y=cy, cf_d=cd, rho=1.0)
            lower[i, j] = main_fit.sensitivity_params["theta"]["lower"][0]

    np.savez(f"{out_dir}/sensitivity_grid.npz", grid=grid, lower=lower)

    # ---------- 3. Failed carries ----------
    mu1_t = float(dt[dt.carry == 1].y.mean()); mu0_adj = mu1_t - theta
    n_carry = int((df.carry == 1).sum())
    fc_tab = []
    for defn in ("narrow", "broad"):
        sub = fc[fc.definition == defn]
        s = n_carry / (n_carry + len(sub))
        f_obs = float(sub.shots_for.mean())
        fc_tab.append(dict(definition=defn, n_failed=len(sub), success_rate=s,
                           success_ci=ci(f"s_{defn}"), shots_for_after=f_obs,
                           shots_against_after=float(sub.shots_against.mean()),
                           gap_dec_f0=decision_gap(mu1_t, mu0_adj, s, 0.0),
                           gap_dec_f0_ci=ci(f"gap_dec_{defn}_f0"),
                           gap_dec_fobs=decision_gap(mu1_t, mu0_adj, s, f_obs),
                           gap_dec_fobs_ci=ci(f"gap_dec_{defn}_fobs")))
    be = dict(raw=break_even(df[df.carry == 1].y.mean(), df[df.carry == 0].y.mean(), 0.0),
              adjusted_dml=break_even(mu1_t, mu0_adj, 0.0), adjusted_dml_ci=ci("break_even_f0"))

    # ---------- outputs ----------
    tbl = pd.read_csv(f"{out_dir}/results_table.csv") if os.path.exists(f"{out_dir}/results_table.csv") else None
    rows = [["3. Double ML, AIPW score (ATE, trimmed), LightGBM", theta, *ci("dml_ate"), len(dt)],
            ["3b. Double ML, linear nuisances (check)", theta_lin, np.nan, np.nan, len(dt)]]
    new = pd.DataFrame(rows, columns=["estimate", "value", "ci_low", "ci_high", "n_entries"])
    if tbl is not None:
        tbl = pd.concat([tbl[~tbl.estimate.str.startswith("3")], new], ignore_index=True)
        tbl.to_csv(f"{out_dir}/results_table.csv", index=False)
    summary = dict(
        dml=dict(theta=theta, ci=ci("dml_ate"), dml_internal_se_iid=float(main_fit.se[0]),
                 theta_linear_nuisances=theta_lin, lin_ate_boot_ci=ci("lin_ate"),
                 n_boot_ok=len(boot), lgbm_carry=prm_m, lgbm_shots=prm_g,
                 propensity_range=[float(e_hat.min()), float(e_hat.max())]),
        sensitivity=dict(rv=sens["rv"], rva=sens["rva"], note=(
            "rv: confounding strength (cf_y = cf_d) at which the bound on the ATE reaches 0, rho = 1. "
            "rva additionally accounts for sampling error but uses DoubleML's iid standard error, "
            "not the game bootstrap.")),
        failed_carries=fc_tab, break_even_success_rate=be,
        mu_carry_trimmed=mu1_t, mu_dump_adjusted=mu0_adj)
    json.dump(summary, open(f"{out_dir}/summary_final.json", "w"), indent=2, default=float)

    # ---------- figures ----------
    INK, MUTED, GRID_C = "#1f2328", "#6b7280", "#e5e7eb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

    # contour of the lower bound
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=200)
    cs = ax.contour(grid, grid, lower, levels=[-0.1, 0.1, 0.2, 0.3], colors=MUTED, linewidths=0.8)
    ax.clabel(cs, fmt="%.1f", fontsize=7)
    ax.contour(grid, grid, lower, levels=[0], colors=INK, linewidths=2)
    small = []
    for _, r in bench.iterrows():
        ax.scatter(r.cf_y, r.cf_d, color=INK, s=16, zorder=3)
        if max(r.cf_y, r.cf_d) > 0.03:
            ax.annotate(r.benchmark.split(" (")[0], (r.cf_y, r.cf_d), xytext=(5, 3),
                        textcoords="offset points", fontsize=7.5, color=INK)
        else:
            small.append(r)
    if small:
        ax.annotate("other observed covariates", (0.002, 0.01), xytext=(10, 18),
                    textcoords="offset points", fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.scatter([sens["rv"]], [sens["rv"]], marker="x", color=INK, s=30, zorder=3)
    ax.annotate(f"RV = {sens['rv']:.2f}", (sens["rv"], sens["rv"]), xytext=(-48, -4),
                textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Confounder strength with shots (cf_y, partial R²)")
    ax.set_ylabel("Confounder strength with carry choice (cf_d)")
    ax.set_title(f"Lower bound on the carry–dump gap (estimate {theta:.2f}); bold line = 0",
                 fontsize=9, loc="left", color=INK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_sensitivity.png"); plt.close(fig)

    # decision-level gap vs carry success rate
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=200)
    s_grid = np.linspace(0.3, 1, 200)
    ax.plot(s_grid, decision_gap(mu1_t, mu0_adj, s_grid, 0.0), color=INK, lw=1.8,
            label="Failed carry gives 0 shots")
    f_b = fc_tab[1]["shots_for_after"]
    ax.plot(s_grid, decision_gap(mu1_t, mu0_adj, s_grid, f_b), color=INK, lw=1.2, ls="--",
            label=f"Failed carry gives {f_b:.2f} shots (broad proxy average)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(be["adjusted_dml"], color=MUTED, lw=0.8, ls=":")
    ax.text(be["adjusted_dml"] + 0.01, -0.08, f"break-even {be['adjusted_dml']:.2f}", fontsize=7.5, color=INK)
    ax.axvspan(fc_tab[1]["success_rate"], fc_tab[0]["success_rate"], color=GRID_C, zorder=0)
    ax.text(fc_tab[1]["success_rate"] - 0.01, 0.33, "proxy range", fontsize=7.5, color=INK, ha="right")
    ax.set_xlabel("Share of carry attempts that succeed")
    ax.set_ylabel("Gap per attempt (shots)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color=GRID_C, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_failed_carries.png"); plt.close(fig)

    print(json.dumps(summary, indent=2, default=float))
    print(tune.round(4).to_string(index=False))
    print(bench.round(3).to_string(index=False))
    if tbl is not None:
        print(tbl.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1000)
