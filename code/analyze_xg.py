"""Carry-vs-dump gap with expected goals (xG) as the outcome: regression, double ML and sensitivity.

Same population, covariates, trimming, cross-fitting and bootstrap as analyze.py / analyze_final.py.
The only change is the outcome: y_xg = summed LightGBM xG of the entering team's shot attempts in
the 20 s after the entry (blocked shots count 0), built by build_entries.py when it is given the
xG predictions from xg_train.py. The xG model is treated as fixed (its own estimation error is not
propagated; it was fitted on 454,147 NHL shots, so that error is small next to the game bootstrap).

1. Raw gap and ratio, interacted regression ATE, DML (AIPW, LightGBM nuisances tuned for the xG
   outcome by game-grouped CV), linear-nuisance DML check, all with the whole-game bootstrap that
   reruns trimming and the DML fit in every resample.
2. Placebos: either-team and own-team xG in the 20 s BEFORE the entry (interacted regression).
3. Sensitivity to an unmeasured confounder (Chernozhukov et al. 2022): robustness values,
   benchmarks from dropping observed covariate groups, bound contour.
4. Robustness: logistic-xG outcome, goals, and xG per unblocked shot (shot quality).
5. Failed carries, as in analyze_final.py but in xG: decision-level gap per carry ATTEMPT,
   s * E[xG | completed carry] + (1 - s) * f - E[xG | dump] (dump mean adjusted by the DML ATE),
   and the break-even carry success rate, with f = 0 and with f = observed xG after the
   failed-carry proxies (xg_for from failed_carries.py run with the xG predictions).

Usage: python analyze_xg.py <entries_csv_with_xg> <failed_carries_csv_with_xg> <out_dir> [n_boot]
Outputs (in <out_dir>): xg_outcome_table.csv, xg_outcome_summary.json, xg_outcome_bootstrap.csv,
    xg_dml_lgbm_tuning.csv, xg_sensitivity_benchmarks.csv, xg_sensitivity_grid.npz,
    xg_failed_carries.csv, fig_xg_sensitivity.png, fig_xg_estimates.png, fig_xg_failed_carries.png
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import design, pipeline
from analyze_final import (BENCH_GROUPS, break_even, clean_cols, decision_gap, fit_dml, lgbm, pick_lgbm,
                           trimmed_frame)

warnings.filterwarnings("ignore")
OUTCOME = "y_xg"


def with_outcome(df, col):
    """Copy of the entries with `col` as y, so the shot-count functions run unchanged."""
    out = df.copy()
    out["y"] = out[col].astype(float)
    return out


def xg_per_shot(d):
    """Mean xG per unblocked shot attempt after carried minus after dumped entries."""
    c, u = d[d.carry == 1], d[d.carry == 0]
    return c.y_xg.sum() / c.n_unblocked.sum() - u.y_xg.sum() / u.n_unblocked.sum()


def break_even_floor(mu1, mu0, f):
    """Break-even success rate, 0 when a failed carry already beats a dump (f >= mu0)."""
    return max(0.0, (mu0 - f) / (mu1 - f)) if mu1 > f else np.nan


def sensitivity(fit, theta):
    """Robustness values, benchmark table and lower-bound grid for a fitted DoubleMLIRM."""
    fit.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    sp = fit.sensitivity_params
    rv, rva = float(sp["rv"][0]), float(sp["rva"][0])
    bench = []
    for name, cols in BENCH_GROUPS.items():
        bm = fit.sensitivity_benchmark(benchmarking_set=cols)
        cf_y, cf_d, rho = (float(bm.loc["d", k]) for k in ("cf_y", "cf_d", "rho"))
        row = dict(benchmark=name, cf_y=cf_y, cf_d=cf_d, rho=rho,
                   delta_theta=float(bm.loc["d", "delta_theta"]))
        for k in (1, 2, 3):
            fit.sensitivity_analysis(cf_y=min(k * cf_y, 0.99), cf_d=min(k * cf_d, 0.99), rho=1.0)
            row[f"lower_bound_{k}x_rho1"] = float(fit.sensitivity_params["theta"]["lower"][0])
        bench.append(row)
    grid = np.linspace(0.001, 0.30, 60)
    lower = np.empty((len(grid), len(grid)))
    for i, cd in enumerate(grid):
        for j, cy in enumerate(grid):
            fit.sensitivity_analysis(cf_y=cy, cf_d=cd, rho=1.0)
            lower[i, j] = fit.sensitivity_params["theta"]["lower"][0]
    return rv, rva, pd.DataFrame(bench), grid, lower


def main(entries_csv, fc_csv, out_dir, B=1000):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(entries_csv)
    if OUTCOME not in raw.columns:
        sys.exit("No y_xg column: rebuild entries with build_entries.py ... <xg_predictions_csv>")
    fc = pd.read_csv(fc_csv)
    if "xg_for" not in fc.columns:
        sys.exit("No xg_for column: rerun failed_carries.py ... <xg_predictions_csv>")
    raw["orig_game"] = raw["game"]
    df = with_outcome(raw, OUTCOME)

    # ---------- 1. point estimates ----------
    dt, point = trimmed_frame(df)
    X = clean_cols(design(dt))
    y, d = dt.y.values, dt.carry.values
    prm_m, prm_g, tune = pick_lgbm(X, y, d, dt.orig_game.values)
    tune.to_csv(f"{out_dir}/xg_dml_lgbm_tuning.csv", index=False)
    ml_m, ml_g = lgbm(prm_m, "clf"), lgbm(prm_g, "reg")
    main_fit = fit_dml(dt, ml_g, ml_m, seeds=range(10))
    lin_fit = fit_dml(dt, make_pipeline(StandardScaler(), LinearRegression()),
                      make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), seeds=range(10))
    theta, theta_lin = float(main_fit.coef[0]), float(lin_fit.coef[0])
    mu1 = float(dt[dt.carry == 1].y.mean())

    # robustness outcomes (point estimates, same trimmed rows and LightGBM settings)
    dt_logit = with_outcome(dt, "y_xg_logit")
    theta_logit = float(fit_dml(dt_logit, ml_g, ml_m, seeds=range(10)).coef[0])
    dt_goals = with_outcome(dt, "y_goals")
    theta_goals = float(fit_dml(dt_goals, ml_g, ml_m, seeds=range(10)).coef[0])

    # ---------- bootstrap ----------
    rng = np.random.default_rng(11)          # same draws as analyze_final.py
    games = df.orig_game.unique()
    fc_n = {defn: fc[fc.definition == defn].groupby("game").size() for defn in ("narrow", "broad")}
    fc_f = {defn: fc[fc.definition == defn].groupby("game").xg_for.sum() for defn in ("narrow", "broad")}
    boot = []
    for b in range(B):
        draw = rng.choice(games, size=len(games), replace=True)
        bdf = pd.concat([df[df.orig_game == g] for g in draw], ignore_index=True)
        try:
            bdt, bp = trimmed_frame(bdf, seed=b)
            bfit = fit_dml(bdt, ml_g, ml_m, seeds=(b,))
        except ValueError:
            continue
        bth = float(bfit.coef[0]); bmu1 = bdt[bdt.carry == 1].y.mean()
        row = dict(raw_all=bp["raw_all"], ratio_all=bp["ratio_all"], lin_ate=bp["lin_ate"],
                   dml_ate=bth, dml_ratio=bmu1 / (bmu1 - bth), xg_per_shot_diff=xg_per_shot(bdf))
        n_carry = (bdf.carry == 1).sum()
        for defn in ("narrow", "broad"):    # same steps as analyze_final.py, xG instead of shots
            nf = sum(fc_n[defn].get(g, 0) for g in draw)
            ff = sum(fc_f[defn].get(g, 0) for g in draw) / nf if nf else 0.0
            s = n_carry / (n_carry + nf)
            row[f"s_{defn}"] = s
            row[f"gap_dec_{defn}_f0"] = decision_gap(bmu1, bmu1 - bth, s, 0.0)
            row[f"gap_dec_{defn}_fobs"] = decision_gap(bmu1, bmu1 - bth, s, ff)
            row[f"break_even_{defn}_fobs"] = break_even_floor(bmu1, bmu1 - bth, ff)
        row["break_even_f0"] = break_even(bmu1, bmu1 - bth, 0.0)
        boot.append(row)
        if b % 100 == 0:
            print(f"bootstrap {b}/{B}  {time.time() - t0:.0f}s", flush=True)
    boot = pd.DataFrame(boot)
    boot.to_csv(f"{out_dir}/xg_outcome_bootstrap.csv", index=False)
    ci = lambda c, s=boot: [float(np.nanpercentile(s[c], 2.5)), float(np.nanpercentile(s[c], 97.5))]

    # ---------- 2. placebos (interacted regression, as in analyze.py) ----------
    plac = {}
    for name, col in [("either_team", "y_pre_xg"), ("own_team", "y_pre_own_xg")]:
        plac[name] = dict(lin_ate=float(pipeline(raw, seed=0, outcome=col)["lin_ate"]), boot=[])
    prng = np.random.default_rng(7)
    for b in range(min(B, 500)):
        draw = prng.choice(games, size=len(games), replace=True)
        bdf = pd.concat([raw[raw.orig_game == g] for g in draw], ignore_index=True)
        try:
            vals = {n: pipeline(bdf, seed=b, outcome=c)["lin_ate"]
                    for n, c in [("either_team", "y_pre_xg"), ("own_team", "y_pre_own_xg")]}
        except ValueError:
            continue
        for n, v in vals.items():
            plac[n]["boot"].append(v)
    for n in plac:
        bb = plac[n].pop("boot")
        plac[n]["ci"] = [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]

    # ---------- 3. sensitivity ----------
    rv, rva, bench, grid, lower = sensitivity(main_fit, theta)
    bench.to_csv(f"{out_dir}/xg_sensitivity_benchmarks.csv", index=False)
    np.savez(f"{out_dir}/xg_sensitivity_grid.npz", grid=grid, lower=lower)

    # ---------- 5. failed carries ----------
    mu0_adj = mu1 - theta
    n_carry = int((raw.carry == 1).sum())
    fc_tab = []
    for defn in ("narrow", "broad"):
        sub = fc[fc.definition == defn]
        s = n_carry / (n_carry + len(sub))
        f_obs = float(sub.xg_for.mean()); a_obs = float(sub.xg_against.mean())
        fc_tab.append(dict(definition=defn, n_failed=len(sub), success_rate=s, success_ci=ci(f"s_{defn}"),
                           xg_for_after=f_obs, xg_against_after=a_obs,
                           xg_against_per_attempt=(1 - s) * a_obs,
                           gap_dec_f0=decision_gap(mu1, mu0_adj, s, 0.0), gap_dec_f0_ci=ci(f"gap_dec_{defn}_f0"),
                           gap_dec_fobs=decision_gap(mu1, mu0_adj, s, f_obs),
                           gap_dec_fobs_ci=ci(f"gap_dec_{defn}_fobs"),
                           break_even_fobs=break_even_floor(mu1, mu0_adj, f_obs),
                           break_even_fobs_ci=ci(f"break_even_{defn}_fobs")))
    pd.DataFrame(fc_tab).to_csv(f"{out_dir}/xg_failed_carries.csv", index=False)
    be = dict(raw=break_even(raw[raw.carry == 1].y_xg.mean(), raw[raw.carry == 0].y_xg.mean(), 0.0),
              adjusted_dml=break_even(mu1, mu0_adj, 0.0), adjusted_dml_ci=ci("break_even_f0"))

    # ---------- outputs ----------
    n_t = len(dt)
    tbl = pd.DataFrame([
        ["1. Raw difference, all entries", point["raw_all"], *ci("raw_all"), len(df)],
        ["1r. Raw ratio, all entries (carried / dumped)", point["ratio_all"], *ci("ratio_all"), len(df)],
        ["2. Interacted regression (ATE, trimmed)", point["lin_ate"], *ci("lin_ate"), n_t],
        ["3. Double ML, AIPW score (ATE, trimmed), LightGBM", theta, *ci("dml_ate"), n_t],
        ["3r. Double ML ratio, carried / adjusted dumped", mu1 / (mu1 - theta), *ci("dml_ratio"), n_t],
        ["3b. Double ML, linear nuisances (check)", theta_lin, np.nan, np.nan, n_t],
        ["4a. Double ML, logistic-xG outcome (check)", theta_logit, np.nan, np.nan, n_t],
        ["4b. Double ML, goals outcome (check)", theta_goals, np.nan, np.nan, n_t],
        ["4c. xG per unblocked shot, carried minus dumped (raw)", xg_per_shot(raw), *ci("xg_per_shot_diff"), len(df)],
        ["5a. Placebo: either team's xG 20 s before (regression)", plac["either_team"]["lin_ate"],
         *plac["either_team"]["ci"], n_t],
        ["5b. Placebo: own team's xG 20 s before (regression)", plac["own_team"]["lin_ate"],
         *plac["own_team"]["ci"], n_t],
        ["6a. Break-even carry success rate (DML-adjusted, failed carry = 0 xG)", be["adjusted_dml"],
         *be["adjusted_dml_ci"], n_t],
        ["6b. Decision-level gap per attempt, broad proxy, failed carry = 0 xG", fc_tab[1]["gap_dec_f0"],
         *fc_tab[1]["gap_dec_f0_ci"], n_t],
        ["6c. Decision-level gap per attempt, broad proxy, observed xG after failure",
         fc_tab[1]["gap_dec_fobs"], *fc_tab[1]["gap_dec_fobs_ci"], n_t],
    ], columns=["estimate", "value", "ci_low", "ci_high", "n_entries"])
    tbl.to_csv(f"{out_dir}/xg_outcome_table.csv", index=False)
    means = raw.groupby("carry")[["y", "y_xg", "y_xg_logit", "y_goals", "n_unblocked"]].mean()
    summary = dict(
        outcome="summed LightGBM xG of the entering team's shots within 20 s (blocked shots = 0)",
        means_by_carry={int(k): v for k, v in means.to_dict("index").items()},
        dml=dict(theta=theta, ci=ci("dml_ate"), se_iid=float(main_fit.se[0]),
                 boot_sd=float(boot.dml_ate.std()), n_boot_ok=len(boot),
                 theta_linear_nuisances=theta_lin, theta_logit_xg=theta_logit, theta_goals=theta_goals,
                 lgbm_carry=prm_m, lgbm_xg=prm_g, mu_carry_trimmed=mu1, mu_dump_adjusted=mu1 - theta),
        placebos=plac,
        failed_carries=fc_tab, break_even_success_rate=be,
        sensitivity=dict(rv=rv, rva=rva, note=(
            "rv: confounding strength (cf_y = cf_d) at which the bound on the ATE reaches 0, rho = 1. "
            "rva additionally accounts for sampling error with DoubleML's iid standard error.")))
    json.dump(summary, open(f"{out_dir}/xg_outcome_summary.json", "w"), indent=2, default=float)

    # ---------- figures ----------
    INK, MUTED, GRID_C, C1 = "#1f2328", "#6b7280", "#e5e7eb", "#2a78d6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=200)
    levels = [round(theta * f, 3) for f in (0.25, 0.5, 0.75)]
    cs = ax.contour(grid, grid, lower, levels=levels, colors=MUTED, linewidths=0.8)
    ax.clabel(cs, fmt="%.3f", fontsize=7)
    ax.contour(grid, grid, lower, levels=[0], colors=INK, linewidths=2)
    for _, r in bench.iterrows():
        ax.scatter(r.cf_y, r.cf_d, color=INK, s=16, zorder=3)
        if max(r.cf_y, r.cf_d) > 0.03:
            ax.annotate(r.benchmark.split(" (")[0], (r.cf_y, r.cf_d), xytext=(5, 3),
                        textcoords="offset points", fontsize=7.5, color=INK)
    ax.scatter([rv], [rv], marker="x", color=INK, s=30, zorder=3)
    ax.annotate(f"RV = {rv:.2f}", (rv, rv), xytext=(-48, -4), textcoords="offset points",
                fontsize=7.5, color=INK)
    ax.set_xlabel("Confounder strength with xG (cf_y, partial R²)")
    ax.set_ylabel("Confounder strength with carry choice (cf_d)")
    ax.set_title(f"Lower bound on the carry–dump xG gap (estimate {theta:.3f}); bold line = 0",
                 fontsize=9, loc="left", color=INK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_xg_sensitivity.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=200)
    s_grid = np.linspace(0.3, 1, 200)
    ax.plot(s_grid, decision_gap(mu1, mu0_adj, s_grid, 0.0), color=INK, lw=1.8,
            label="Failed carry gives 0 xG")
    f_b = fc_tab[1]["xg_for_after"]
    ax.plot(s_grid, decision_gap(mu1, mu0_adj, s_grid, f_b), color=INK, lw=1.2, ls="--",
            label=f"Failed carry gives {f_b:.4f} xG (broad proxy average)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(be["adjusted_dml"], color=MUTED, lw=0.8, ls=":")
    ax.text(be["adjusted_dml"] + 0.01, -0.004, f"break-even {be['adjusted_dml']:.2f}", fontsize=7.5, color=INK)
    ax.axvspan(fc_tab[1]["success_rate"], fc_tab[0]["success_rate"], color=GRID_C, zorder=0)
    ax.text(fc_tab[1]["success_rate"] - 0.01, 0.9 * decision_gap(mu1, mu0_adj, 1.0, 0.0), "proxy range",
            fontsize=7.5, color=INK, ha="right")
    ax.set_xlabel("Share of carry attempts that succeed")
    ax.set_ylabel("Gap per attempt (xG)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color=GRID_C, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_xg_failed_carries.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 2.4), dpi=200)
    rows = [("Raw, all entries", point["raw_all"], ci("raw_all")),
            ("Interacted regression\n(ATE, trimmed)", point["lin_ate"], ci("lin_ate")),
            ("Double ML (LightGBM)", theta, ci("dml_ate")),
            ("Placebo: either team,\n20 s before", plac["either_team"]["lin_ate"], plac["either_team"]["ci"])]
    for i, (lab, v, (lo, hi)) in enumerate(rows):
        yv = len(rows) - 1 - i
        ax.plot([lo, hi], [yv, yv], color=C1 if i < 3 else MUTED, lw=2)
        ax.scatter([v], [yv], color=C1 if i < 3 else MUTED, s=28, zorder=3)
        ax.text(hi + 0.001, yv, f"{v:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel("Carried minus dumped: expected goals within 20 s per entry")
    ax.set_xlim(min(r[2][0] for r in rows) - 0.005, max(r[2][1] for r in rows) + 0.02)
    ax.grid(axis="x", color=GRID_C, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_xg_estimates.png"); plt.close(fig)

    print(json.dumps(summary, indent=2, default=float))
    print(tune.round(5).to_string(index=False))
    print(bench.round(4).to_string(index=False))
    print(pd.DataFrame(fc_tab).drop(columns=[c for c in fc_tab[0] if c.endswith("_ci")]).round(4).to_string(index=False))
    print(tbl.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1000)
