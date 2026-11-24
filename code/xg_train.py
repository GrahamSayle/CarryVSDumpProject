"""Train expected-goals (xG) models on MoneyPuck NHL shots and apply them to Big Data Cup shots.

Run after xg_features.py:
    python code/xg_train.py out/xg_mp_2021_2025.csv out/xg_bdc_shots.csv out

Design
------
- Unblocked shots only (MoneyPuck has no blocked shots). Regular season only.
- Train: 2021-22 to 2024-25. Test: 2025-26 (never touched until the end).
- LightGBM settings are picked once on a validation split (train 2021-22 to
  2023-24, validate 2024-25) by log loss, then refit on all four training seasons.
- Three models:
    logit_simple : logistic regression on distance and |angle| (the classic baseline)
    logit_full   : logistic regression on every shared feature (+ log distance, angle^2)
    lgbm         : gradient-boosted trees on every shared feature
  MoneyPuck's own xGoal is the benchmark on the NHL test season.
- Big Data Cup: predictions for the 1,145 unblocked shots, compared with the 70
  goals. Intervals resample whole games (2,000 draws).

Outputs (in <out_dir>)
----------------------
  xg_metrics.csv          log loss, Brier, AUC, expected / observed goals, per model and data set
  xg_bdc_predictions.csv  every BDC shot with each model's xG (blank for blocked shots)
  xg_lgbm_tuning.csv      validation log loss per LightGBM setting
  xg_lgbm_importance.csv  LightGBM gain importance
  fig_xg_calibration.png  calibration: NHL test season and BDC shots

Please credit MoneyPuck.com wherever MoneyPuck data is used.
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Newer LightGBM versions warn that eval_set will be renamed; older ones lack the new name.
warnings.filterwarnings("ignore", message=".*eval_set.*")

SEED = 2112
TRAIN_SEASONS = (2021, 2022, 2023, 2024)
VALID_SEASON = 2024
TEST_SEASON = 2025
N_BOOT = 2000

SHOT_GROUPS = ["slap", "tip", "wrap"]  # reference level: wrist_snap
NUMERIC = [
    "distance", "abs_angle", "x_adj", "y_abs",
    "skaters_for", "skaters_against", "empty_net_against", "score_diff", "period", "t_game", "is_home",
    "time_since_faceoff", "shooter_toi",
    "shoot_toi_mean", "shoot_toi_min", "shoot_toi_max", "def_toi_mean", "def_toi_min", "def_toi_max",
    "toi_min_diff", "rebound", "rebound_angle_change",
]
SIMPLE = ["distance", "abs_angle"]
LGBM_GRID = [dict(num_leaves=n, min_child_samples=m) for n in (15, 31, 63) for m in (100, 400)]


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------

def base_frame(df):
    """Shared feature frame (same columns for both sources), NaN left in place."""
    X = df[[c for c in NUMERIC if c != "y_abs"]].astype(float).copy()
    X["y_abs"] = df["y_adj"].abs()
    X = X[NUMERIC]
    for g in SHOT_GROUPS:
        X[f"shot_{g}"] = (df["shot_group"] == g).astype(float)
    return X


def logit_frame(X, medians):
    """Logistic-regression version: impute medians, clip long tails, add curvature terms."""
    Z = X.fillna(medians).copy()
    for c, hi in [("time_since_faceoff", 300), ("shooter_toi", 200), ("shoot_toi_mean", 200),
                  ("shoot_toi_max", 200), ("def_toi_mean", 200), ("def_toi_max", 200),
                  ("shoot_toi_min", 200), ("def_toi_min", 200)]:
        Z[c] = Z[c].clip(upper=hi)
    Z["toi_min_diff"] = Z["toi_min_diff"].clip(-200, 200)
    Z["score_diff"] = Z["score_diff"].clip(-3, 3)
    Z["log_distance"] = np.log1p(Z["distance"])
    Z["abs_angle_sq"] = Z["abs_angle"] ** 2
    return Z


def fit_logit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def fit_lgbm(X, y, params, n_estimators, seed=SEED):
    m = LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=4, **params)
    return m.fit(X, y)


def tune_lgbm(Xtr, ytr, Xva, yva):
    """Pick num_leaves / min_child_samples and the number of trees by validation log loss."""
    rows = []
    for i, prm in enumerate(LGBM_GRID):
        m = LGBMClassifier(n_estimators=2000, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4, **prm)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
              callbacks=[__import__("lightgbm").early_stopping(100, verbose=False)])
        rows.append(dict(setting=i, **prm, best_iteration=int(m.best_iteration_),
                         valid_logloss=float(m.best_score_["valid_0"]["binary_logloss"])))
    t = pd.DataFrame(rows).sort_values("valid_logloss").reset_index(drop=True)
    best = t.iloc[0]
    return dict(num_leaves=int(best.num_leaves), min_child_samples=int(best.min_child_samples)), int(best.best_iteration), t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return dict(n_shots=int(len(y)), goals=int(y.sum()), expected_goals=float(p.sum()),
                obs_over_exp=float(y.sum() / p.sum()), log_loss=float(log_loss(y, p)),
                brier=float(brier_score_loss(y, p)), auc=float(roc_auc_score(y, p)))


def game_bootstrap(df, pred_cols, rng, n_boot=N_BOOT):
    """Resample whole games; return 95% intervals for obs/exp, log loss and AUC per model."""
    games = df["game_id"].unique()
    by_game = {g: df[df["game_id"] == g] for g in games}
    draws = {c: [] for c in pred_cols}
    for _ in range(n_boot):
        b = pd.concat([by_game[g] for g in rng.choice(games, size=len(games), replace=True)])
        if b["goal"].nunique() < 2:
            continue
        for c in pred_cols:
            m = metrics(b["goal"], b[c])
            draws[c].append((m["obs_over_exp"], m["log_loss"], m["auc"]))
    out = {}
    for c in pred_cols:
        d = np.array(draws[c])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        out[c] = dict(obs_over_exp_ci=[lo[0], hi[0]], log_loss_ci=[lo[1], hi[1]], auc_ci=[lo[2], hi[2]])
    return out


def reliability(y, p, n_bins):
    """Mean prediction and observed goal rate in quantile bins of the prediction."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C_OURS, C_MP = "#2a78d6", "#eb6834"  # validated categorical slots 1 and 2


def plot_calibration(test, bdc, bdc_boot_bins, out_path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=200)

    ax = axes[0]
    lim = 0.45
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--", zorder=1)
    for col, color, label in [("xg_lgbm", C_OURS, "Our model (LightGBM)"), ("mp_xGoal", C_MP, "MoneyPuck xG")]:
        r = reliability(test["goal"], test[col], 20)
        ax.plot(r["p"], r["y"], color=color, lw=2, marker="o", ms=4.5, mec="white", mew=1, label=label, zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted goal probability (bin average)")
    ax.set_ylabel("Observed goal rate")
    ax.set_title(f"NHL test season 2025-26 ({len(test):,} shots)", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax = axes[1]
    lim = 0.45
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--", zorder=1)
    r = reliability(bdc["goal"], bdc["xg_lgbm"], 5)
    ax.vlines(r["p"], bdc_boot_bins[:, 0], bdc_boot_bins[:, 1], color=C_OURS, lw=1.2, alpha=0.6, zorder=2)
    ax.plot(r["p"], r["y"], color=C_OURS, lw=2, marker="o", ms=5, mec="white", mew=1, zorder=3,
            label="Our model (LightGBM), 95% game-bootstrap bars")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted goal probability (bin average)")
    ax.set_title(f"Big Data Cup ({len(bdc):,} shots, {int(bdc['goal'].sum())} goals)", loc="left",
                 fontsize=9, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    for ax in axes:
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def bdc_bin_intervals(bdc, rng, n_bins=5, n_boot=N_BOOT):
    """Game-bootstrap 95% interval of the observed goal rate in each prediction bin."""
    bins = pd.qcut(bdc["xg_lgbm"].rank(method="first"), n_bins, labels=False)
    d = bdc.assign(bin=bins.values)
    games = d["game_id"].unique()
    by_game = {g: d[d["game_id"] == g] for g in games}
    rates = []
    for _ in range(n_boot):
        b = pd.concat([by_game[g] for g in rng.choice(games, size=len(games), replace=True)])
        rates.append(b.groupby("bin")["goal"].mean().reindex(range(n_bins)).to_numpy())
    return np.nanpercentile(np.array(rates), [2.5, 97.5], axis=0).T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mp_csv, bdc_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    mp = pd.read_csv(mp_csv, low_memory=False)
    mp = mp[(mp["mp_playoff"] == 0) & (mp["unblocked"] == 1)].reset_index(drop=True)
    bdc_all = pd.read_csv(bdc_csv)
    bdc = bdc_all[bdc_all["unblocked"] == 1].reset_index(drop=True)

    train = mp[mp["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test = mp[mp["season"] == TEST_SEASON].reset_index(drop=True)
    print(f"train {len(train):,} shots ({train.goal.mean():.4f} goals), test {len(test):,}, BDC {len(bdc):,}")

    Xtr, Xte, Xb = base_frame(train), base_frame(test), base_frame(bdc)
    ytr = train["goal"].to_numpy()
    medians = Xtr.median()

    # Logistic models
    simple = fit_logit(Xtr[SIMPLE].fillna(medians[SIMPLE]), ytr)
    full = fit_logit(logit_frame(Xtr, medians), ytr)

    # LightGBM: tune on 2021-23 -> 2024, refit on 2021-24
    tr_mask = train["season"] != VALID_SEASON
    params, n_trees, tuning = tune_lgbm(Xtr[tr_mask], ytr[tr_mask], Xtr[~tr_mask], ytr[~tr_mask])
    tuning.to_csv(f"{out_dir}/xg_lgbm_tuning.csv", index=False)
    lgbm = fit_lgbm(Xtr, ytr, params, n_trees)
    print(f"LightGBM: {params}, {n_trees} trees")

    def predict(X):
        return {"xg_logit_simple": simple.predict_proba(X[SIMPLE].fillna(medians[SIMPLE]))[:, 1],
                "xg_logit_full": full.predict_proba(logit_frame(X, medians))[:, 1],
                "xg_lgbm": lgbm.predict_proba(X)[:, 1]}

    for k, v in predict(Xte).items():
        test[k] = v
    for k, v in predict(Xb).items():
        bdc[k] = v

    # Metrics
    rows = []
    for col in ["xg_logit_simple", "xg_logit_full", "xg_lgbm", "mp_xGoal"]:
        rows.append(dict(data="NHL test 2025-26", model=col, **metrics(test["goal"], test[col])))
    boot = game_bootstrap(bdc, ["xg_logit_simple", "xg_logit_full", "xg_lgbm"], rng)
    for col in ["xg_logit_simple", "xg_logit_full", "xg_lgbm"]:
        rows.append(dict(data="Big Data Cup", model=col, **metrics(bdc["goal"], bdc[col]), **boot[col]))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{out_dir}/xg_metrics.csv", index=False)

    imp = pd.DataFrame({"feature": Xtr.columns, "gain": lgbm.booster_.feature_importance("gain")})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp.sort_values("gain", ascending=False).to_csv(f"{out_dir}/xg_lgbm_importance.csv", index=False)

    # BDC predictions for every shot (blocked shots get no xG)
    pred_cols = ["xg_logit_simple", "xg_logit_full", "xg_lgbm"]
    out = bdc_all.merge(bdc[["game_id", "event_idx"] + pred_cols], on=["game_id", "event_idx"], how="left")
    out.to_csv(f"{out_dir}/xg_bdc_predictions.csv", index=False)

    plot_calibration(test, bdc, bdc_bin_intervals(bdc, rng), f"{out_dir}/fig_xg_calibration.png")

    pd.set_option("display.width", 200)
    show = tbl.drop(columns=[c for c in tbl.columns if c.endswith("_ci")])
    print(show.round(4).to_string(index=False))
    for col in pred_cols:
        b = boot[col]
        print(f"BDC {col}: obs/exp 95% CI {np.round(b['obs_over_exp_ci'], 3)}, "
              f"AUC CI {np.round(b['auc_ci'], 3)}")
    print(imp.sort_values("gain", ascending=False).head(10).round(3).to_string(index=False))
    json.dump({"lgbm_params": params, "lgbm_trees": n_trees}, open(f"{out_dir}/xg_model_settings.json", "w"), indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
