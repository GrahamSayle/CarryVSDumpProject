"""Midway analysis: raw gap, trimmed raw gap, interacted (Lin) regression ATE on the trimmed population,
game-level cluster bootstrap that reruns the whole pipeline, leave-one-game-out, balance, overlap and placebo.

Usage: python analyze.py <entries_csv> <out_dir> [n_boot]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TRIM = (0.05, 0.95)
N_FOLDS = 5
NUM = ["period_elapsed", "skater_diff", "even_strength", "score_diff", "lane",
       "since_faceoff", "shift_time", "home"]
CAT = ["period", "team"]          # team = entering team (release-prefixed); absorbs the release indicator


def design(df):
    X = pd.concat([df[NUM].astype(float),
                   pd.get_dummies(df[CAT].astype(str), drop_first=True, dtype=float)], axis=1)
    return X


def crossfit_propensity(df, X, seed=0):
    """Out-of-fold P(carry | X), folds formed from *original* game ids so bootstrap copies share a fold."""
    rng = np.random.default_rng(seed)
    games = np.array(sorted(df["orig_game"].unique()))
    fold_of = dict(zip(games, rng.permutation(np.arange(len(games)) % N_FOLDS)))
    folds = df["orig_game"].map(fold_of).values
    e = np.empty(len(df))
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        if te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), df["carry"].values[tr])
        e[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return e


def lin_ate(y, d, X):
    """Lin (2013) interacted regression: y ~ d + Xc + d:Xc, Xc centred on this sample. Coef on d = ATE."""
    Xc = X - X.mean(axis=0)
    keep = Xc.std(axis=0) > 0
    Xc = Xc[:, keep]
    Z = np.column_stack([np.ones(len(y)), d, Xc, d[:, None] * Xc])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return beta[1]


def ipw_smd(df, X_num, w):
    """Standardized mean differences of numeric covariates, carried minus dumped, with weights w."""
    out = {}
    d = df["carry"].values == 1
    for c in X_num:
        x = df[c].values.astype(float)
        sd = np.sqrt((x[d].var() + x[~d].var()) / 2) or 1.0
        m1 = np.average(x[d], weights=w[d]); m0 = np.average(x[~d], weights=w[~d])
        out[c] = (m1 - m0) / sd
    return out


def pipeline(df, seed=0, outcome="y", keep_details=False):
    X = design(df).values
    y = df[outcome].values.astype(float)
    d = df["carry"].values.astype(float)
    raw_all = y[d == 1].mean() - y[d == 0].mean()
    ratio_all = y[d == 1].mean() / y[d == 0].mean()
    e = crossfit_propensity(df, X, seed)
    keep = (e > TRIM[0]) & (e < TRIM[1])
    yt, dt, Xt = y[keep], d[keep], X[keep]
    res = dict(raw_all=raw_all, ratio_all=ratio_all,
               raw_trim=yt[dt == 1].mean() - yt[dt == 0].mean(),
               lin_ate=lin_ate(yt, dt, Xt),
               n_trimmed=int((~keep).sum()))
    if keep_details:
        res["e"], res["keep"] = e, keep
    return res


def cluster_bootstrap(df, B, seed=1):
    rng = np.random.default_rng(seed)
    games = df["orig_game"].unique()
    out = []
    for b in range(B):
        draw = rng.choice(games, size=len(games), replace=True)
        bdf = pd.concat([df[df.orig_game == g] for g in draw], ignore_index=True)
        try:
            out.append(pipeline(bdf, seed=b))
        except ValueError:          # e.g. a resample where a fold lacks one class
            continue
    return pd.DataFrame(out)


def main(entries_csv, out_dir, B=1000):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(entries_csv)
    df["orig_game"] = df["game"]

    point = pipeline(df, seed=0, keep_details=True)
    e, keep = point.pop("e"), point.pop("keep")
    placebo = pipeline(df, seed=0, outcome="y_pre")
    has_own = "y_pre_own" in df.columns   # sensitivity placebo: entering team's shots only
    placebo_own = pipeline(df, seed=0, outcome="y_pre_own") if has_own else None

    boot = cluster_bootstrap(df, B)
    stats = ["raw_all", "ratio_all", "raw_trim", "lin_ate"]
    ci = {s: (np.percentile(boot[s], 2.5), np.percentile(boot[s], 97.5)) for s in stats}

    # leave one game out
    logo = []
    for g in sorted(df.game.unique()):
        r = pipeline(df[df.game != g].reset_index(drop=True), seed=0)
        r["dropped"] = g
        logo.append(r)
    logo = pd.DataFrame(logo)

    # per-game raw gap
    pg = df.groupby(["game", "carry"])["y"].mean().unstack()
    per_game = pd.DataFrame({"carried": pg[1], "dumped": pg[0], "gap": pg[1] - pg[0],
                             "n_entries": df.groupby("game").size()})

    # balance: before (unweighted, all) vs after (ATE inverse-probability weights, trimmed set)
    w_before = np.ones(len(df))
    dt = df[keep].reset_index(drop=True); et = e[keep]
    w_after = np.where(dt.carry == 1, 1 / et, 1 / (1 - et))
    bal = pd.DataFrame({"smd_before": ipw_smd(df, NUM, w_before),
                        "smd_after_ipw_trimmed": ipw_smd(dt, NUM, w_after)})

    # ---------- outputs ----------
    tbl = pd.DataFrame([
        ["1. Raw difference, all entries", point["raw_all"], *ci["raw_all"], len(df)],
        ["1b. Raw difference, trimmed entries", point["raw_trim"], *ci["raw_trim"], len(df) - point["n_trimmed"]],
        ["2. Interacted regression (ATE, trimmed)", point["lin_ate"], *ci["lin_ate"], len(df) - point["n_trimmed"]],
    ], columns=["estimate", "value", "ci_low", "ci_high", "n_entries"])
    tbl.to_csv(f"{out_dir}/results_table.csv", index=False)
    per_game.to_csv(f"{out_dir}/per_game.csv")
    logo.to_csv(f"{out_dir}/leave_one_game_out.csv", index=False)
    bal.to_csv(f"{out_dir}/balance.csv")
    summary = dict(point=point, ci=ci, n_boot_ok=len(boot), placebo_lin_ate=placebo["lin_ate"],
                   placebo_raw_all=placebo["raw_all"],
                   logo_lin_range=[logo.lin_ate.min(), logo.lin_ate.max()],
                   logo_raw_range=[logo.raw_all.min(), logo.raw_all.max()],
                   mean_y_carry=float(df[df.carry == 1].y.mean()),
                   mean_y_dump=float(df[df.carry == 0].y.mean()))
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2, default=float)

    # placebo bootstrap CI (fewer resamples is fine)
    pboot, pboot_own = [], []
    rng = np.random.default_rng(7); games = df.orig_game.unique()
    for b in range(min(B, 500)):
        draw = rng.choice(games, size=len(games), replace=True)
        bdf = pd.concat([df[df.orig_game == g] for g in draw], ignore_index=True)
        try:
            pboot.append(pipeline(bdf, seed=b, outcome="y_pre")["lin_ate"])
            if has_own:   # same resample, so the two placebo intervals are comparable
                pboot_own.append(pipeline(bdf, seed=b, outcome="y_pre_own")["lin_ate"])
        except ValueError:
            pass
    summary["placebo_ci"] = [np.percentile(pboot, 2.5), np.percentile(pboot, 97.5)]
    if has_own:
        summary["placebo_own_lin_ate"] = placebo_own["lin_ate"]
        summary["placebo_own_ci"] = [np.percentile(pboot_own, 2.5), np.percentile(pboot_own, 97.5)]
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2, default=float)

    # ---------- figure 1: overlap ----------
    INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
    C_CARRY, C_DUMP = "#2563eb", "#d97706"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(6.2, 2.9), dpi=200)
    bins = np.linspace(0, 1, 41)
    ax.hist(e[df.carry == 1], bins=bins, alpha=0.55, color=C_CARRY, label=f"Carried (n={int(df.carry.sum())})")
    ax.hist(e[df.carry == 0], bins=bins, alpha=0.55, color=C_DUMP, label=f"Dumped (n={int((1-df.carry).sum())})")
    for v in TRIM:
        ax.axvline(v, color=INK, lw=0.8, ls="--")
    ax.text(TRIM[1] + 0.01, ax.get_ylim()[1] * 0.92, "trim", color=INK, fontsize=8)
    ax.set_xlabel("Estimated probability of carrying, e(X) (cross-fitted by game)")
    ax.set_ylabel("Entries")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Overlap: {point['n_trimmed']} of {len(df)} entries fall outside 0.05–0.95 and are trimmed",
                 fontsize=9, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_overlap.png"); plt.close(fig)

    # ---------- figure 2: estimates with intervals + leave-one-game-out ----------
    fig, ax = plt.subplots(figsize=(6.2, 2.5), dpi=200)
    labels = ["Raw, all entries", "Raw, trimmed", "Interacted regression\n(ATE, trimmed)"]
    keys = ["raw_all", "raw_trim", "lin_ate"]
    for i, k in enumerate(keys):
        yv = len(keys) - 1 - i
        ax.scatter(logo[k], np.full(len(logo), yv) - 0.22, s=10, color=MUTED, alpha=0.7,
                   label="Leave one game out" if i == 0 else None, zorder=2)
        ax.plot(ci[k], [yv, yv], color=C_CARRY, lw=2, zorder=3)
        ax.scatter([point[k]], [yv], color=C_CARRY, s=28, zorder=4,
                   label="Estimate, 95% game-bootstrap interval" if i == 0 else None)
        ax.text(ci[k][1] + 0.01, yv, f"{point[k]:.2f} [{ci[k][0]:.2f}, {ci[k][1]:.2f}]",
                va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Carried minus dumped: shot attempts within 20 s per entry")
    ax.set_xlim(-0.05, max(ci[k][1] for k in keys) + 0.22)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_estimates.png"); plt.close(fig)

    print(json.dumps(summary, indent=2, default=float))
    print(tbl.round(3).to_string(index=False))
    print(bal.round(3))
    print(per_game.round(3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
