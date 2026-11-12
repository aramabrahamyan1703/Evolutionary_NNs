"""Phase 4 analysis: collect run JSONs into tables and compare methods / experiments.

    df  = load_runs(results/raw/<experiment>)      # one row per run
    med = make_medians(df)                         # median + IQR per (method, n)
    wil = make_wilcoxon(df)                        # every method pair at every n, Holm-corrected

Runs are always paired by seed: the same seed means the same training set and
the same initial weights, for every method and every experiment.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "results" / "processed"

MEDIAN_METRICS = ["train_mse", "test_mse", "test_mse_clean", "excess_test", "gap",
                  "frac_at_bounds", "evals_used", "selected_evals", "wall_time"]


def processed_dir(experiment):
    """Output folder of one experiment: results/processed/<experiment>/."""
    return PROCESSED / experiment


def cliffs_delta(x, y):
    """Cliff's delta effect size: P(x > y) - P(x < y), in [-1, 1]."""
    x, y = np.asarray(x), np.asarray(y)
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((greater - less) / (len(x) * len(y)))


def vargha_delaney(x, y):
    """Vargha-Delaney A12: P(x > y) + P(x = y) / 2, in [0, 1] (0.5: no effect; Cliff's delta = 2 A12 - 1)."""
    x, y = np.asarray(x), np.asarray(y)
    return float((np.sum(x[:, None] > y[None, :]) + 0.5 * np.sum(x[:, None] == y[None, :])) / (len(x) * len(y)))


BOOTSTRAP = 10_000    # resamples for the confidence intervals
BOOTSTRAP_SEED = 0


def paired_bootstrap(x, y, n_boot=BOOTSTRAP, level=0.95, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap over seeds (pairs resampled together) for the difference of medians
    median(x) - median(y) and for A12. Returns (diff, diff_lo, diff_hi, a12, a12_lo, a12_hi)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    xb, yb = x[idx], y[idx]
    diffs = np.median(xb, axis=1) - np.median(yb, axis=1)
    gt = (xb[:, :, None] > yb[:, None, :]).mean(axis=(1, 2))
    eq = (xb[:, :, None] == yb[:, None, :]).mean(axis=(1, 2))
    a12s = gt + 0.5 * eq
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    return (float(np.median(x) - np.median(y)), *np.percentile(diffs, q).tolist(),
            vargha_delaney(x, y), *np.percentile(a12s, q).tolist())


def holm_correction(p_values):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (n - rank)))   # monotone in the sorted order
        adjusted[idx] = running
    return adjusted


SELECTIONS = ("final", "best_val", "best_select")


def load_runs(raw_dir, selection="final"):
    """One row per run JSON in raw_dir: identifiers, budget used, and the metrics of the
    network chosen by `selection`:
      final        the returned network (lowest training loss, D6)
      best_val     the checkpoint with the lowest MSE on the 5000-point validation set
      best_select  the checkpoint with the lowest MSE on the small selection set (D10, Phase 5)
    selected_evals is the evaluation count of the chosen network (evals_used for final)."""
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}")
    rows = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        d = json.loads(f.read_text())
        if selection not in d:
            raise KeyError(f"{f.name} has no {selection!r} (run with a `selection:` config block)")
        chosen = {k: v for k, v in d[selection].items() if k not in ("theta", "evals")}
        rows.append({"method": d["meta"]["method"], "n_train": d["meta"]["n_train"],
                     "seed": d["meta"]["seed"], "evals_used": d["budget"]["evals_used"],
                     "selected_evals": d[selection].get("evals", d["budget"]["evals_used"]),
                     "wall_time": d["budget"]["wall_time"], "git_commit": d["meta"]["git_commit"],
                     **chosen})
    if not rows:
        raise FileNotFoundError(f"no run JSONs in {raw_dir}")
    return pd.DataFrame(rows).sort_values(["method", "n_train", "seed"]).reset_index(drop=True)


def make_medians(df, metrics=MEDIAN_METRICS):
    """Median and quartiles of every metric per (method, n_train)."""
    rows = []
    for (method, n), g in df.groupby(["method", "n_train"]):
        row = {"method": method, "n_train": n, "runs": len(g)}
        for m in metrics:
            row[f"{m}_median"] = g[m].median()
            row[f"{m}_q25"] = g[m].quantile(0.25)
            row[f"{m}_q75"] = g[m].quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_train", "method"]).reset_index(drop=True)


def paired(df, metric, n, a, b):
    """Values of methods a and b at size n, matched by seed. Raises if the seeds differ."""
    sub = df[df["n_train"] == n]
    xa = sub[sub["method"] == a].set_index("seed")[metric]
    xb = sub[sub["method"] == b].set_index("seed")[metric]
    if set(xa.index) != set(xb.index):
        raise ValueError(f"n={n}: {a} and {b} have different seeds "
                         f"({sorted(set(xa.index) ^ set(xb.index))} not in both)")
    seeds = sorted(xa.index)
    return xa.loc[seeds].to_numpy(), xb.loc[seeds].to_numpy()


def _wilcoxon_p(x, y):
    if np.all(x == y):
        return np.nan, 1.0
    stat, p = wilcoxon(x, y, alternative="two-sided")
    return float(stat), float(p)


def make_wilcoxon(df, metric="test_mse_clean"):
    """Paired Wilcoxon signed-rank test for every method pair at every n, Holm-corrected
    over all tests in the table; Cliff's delta > 0 (A12 > 0.5) means method_a has the larger
    values. Effect sizes come with 95% paired-bootstrap intervals: the difference of medians
    (diff_lo, diff_hi) and A12 (a12_lo, a12_hi). Empty table if there is only one method."""
    methods = sorted(df["method"].unique())
    rows = []
    for n in sorted(df["n_train"].unique()):
        for a, b in combinations(methods, 2):
            x, y = paired(df, metric, n, a, b)
            stat, p = _wilcoxon_p(x, y)
            diff, lo, hi, a12, a_lo, a_hi = paired_bootstrap(x, y)
            rows.append({"n_train": n, "method_a": a, "method_b": b, "runs": len(x),
                         "median_a": float(np.median(x)), "median_b": float(np.median(y)),
                         "median_diff": diff, "diff_lo": lo, "diff_hi": hi,
                         "a12": a12, "a12_lo": a_lo, "a12_hi": a_hi,
                         "cliffs_delta": cliffs_delta(x, y), "W": stat, "p_raw": p})
    result = pd.DataFrame(rows, columns=["n_train", "method_a", "method_b", "runs", "median_a", "median_b",
                                         "median_diff", "diff_lo", "diff_hi", "a12", "a12_lo", "a12_hi",
                                         "cliffs_delta", "W", "p_raw"])
    result["p_holm"] = holm_correction(result["p_raw"]) if len(result) else []
    result["significant"] = result["p_holm"] < 0.05
    return result


def combine_with_base(other, base):
    """`other` plus the base runs of every method missing from `other` (used when an
    experiment reruns only some methods, e.g. grad_cost only affects gradient methods)."""
    missing = sorted(set(base["method"]) - set(other["method"]))
    keep = base[base["method"].isin(missing) & base["n_train"].isin(other["n_train"].unique())]
    return pd.concat([other, keep], ignore_index=True), missing


def compare_methods(base, other, metrics=("train_mse", "test_mse_clean"), test_metric="test_mse_clean"):
    """Every (method, n) of `other` against the same method in `base`, paired by seed."""
    rows = []
    for (method, n), g in other.groupby(["method", "n_train"]):
        b = base[(base["method"] == method) & (base["n_train"] == n)].set_index("seed")
        o = g.set_index("seed")
        if set(b.index) != set(o.index):
            raise ValueError(f"{method} n={n}: base and other have different seeds")
        seeds = sorted(o.index)
        row = {"method": method, "n_train": n, "runs": len(seeds)}
        for m in metrics:
            row[f"{m}_base"] = b.loc[seeds, m].median()
            row[f"{m}_other"] = o.loc[seeds, m].median()
        _, row["p_raw"] = _wilcoxon_p(o.loc[seeds, test_metric].to_numpy(), b.loc[seeds, test_metric].to_numpy())
        row["cliffs_delta"] = cliffs_delta(o.loc[seeds, test_metric], b.loc[seeds, test_metric])
        rows.append(row)
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_correction(result["p_raw"])
    return result


def compare_pairs(base_wil, other_wil):
    """Significant method pairs in base vs other: same verdict, or changed?"""
    keys = ["n_train", "method_a", "method_b"]
    cols = keys + ["median_diff", "cliffs_delta", "p_holm", "significant"]
    m = base_wil[cols].merge(other_wil[cols], on=keys, suffixes=("_base", "_other"))
    sign = lambda s: np.sign(s).astype(int)
    m["same_verdict"] = (m["significant_base"] == m["significant_other"]) & (
        ~m["significant_base"] | (sign(m["median_diff_base"]) == sign(m["median_diff_other"])))
    return m


def rankings(df, metric="test_mse_clean"):
    """Methods ordered by median metric (best first) at every n: {n: [method, ...]}."""
    med = df.groupby(["n_train", "method"])[metric].median()
    return {int(n): list(med.loc[n].sort_values().index) for n in sorted(df["n_train"].unique())}
