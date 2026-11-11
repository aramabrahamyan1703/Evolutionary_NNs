"""Phase 5.2: sharpness of the networks found (D11), from the saved checkpoint weights.

    python scripts/sharpness.py --jobs 8            # configs/p5_selection.yaml

For every run three networks are measured on its training set:
    final        the returned network (lowest training MSE)
    best_select  the early-stopping checkpoint (lowest selection-set MSE, D10)
    matched      the first checkpoint whose training MSE is down to the GA's final
                 training MSE on the same seed (for the GA: its final network), so
                 all methods are compared at the same training error
Writes results/processed/<experiment>/sharpness.csv (one row per run and network) and
sharpness_wilcoxon.csv (paired tests on the perturbation sharpness of the matched networks).
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from evonn.analysis import make_wilcoxon, processed_dir
from evonn.experiment import build_data, load_config
from evonn.methods import LABELS
from evonn.mlp import MLP
from evonn.sharpness import matched_checkpoint, measure

ROOT = Path(__file__).resolve().parents[1]
KEEP = ("train_mse", "test_mse_clean", "select_mse")


def measure_run(cfg, path, target):
    d = json.loads(Path(path).read_text())
    method, n, seed = d["meta"]["method"], d["meta"]["n_train"], d["meta"]["seed"]
    model = MLP(d_in=cfg["data"]["input_dim"], hidden=cfg["model"]["hidden"])
    X, y = build_data(cfg, seed)["train"][n]
    bound = cfg["model"]["weight_bounds"][1]
    final = {**d["final"], "theta": d["theta"], "evals": d["budget"]["evals_used"]}
    networks = {"final": final, "best_select": d["best_select"],
                "matched": final if method == "ga" else matched_checkpoint(d["curve"], target)}
    return [{"method": method, "n_train": n, "seed": seed, "network": which, "evals": net["evals"],
             **{k: net[k] for k in KEEP}, **measure(model, net["theta"], X, y, bound)}
            for which, net in networks.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p5_selection.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    raw = ROOT / cfg["experiment"]["output_dir"] / name
    runs = sorted(raw.glob("*.json"))
    targets = {}                                   # (n, seed) -> GA final training MSE
    for p in raw.glob("ga_*.json"):
        d = json.loads(p.read_text())
        targets[(d["meta"]["n_train"], d["meta"]["seed"])] = d["final"]["train_mse"]
    if not runs or not targets:
        parser.error(f"no runs (or no GA runs) in {raw}")

    def key(p):
        _, n, s = p.stem.split("_")
        return int(n[1:]), int(s[1:])

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(measure_run, cfg, p, targets[key(p)]) for p in runs]
        df = pd.DataFrame([row for f in futures for row in f.result()])

    out = processed_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sharpness.csv", index=False)
    matched = df[df["network"] == "matched"]
    wil = make_wilcoxon(matched, metric="perturb_sharpness")
    wil.to_csv(out / "sharpness_wilcoxon.csv", index=False)

    cols = ["train_mse", "test_mse_clean", "perturb_sharpness", "top_eigenvalue", "theta_norm", "frac_at_bounds"]
    for which in ("matched", "final", "best_select"):
        print(f"\n{which} networks (medians)")
        med = df[df["network"] == which].groupby(["n_train", "method"])[cols].median()
        med.index = med.index.set_levels([med.index.levels[0], [LABELS[m] for m in med.index.levels[1]]])
        print(med.to_string(float_format="{:.4g}".format))
    sig = wil[wil["significant"]]
    print(f"\nmatched networks, perturbation sharpness: {len(sig)} of {len(wil)} pairs significant (p_holm < 0.05)")
    for _, r in wil[(wil["method_a"] == "ga") | (wil["method_b"] == "ga")].iterrows():
        print(f"  n={int(r.n_train):4d} {r.method_a} vs {r.method_b}: delta={r.cliffs_delta:+.2f} p_holm={r.p_holm:.2g}")
    print(f"\n-> {out.relative_to(ROOT)}/sharpness.csv, sharpness_wilcoxon.csv")


if __name__ == "__main__":
    main()
