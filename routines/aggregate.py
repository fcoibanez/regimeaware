"""Aggregates results from model simulations."""

if __name__ == "__main__":
    import argparse
    import os

    import pandas as pd

    from regimeaware.constants import DataConstants

    ALL_MODELS = [
        "model",
        "baseline",
        "equalweighted",
        "global_min_var",
        "rolling_ols",
        "ck_uni",
        "ck_multi",
        "rwls_mvo",
        "rwls_mixture",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=ALL_MODELS,
        help="Subset to aggregate. Each arm holds thousands of files, so "
        "re-collecting all of them to refresh one is needlessly slow.",
    )
    args = parser.parse_args()

    # Collect results
    for mdl in args.models:
        folder_path = f"{DataConstants.WDIR.value}/results/{mdl}"
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        if not files:
            continue
        collect_res = []
        for file in files:
            if file.endswith(".pkl"):
                phi = int(file.split("_")[0].replace("phi", ""))
                data = pd.read_pickle(os.path.join(folder_path, file))
                data = data.reset_index()
                data["phi"] = phi
                data = data.set_index(["phi", "iteration", "period"])
                collect_res.append(data)

        res = pd.concat(collect_res).sort_index()
        res.to_pickle(f"{DataConstants.WDIR.value}/results/{mdl}.pkl")
