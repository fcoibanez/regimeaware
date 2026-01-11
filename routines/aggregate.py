"""Aggregates results from model simulations."""

if __name__ == "__main__":
    import pandas as pd
    from regimeaware.constants import DataConstants
    import os

    # Collect results
    for mdl in ["model", "baseline"]:
        folder_path = f"{DataConstants.WDIR.value}/results/{mdl}"
        files = os.listdir(folder_path)
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
