from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(
    "/home/forootani/Documents/ReSTEP/RAFL_revision/geany_sim/results_rafl_fmnist_suite"
)


def load_all_scenarios(base_dir=BASE_DIR):
    rows = []

    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"

        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        row = {
            "results_dir": str(run_dir),
            "suite": config.get("suite"),
            "aggregator": config.get("aggregator"),
            "attack": config.get("attack"),
            "num_clients": config.get("num_clients"),
            "num_clients_per_round": config.get("num_clients_per_round"),
            "num_rounds": config.get("num_rounds"),
            "alpha_stale": config.get("alpha_stale"),
            "trigger_eps": config.get("trigger_eps"),
            "byz_frac": config.get("byz_frac"),
            "tau_max_rounds": config.get("tau_max_rounds"),
            "final_test_acc": metrics.get("final_test_acc"),
            "final_test_loss": metrics.get("final_test_loss"),
            "mean_comm_reduction": metrics.get("mean_comm_reduction"),
            "mean_round_wall_time": metrics.get("mean_round_wall_time"),
        }

        rows.append(row)

    return pd.DataFrame(rows)


"""
def make_latex_safe_values(df):
    df = df.copy()

    # attack names
    if "attack" in df.columns:
        df["attack"] = df["attack"].replace({
            "label_flip": "label-flip",
            "model_replacement": "model replacement",
            "signflip": "sign-flip",
        })

    # aggregator names
    if "aggregator" in df.columns:
        df["aggregator"] = df["aggregator"].replace({
            "abs": "ABS",
            "trimmed_mean": "Trimmed-mean",
            "fedasync": "FedAsync",
            "fedbuff": "FedBuff",
            "krum": "Krum",
            "mean": "Mean",
            "median": "Median",
            "asb": "ASB",
            "aflguard": "AFLGuard",
            "zeno": "Zeno"
        })

    return df
"""


def make_latex_safe_values(df):
    df = df.copy()

    df["attack"] = df["attack"].replace({
        "gaussian": "Gaussian",
        "label_flip": "Label-flip",
        "model_replacement": "Model Replacement",
        "none": "None",
        "signflip": "Sign-flip",
    })

    df["aggregator"] = df["aggregator"].replace({
        "trimmed_mean": "Trimmed-Mean",
        "fedasync": "FedAsync",
        "fedbuff": "FedBuff",
        "krum": "Krum",
        "mean": "Mean",
        "median": "Median",
        "asb": r"$\mathcal{A}_{\mathrm{SB}}$",
        "aflguard": "AFLGuard",
        "zeno": "Zeno",
    })

    return df



def save_scenario_csvs_R500(base_dir=BASE_DIR):
    df = load_all_scenarios(base_dir)

    df = make_latex_safe_values(df)

    merged_csv = base_dir / "all_summaries_merged.csv"
    df.to_csv(merged_csv, index=False)
    print("Saved merged CSV:", merged_csv)

    df_R500 = df[df["num_rounds"] == 500].copy()

    print("Rows total:", len(df))
    print("Rows with R=500:", len(df_R500))
    print("Suites with R=500:", df_R500["suite"].unique())

    out_dir = base_dir / "fmnist_csv_for_latex_R500"
    out_dir.mkdir(exist_ok=True)

    sort_cols = [
        "aggregator",
        "attack",
        "trigger_eps",
        "byz_frac",
        "tau_max_rounds",
    ]

    for suite_name, sdf in df_R500.groupby("suite"):
        safe_name = str(suite_name).replace("/", "_")
        out_path = out_dir / f"{safe_name}_R500.csv"

        sdf = sdf.sort_values(
            by=[c for c in sort_cols if c in sdf.columns]
        )

        # Save ALL columns
        sdf.to_csv(out_path, index=False)

        print(f"Saved: {out_path} ({len(sdf)} rows)")


save_scenario_csvs_R500()