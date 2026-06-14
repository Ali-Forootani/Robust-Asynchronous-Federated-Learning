#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 06:20:19 2026

@author: forootani
"""

from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(
    "/home/forootani/Documents/ReSTEP/RAFL_revision/geany_sim/results_rafl_cifar10_suite"
)

DEFAULTS = {
    "alpha_stale": 0.2,
    "trigger_eps": 0.0,
    "byz_frac": 0.3,
    "tau_max_rounds": 3,
    "num_clients": 10,
    "num_clients_per_round": 5,
}


def infer_reporting_suite(row):
    suite = row["suite"]

    if suite != "single":
        return suite

    if row["alpha_stale"] != DEFAULTS["alpha_stale"]:
        return "sensitivity_alpha_stale"

    if row["trigger_eps"] != DEFAULTS["trigger_eps"]:
        return "sensitivity_trigger_eps"

    if row["byz_frac"] != DEFAULTS["byz_frac"]:
        return "sensitivity_byz_frac"

    if row["tau_max_rounds"] != DEFAULTS["tau_max_rounds"]:
        return "sensitivity_tau_max_rounds"

    if row["num_clients"] != DEFAULTS["num_clients"]:
        return "sensitivity_num_clients"

    return "single_default"


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

        row["reporting_suite"] = infer_reporting_suite(row)
        rows.append(row)

    return pd.DataFrame(rows)

"""
def make_latex_safe_values(df):
    df = df.copy()

    df["attack"] = df["attack"].replace({
        "label_flip": "label-flip",
        "model_replacement": "model replacement",
        "signflip": "sign-flip",
    })

    df["aggregator"] = df["aggregator"].replace({
        "trimmed_mean": "Trimmed-mean",
        "fedasync": "FedAsync",
        "fedbuff": "FedBuff",
        "krum": "Krum",
        "mean": "Mean",
        "median": "Median",
        "asb": "ASB",
        "aflguard": "AFLGuard",
        "zeno": "Zeno",
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




def save_scenario_csvs_R400(base_dir=BASE_DIR):
    df = load_all_scenarios(base_dir)
    df = make_latex_safe_values(df)

    merged_csv = base_dir / "all_summaries_merged.csv"
    df.to_csv(merged_csv, index=False)
    print("Saved merged CSV:", merged_csv)

    df_R400 = df[df["num_rounds"] == 400].copy()

    print("Rows total:", len(df))
    print("Rows with R=400:", len(df_R400))
    print("Reporting suites with R=400:")
    print(df_R400["reporting_suite"].value_counts())

    out_dir = base_dir / "cifar10_csv_for_latex_R400"
    out_dir.mkdir(exist_ok=True)

    sort_cols = [
        "reporting_suite",
        "aggregator",
        "attack",
        "alpha_stale",
        "trigger_eps",
        "byz_frac",
        "tau_max_rounds",
        "num_clients",
    ]

    for suite_name, sdf in df_R400.groupby("reporting_suite"):
        safe_name = str(suite_name).replace("/", "_")
        out_path = out_dir / f"{safe_name}_R400.csv"

        sdf = sdf.sort_values(by=[c for c in sort_cols if c in sdf.columns])
        sdf.to_csv(out_path, index=False)

        print(f"Saved: {out_path} ({len(sdf)} rows)")


save_scenario_csvs_R400()