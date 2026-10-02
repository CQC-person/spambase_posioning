#!/usr/bin/env python3
"""CLI script to run the multi-seed data poisoning pilot study for Stage 3."""

import os
import sys
import argparse

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments import run_pilot_study


def main():
    parser = argparse.ArgumentParser(description="Run Spambase Data Poisoning Pilot Study (Stage 3)")
    parser.add_argument("--data-path", default="data/spambase.data", help="Path to spambase.data")
    parser.add_argument("--names-path", default="data/spambase.names", help="Path to spambase.names")
    parser.add_argument("--output-csv", default="results/stage3_pilot_results.csv", help="Path to output results CSV")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds for pilot study")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.10, 0.20], help="Poisoning rates")

    args = parser.parse_args()

    print("================================================================")
    print("      STAGE 3: REPRODUCIBLE DATA POISONING PILOT RUNNER         ")
    print("================================================================")
    print(f"Seeds:        {args.seeds}")
    print(f"Rates:        {args.rates}")
    print(f"Output File:  {args.output_csv}")
    print("================================================================\n")

    results_df = run_pilot_study(
        data_path=args.data_path,
        names_path=args.names_path,
        seeds=args.seeds,
        poison_rates=args.rates,
        output_csv=args.output_csv
    )

    print("\nSummary preview (first 5 clean rows):")
    print(results_df[results_df["attack"] == "clean"].head())

    print("\nSummary preview (first 5 poisoned rows):")
    print(results_df[results_df["attack"] != "clean"].head())

    print("\n[SUCCESS] Stage 3 pilot study successfully completed and verified.")


if __name__ == "__main__":
    main()
