"""
Entry point: run all QAOA experiments.
"""

import argparse
from experiment import (
    task2_p1_baseline, task3_p_scan, task4_init_comparison,
    task5_optimizer_comparison, task6_shot_comparison, save_results
)
from visualize import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="QAOA MaxCut Experiments")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "2", "3", "4", "5", "6", "plot"],
                        help="Which task to run")
    parser.add_argument("--max-p", type=int, default=5,
                        help="Maximum depth p for Task 3")
    args = parser.parse_args()

    if args.task in ("all", "2"):
        print("=== Task 2: p=1 Baseline ===")
        r = task2_p1_baseline()
        save_results(r, "task2_p1_baseline.json")

    if args.task in ("all", "3"):
        print("=== Task 3: p-Scan ===")
        r = task3_p_scan(max_p=args.max_p)
        save_results(r, "task3_p_scan.json")

    if args.task in ("all", "4"):
        print("=== Task 4: Init Strategy Comparison ===")
        r = task4_init_comparison()
        save_results(r, "task4_init_comparison.json")

    if args.task in ("all", "5"):
        print("=== Task 5: Optimizer Comparison ===")
        r = task5_optimizer_comparison()
        save_results(r, "task5_optimizer_comparison.json")

    if args.task in ("all", "6"):
        print("=== Task 6: Shot Number Comparison ===")
        r = task6_shot_comparison()
        save_results(r, "task6_shot_comparison.json")

    if args.task in ("all", "plot"):
        print("=== Generating Plots ===")
        generate_all_plots()

    print("\nDone!")


if __name__ == "__main__":
    main()
