from pathlib import Path

from src.final_analysis import run_final_analysis


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    results = run_final_analysis(
        project_root=PROJECT_ROOT,
    )

    print("\nAUTHORITATIVE COMPARISON")
    print(results["comparison"].to_string(index=False))
    print("\nWIN COUNTS")
    print(results["win_counts"].to_string(index=False))
    print("\nFORECAST COMPARISON TESTS")
    columns = [
        "dataset",
        "horizon",
        "comparison",
        "test_statistic",
        "p_value",
        "holm_adjusted_p_value",
    ]
    print(results["tests"][columns].to_string(index=False))
    print("\nAll final-analysis artifacts were generated from frozen files.")


if __name__ == "__main__":
    main()
