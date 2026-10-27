"""Backward-compatible runner for hypothesis 1."""

from hypothesis1_plot import main as plot_main
from hypothesis1_stats import main as stats_main


def main() -> None:
    stats_main()
    plot_main()


if __name__ == "__main__":
    main()
