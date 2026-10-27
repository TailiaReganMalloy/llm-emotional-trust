"""Backward-compatible runner for hypothesis 5."""

from hypothesis5_plot import main as plot_main
from hypothesis5_stats import main as stats_main


def main() -> None:
    stats_main()
    plot_main()


if __name__ == "__main__":
    main()
