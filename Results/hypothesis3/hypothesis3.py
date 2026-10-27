"""Backward-compatible runner for hypothesis 3."""

from hypothesis3_plot import main as plot_main
from hypothesis3_stats import main as stats_main


def main() -> None:
    stats_main()
    plot_main()


if __name__ == "__main__":
    main()
