import pandas as pd
import matplotlib.pyplot as plt


Cleaned = pd.read_csv("./data/Cleaned.csv")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


interactive = Cleaned[Cleaned["Condition"] == "Interactive"].copy()
text = Cleaned[Cleaned["Condition"] == "Text"].copy()

analytical_pre_interactive = _to_numeric(interactive["Total Analytical Trust"])
analytical_pre_text = _to_numeric(text["Total Analytical Trust"])
analytical_post_interactive = _to_numeric(interactive["Total Analytical Trust Post"])
analytical_post_text = _to_numeric(text["Total Analytical Trust Post"])
analytical_diff_interactive = _to_numeric(interactive["Analytical Trust Difference"])
analytical_diff_text = _to_numeric(text["Analytical Trust Difference"])

emotional_pre_interactive = _to_numeric(interactive["Total Emotional Trust"])
emotional_pre_text = _to_numeric(text["Total Emotional Trust"])
emotional_post_interactive = _to_numeric(interactive["Total Emotional Trust Post"])
emotional_post_text = _to_numeric(text["Total Emotional Trust Post"])
emotional_diff_interactive = _to_numeric(interactive["Emotional Trust Difference"])
emotional_diff_text = _to_numeric(text["Emotional Trust Difference"])


fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)


def _plot_hist(ax, interactive_data: pd.Series, text_data: pd.Series, title: str) -> None:
    ax.hist(interactive_data, bins=20, alpha=0.6, label="Interactive", color="tab:blue")
    ax.hist(text_data, bins=20, alpha=0.6, label="Text", color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()


_plot_hist(axes[0, 0], analytical_pre_interactive, analytical_pre_text, "Analytical Trust (Pre)")
_plot_hist(axes[0, 1], analytical_post_interactive, analytical_post_text, "Analytical Trust (Post)")
_plot_hist(axes[0, 2], analytical_diff_interactive, analytical_diff_text, "Analytical Trust Difference")

_plot_hist(axes[1, 0], emotional_pre_interactive, emotional_pre_text, "Emotional Trust (Pre)")
_plot_hist(axes[1, 1], emotional_post_interactive, emotional_post_text, "Emotional Trust (Post)")
_plot_hist(axes[1, 2], emotional_diff_interactive, emotional_diff_text, "Emotional Trust Difference")

plt.tight_layout()
plot_path = "./data/trust_pre_post_difference_histograms_by_condition.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Saved plot to {plot_path}")
