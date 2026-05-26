

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Polygon, Rectangle
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

chris_sims = {
	"Country": "USA",
	"Citations": 1993,
	"Birthplace HDI": 0.938,
	"Year":2022,
}

peter_danielson = {
	"Country": "Canada",
	"Citations": 1571,
	"Birthplace HDI": 0.939,
	"Year":2018,
}

coty_gonzalez = {
	"Country": "Mexico",
	"Citations": 12541,
	"Birthplace HDI": 0.755,
	"Year":2025,
}

tegawende_bissyande = {
	"Country": "Burkina Faso",
	"Citations": 15094,
	"Birthplace HDI": 0.459,
	"Year":2027,
}

fei_fang = {
	"Country": "China",
	"Citations": 6747,
	"Birthplace HDI": 0.797,
	"Year":2028,
}


def _build_flag_icon(country: str, width: int = 18, height: int = 12) -> DrawingArea:
	icon = DrawingArea(width, height, 0, 0)

	def add_rect(x: float, y: float, w: float, h: float, color: str) -> None:
		icon.add_artist(Rectangle((x, y), w, h, facecolor=color, edgecolor="none"))

	add_rect(0, 0, width, height, "white")

	if country == "USA":
		stripe_h = height / 7
		for i in range(7):
			color = "#B22234" if i % 2 == 0 else "white"
			add_rect(0, i * stripe_h, width, stripe_h, color)
		add_rect(0, height - 4 * stripe_h, width * 0.45, 4 * stripe_h, "#3C3B6E")
	elif country == "Canada":
		add_rect(0, 0, width / 4, height, "#D80621")
		add_rect(width / 4, 0, width / 2, height, "white")
		add_rect(3 * width / 4, 0, width / 4, height, "#D80621")
	elif country == "Mexico":
		add_rect(0, 0, width / 3, height, "#006847")
		add_rect(width / 3, 0, width / 3, height, "white")
		add_rect(2 * width / 3, 0, width / 3, height, "#CE1126")
	elif country == "Burkina Faso":
		add_rect(0, height / 2, width, height / 2, "#EF2B2D")
		add_rect(0, 0, width, height / 2, "#009E49")
		icon.add_artist(Circle((width / 2, height / 2), height * 0.16, color="#FCD116"))
	elif country == "China":
		add_rect(0, 0, width, height, "#DE2910")
		icon.add_artist(Circle((width * 0.26, height * 0.72), height * 0.14, color="#FFDE00"))
	elif country == "South Sudan":
		add_rect(0, 2 * height / 3, width, height / 3, "#000000")
		add_rect(0, height / 3, width, height / 3, "#DA121A")
		add_rect(0, 0, width, height / 3, "#078930")
		add_rect(0, height / 3 - 0.5, width, 1.0, "white")
		add_rect(0, 2 * height / 3 - 0.5, width, 1.0, "white")
		icon.add_artist(
			Polygon(
				[(0, 0), (0, height), (width * 0.46, height / 2)],
				closed=True,
				facecolor="#0F47AF",
				edgecolor="none",
			)
		)
		icon.add_artist(Circle((width * 0.15, height / 2), height * 0.12, color="#FCD116"))

	icon.add_artist(
		Rectangle((0, 0), width, height, facecolor="none", edgecolor="#222222", linewidth=0.4)
	)
	return icon


def _build_bird_icon(width: int = 12, height: int = 10) -> DrawingArea:
	icon = DrawingArea(width, height, 0, 0)

	icon.add_artist(Circle((width * 0.38, height * 0.5), height * 0.26, color="#4F7CAC"))
	icon.add_artist(Circle((width * 0.62, height * 0.52), height * 0.2, color="#6DAEDB"))
	icon.add_artist(
		Polygon(
			[(width * 0.78, height * 0.52), (width * 0.98, height * 0.58), (width * 0.8, height * 0.43)],
			closed=True,
			facecolor="#E8A317",
			edgecolor="none",
		)
	)
	icon.add_artist(Circle((width * 0.68, height * 0.58), height * 0.035, color="#111111"))

	return icon


def _annotate_with_flag(ax, x: float, y: float, country: str, label: str) -> None:
	flag_icon = _build_flag_icon(country)
	ax.add_artist(
		AnnotationBbox(
			flag_icon,
			(x, y),
			xycoords="data",
			xybox=(8, 8),
			boxcoords="offset points",
			frameon=False,
			box_alignment=(0, 0.5),
		)
	)

	if country == "Mexico":
		bird_icon = _build_bird_icon()
		ax.add_artist(
			AnnotationBbox(
				bird_icon,
				(x, y),
				xycoords="data",
				xybox=(22, 22),
				boxcoords="offset points",
				frameon=False,
			)
		)

	ax.annotate(
		label,
		(x, y),
		textcoords="offset points",
		xytext=(30, 8),
		ha="left",
		va="bottom",
	)


def plot_advisors_regression() -> None:
	advisors = {
		"Chris Sims": chris_sims,
		"Peter Danielson": peter_danielson,
		"Coty Gonzalez": coty_gonzalez,
		"Tegawende Bissyande": tegawende_bissyande,
		"Fei Fang": fei_fang,
	}

	df = pd.DataFrame(
		[
			{
				"Name": name,
				"Country": values["Country"],
				"Year": values["Year"],
				"Birthplace HDI": values["Birthplace HDI"],
				"Citations": values["Citations"],
			}
			for name, values in advisors.items()
		]
	)

	hdi_regression = linregress(df["Birthplace HDI"], df["Citations"])
	hdi_r_squared = hdi_regression.rvalue ** 2
	hdi_p_value = hdi_regression.pvalue

	south_sudan_hdi = 0.381
	south_sudan_citations = hdi_regression.slope * south_sudan_hdi + hdi_regression.intercept

	year_regression = linregress(df["Year"], df["Citations"])
	year_r_squared = year_regression.rvalue ** 2
	year_p_value = year_regression.pvalue
	next_advisor_year = 2028
	next_advisor_citations = year_regression.slope * next_advisor_year + year_regression.intercept

	sns.set_theme(style="whitegrid")
	fig, (ax_hdi, ax_year) = plt.subplots(1, 2, figsize=(18, 7))

	sns.regplot(
		data=df,
		x="Birthplace HDI",
		y="Citations",
		ax=ax_hdi,
		ci=None,
		scatter_kws={"s": 90, "color": "#1f77b4"},
		line_kws={"color": "#d62728", "linewidth": 2},
	)

	for _, row in df.iterrows():
		x = row["Birthplace HDI"]
		y = row["Citations"]
		_annotate_with_flag(ax_hdi, x, y, row["Country"], row["Name"])

	ax_hdi.scatter(
		x=south_sudan_hdi,
		y=south_sudan_citations,
		s=130,
		marker="X",
		color="#111111",
		zorder=5,
	)
	_annotate_with_flag(
		ax_hdi,
		south_sudan_hdi,
		south_sudan_citations,
		"South Sudan",
		f"South Sudan (pred., {south_sudan_citations:,.0f})",
	)

	ax_hdi.set_xlabel("Birthplace HDI")
	ax_hdi.set_ylabel("Citations")
	ax_hdi.set_title("Birthplace HDI vs Citation Count")

	ax_hdi.text(
		0.97,
		0.97,
		(
			f"$R^2$ = {hdi_r_squared:.3f}\n"
			f"p-value = {hdi_p_value:.3g}\n"
			f"South Sudan est. citations = {south_sudan_citations:,.0f}\n"
			f"(at HDI = {south_sudan_hdi:.3f})"
		),
		transform=ax_hdi.transAxes,
		verticalalignment="top",
		horizontalalignment="right",
		bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8},
	)

	sns.regplot(
		data=df,
		x="Year",
		y="Citations",
		ax=ax_year,
		ci=None,
		scatter_kws={"s": 90, "color": "#1f77b4"},
		line_kws={"color": "#2ca02c", "linewidth": 2},
	)

	for _, row in df.iterrows():
		x = row["Year"]
		y = row["Citations"]
		_annotate_with_flag(ax_year, x, y, row["Country"], row["Name"])

	ax_year.scatter(
		x=next_advisor_year,
		y=next_advisor_citations,
		s=130,
		marker="X",
		color="#111111",
		zorder=5,
	)
	ax_year.annotate(
		f"Next Advisor (pred., {next_advisor_citations:,.0f})",
		(next_advisor_year, next_advisor_citations),
		textcoords="offset points",
		xytext=(10, -12),
		ha="left",
		va="top",
	)

	ax_year.set_xlabel("Year")
	ax_year.set_ylabel("Citations")
	ax_year.set_title("Year vs Citation Count")
	ax_year.set_xticks(sorted(set(df["Year"].astype(int).tolist() + [next_advisor_year])))

	ax_year.text(
		0.97,
		0.97,
		(
			f"$R^2$ = {year_r_squared:.3f}\n"
			f"p-value = {year_p_value:.3g}\n"
			f"Next advisor est. citations = {next_advisor_citations:,.0f}\n"
			f"(at year = {next_advisor_year})"
		),
		transform=ax_year.transAxes,
		verticalalignment="top",
		horizontalalignment="right",
		bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8},
	)

	fig.suptitle("Advisor Citation Regressions", fontsize=14)

	print(
		f"Estimated citations for South Sudan (HDI {south_sudan_hdi:.3f}): "
		f"{south_sudan_citations:,.0f}"
	)
	print(
		f"Estimated citations for next advisor in {next_advisor_year}: "
		f"{next_advisor_citations:,.0f}"
	)

	plt.tight_layout(rect=(0, 0, 1, 0.95))
	plt.show()


if __name__ == "__main__":
	plot_advisors_regression()

