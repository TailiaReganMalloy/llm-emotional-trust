#!/usr/bin/env python3
"""Generate supplementaryMaterials.tex from questionnaire and Vue sources.

Outputs:
- Original questionnaire wording (from oldHypotheses/combine.py mappings)
- Response options (from questionnaire spreadsheets when available)
- Demographics prompts and options
- User entry locations in interactiveExplain.vue and staticExplain.vue
"""

from __future__ import annotations

import ast
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent

OLD_COMBINE_PATH = ROOT / "oldHypotheses" / "combine.py"
METRICS_PATH = ROOT / "oldHypotheses" / "metrics.py"
INTERACTIVE_VUE_PATH = ROOT / "interactiveExplain.vue"
STATIC_VUE_PATH = ROOT / "staticExplain.vue"

QUESTIONNAIRE_XLSX_PATHS = [
	ROOT / "data" / "raw" / "QuestionnaireInteractive.xlsx",
	ROOT / "data" / "raw" / "QuestionnaireText.xlsx",
]

DEMOGRAPHICS_CSV_PATHS = [
	ROOT / "data" / "raw" / "DemographicsInteractive1.csv",
	ROOT / "data" / "raw" / "DemographicsInteractive2.csv",
	ROOT / "data" / "raw" / "DemographicsText1.csv",
	ROOT / "data" / "raw" / "DemographicsText2.csv",
]

OUTPUT_TEX_PATH = ROOT / "supplementaryMaterials.tex"


LIKERT_OPTIONS = [
	"Strongly Disagree",
	"Disagree",
	"Neither Agree nor Disagree",
	"Agree",
	"Strongly Agree",
]

ANALYTICAL_CANONICALS = {
	"AI Deceptive",
	"AI Dishonest",
	"AI Suspicious",
	"AI Wary",
	"AI Harm",
	"AI Confident",
	"AI Security",
	"AI Trustworthy",
	"AI Reliable",
	"AI Trust",
	"AI Deceptive Post",
	"AI Dishonest Post",
	"AI Suspicious Post",
	"AI Wary Post",
	"AI Wary/Deceptive Post",
	"AI Harm Post",
	"AI Harm Post 1",
	"AI Harm Post 2",
	"AI Confident Post",
	"AI Security Post",
	"AI Trustworthy Post",
	"AI Reliable Post",
	"AI Trust Post",
	"AI Trust Post 2",
}

REDACTION_MARKERS = {
	"CONSENT_REVOKED",
	"DATA_EXPIRED",
	"Not Applicable",
	"",
}


@dataclass
class QuestionRecord:
	canonical: str
	wordings: list[str] = field(default_factory=list)
	options: list[str] = field(default_factory=list)


def normalize_column_name(value: object) -> str:
	text = str(value or "")
	return "".join(ch for ch in text.lower() if ch.isalnum())


def latex_escape(value: object) -> str:
	text = str(value)
	replacements = {
		"\\": r"\textbackslash{}",
		"&": r"\&",
		"%": r"\%",
		"$": r"\$",
		"#": r"\#",
		"_": r"\_",
		"{": r"\{",
		"}": r"\}",
		"~": r"\textasciitilde{}",
		"^": r"\textasciicircum{}",
	}
	for old, new in replacements.items():
		text = text.replace(old, new)
	return text


def read_python_module(path: Path) -> ast.Module:
	return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def get_assign_node(module: ast.Module, var_name: str) -> ast.AST | None:
	for node in module.body:
		if not isinstance(node, ast.Assign):
			continue
		for target in node.targets:
			if isinstance(target, ast.Name) and target.id == var_name:
				return node.value
	return None


def extract_normalized_call_arg(node: ast.AST) -> str | None:
	if not isinstance(node, ast.Call):
		return None
	if not isinstance(node.func, ast.Name) or node.func.id != "_normalize_column_name":
		return None
	if not node.args:
		return None
	first = node.args[0]
	if isinstance(first, ast.Constant) and isinstance(first.value, str):
		return first.value.strip()
	return None


def extract_str_constant(node: ast.AST) -> str | None:
	if isinstance(node, ast.Constant) and isinstance(node.value, str):
		return node.value.strip()
	return None


def load_question_mapping() -> list[tuple[str, str]]:
	module = read_python_module(OLD_COMBINE_PATH)
	mapping_node = get_assign_node(module, "normalized_column_rename_map")
	if not isinstance(mapping_node, ast.Dict):
		return []

	pairs: list[tuple[str, str]] = []
	for key_node, value_node in zip(mapping_node.keys, mapping_node.values):
		question = extract_normalized_call_arg(key_node)
		canonical = extract_str_constant(value_node)
		if question and canonical:
			pairs.append((question, canonical))
	return pairs


def load_consent_drop_items() -> list[str]:
	module = read_python_module(OLD_COMBINE_PATH)
	drop_node = get_assign_node(module, "drop_column_norms")
	if not isinstance(drop_node, ast.Set):
		return []

	out: list[str] = []
	for elt in drop_node.elts:
		question = extract_normalized_call_arg(elt)
		if not question:
			continue
		if question == "Custom study tncs accepted at":
			continue
		if "Please head over to the following link" in question:
			continue
		out.append(question)
	return out


def load_questionnaire_value_map() -> dict[str, set[str]]:
	values_by_normalized_column: dict[str, set[str]] = defaultdict(set)

	for path in QUESTIONNAIRE_XLSX_PATHS:
		if not path.exists():
			continue
		try:
			frame = pd.read_excel(path)
		except Exception as exc:
			print(f"Warning: could not read {path}: {exc}")
			continue

		for column in frame.columns:
			normalized = normalize_column_name(column)
			series = frame[column].dropna()
			for value in series:
				cleaned = " ".join(str(value).split())
				if cleaned:
					values_by_normalized_column[normalized].add(cleaned)

	return values_by_normalized_column


def lookup_observed_values(question: str, value_map: dict[str, set[str]]) -> list[str]:
	normalized = normalize_column_name(question)
	values = sorted(value_map.get(normalized, set()), key=lambda x: x.lower())
	return [v for v in values if v and v not in REDACTION_MARKERS]


def emotional_pair_options_from_metrics() -> dict[str, list[str]]:
	module = read_python_module(METRICS_PATH)
	pair_map_node = get_assign_node(module, "emotional_polarity")
	if not isinstance(pair_map_node, ast.Dict):
		return {}

	out: dict[str, list[str]] = {}
	for key_node, value_node in zip(pair_map_node.keys, pair_map_node.values):
		canonical = extract_str_constant(key_node)
		if not canonical:
			continue
		if not isinstance(value_node, ast.Dict):
			continue
		endpoints: list[str] = []
		for polarity_key in value_node.keys:
			endpoint = extract_str_constant(polarity_key)
			if endpoint:
				endpoints.append(endpoint.title())
		if endpoints:
			options = sorted(set(endpoints), key=str.lower)
			out[canonical] = options

			match = re.match(r"^AI systems are (\d+)$", canonical)
			if match:
				idx = match.group(1)
				out[f"AI systems are {idx} Post"] = options
				out[f"AI systems are Post {idx}"] = options
			else:
				out[f"{canonical} Post"] = options
	return out


def summarize_options(canonical: str, question: str, observed: list[str], emotional_map: dict[str, list[str]]) -> list[str]:
	if canonical in ANALYTICAL_CANONICALS:
		return LIKERT_OPTIONS.copy()

	if canonical in emotional_map:
		return emotional_map[canonical]

	question_lower = question.lower()
	if "what is your age" in question_lower:
		return ["Numeric entry (free response)"]

	if "email adress" in question_lower or "prolific" in question_lower:
		return ["Open text entry (email or Prolific ID)"]

	if "please head over to the following link" in question_lower:
		return ["Open text response"]

	if not observed:
		return ["Open text response"]

	# Use observed option values when the list is manageable and clearly categorical.
	if len(observed) > 15 or any(len(item) > 80 for item in observed):
		return ["Open text response"]

	return observed


def build_question_records() -> list[QuestionRecord]:
	mapping_pairs = load_question_mapping()
	consent_items = load_consent_drop_items()
	value_map = load_questionnaire_value_map()
	emotional_map = emotional_pair_options_from_metrics()

	by_canonical: dict[str, QuestionRecord] = {}

	for question, canonical in mapping_pairs:
		if canonical not in by_canonical:
			observed = lookup_observed_values(question, value_map)
			by_canonical[canonical] = QuestionRecord(
				canonical=canonical,
				wordings=[question],
				options=summarize_options(canonical, question, observed, emotional_map),
			)
			continue

		if question not in by_canonical[canonical].wordings:
			by_canonical[canonical].wordings.append(question)

	for consent_question in consent_items:
		canonical = f"Consent Item: {consent_question[:30]}..."
		by_canonical[canonical] = QuestionRecord(
			canonical=canonical,
			wordings=[consent_question],
			options=["Consent checkbox/agreement"],
		)

	return list(by_canonical.values())


def load_demographics_values() -> dict[str, list[str]]:
	values: dict[str, set[str]] = defaultdict(set)
	demographic_fields = {
		"Gender",
		"Ethnicity",
		"Age",
		"Sex",
		"Ethnicity simplified",
		"Country of birth",
		"Country of residence",
		"Nationality",
		"Language",
		"Student status",
		"Employment status",
	}

	for path in DEMOGRAPHICS_CSV_PATHS:
		if not path.exists():
			continue
		with path.open("r", encoding="utf-8-sig", newline="") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				for field in demographic_fields:
					if field not in row:
						continue
					value = " ".join(str(row[field] or "").split())
					if not value or value in REDACTION_MARKERS:
						continue
					values[field].add(value)

	out: dict[str, list[str]] = {}
	for field, opts in values.items():
		ordered = sorted(opts, key=str.lower)
		if field == "Age":
			out[field] = ["Numeric entry (free response)"]
			continue
		if field in {"Country of birth", "Country of residence", "Nationality", "Language"}:
			sample = ordered[:12]
			if len(ordered) > 12:
				sample.append(f"... ({len(ordered) - 12} additional observed values)")
			out[field] = ["Open text response"] + sample
			continue
		if len(ordered) > 18:
			sample = ordered[:18]
			sample.append(f"... ({len(ordered) - 18} additional observed values)")
			out[field] = sample
		else:
			out[field] = ordered

	return out


def extract_vue_entry_points(path: Path, interactive: bool) -> list[tuple[str, list[str]]]:
	content = path.read_text(encoding="utf-8")
	points: list[tuple[str, list[str]]] = []

	prompts = re.findall(r"'([^']*required to[^']*)'", content)
	prompts = list(dict.fromkeys(prompts))

	placeholder_match = re.search(r'placeholder="([^"]+)"', content)
	placeholder = placeholder_match.group(1) if placeholder_match else ""

	textarea_details = [
		"Control: textarea with id=section-response and v-model=currentResponse",
	]
	if prompts:
		textarea_details.append("Prompt text:")
		textarea_details.extend([f"- {item}" for item in prompts])
	if placeholder:
		textarea_details.append(f"Placeholder: {placeholder}")

	points.append(("Section reflection text entry", textarea_details))

	if interactive:
		points.append(
			(
				"Embedded interactive widget edits",
				[
					"Users can edit sentence/pair fields rendered in currentSection.html via dynamic scripts.",
					"Updates are logged via window.logInteractiveSubmission(originalText, updatedText, submissionType).",
					"Continue is disabled until at least one interactive submission exists in a section.",
				],
			)
		)

	return points


def render_options_itemize(options: Iterable[str], indent: str = "") -> list[str]:
	lines = [f"{indent}\\begin{{itemize}}"]
	for option in options:
		lines.append(f"{indent}  \\item {latex_escape(option)}")
	lines.append(f"{indent}\\end{{itemize}}")
	return lines


def build_tex_document() -> str:
	questionnaire = build_question_records()
	demographics = load_demographics_values()
	interactive_points = extract_vue_entry_points(INTERACTIVE_VUE_PATH, interactive=True)
	static_points = extract_vue_entry_points(STATIC_VUE_PATH, interactive=False)

	lines: list[str] = []
	lines.append("% Auto-generated by list.py")
	lines.append(f"% Generation date: {date.today().isoformat()}")
	lines.append("\\section*{Supplementary Materials}")
	lines.append("\\subsection*{1. Original Questionnaire Wording and Response Options}")
	lines.append("\\begin{enumerate}")

	for record in questionnaire:
		lines.append("  \\item")
		lines.append(f"  \\textbf{{Canonical variable:}} {latex_escape(record.canonical)}\\\\")
		lines.append("  \\textbf{Original wording:}")
		lines.extend(render_options_itemize(record.wordings, indent="  "))
		lines.append("  \\textbf{Response options:}")
		lines.extend(render_options_itemize(record.options, indent="  "))

	lines.append("\\end{enumerate}")

	lines.append("\\subsection*{2. Demographic Questionnaire Items and Options}")
	lines.append("\\begin{enumerate}")
	for field_name in sorted(demographics.keys(), key=str.lower):
		lines.append("  \\item")
		lines.append(f"  \\textbf{{Question field:}} {latex_escape(field_name)}")
		lines.append("  \\textbf{Options:}")
		lines.extend(render_options_itemize(demographics[field_name], indent="  "))
	lines.append("\\end{enumerate}")

	lines.append("\\subsection*{3. User Entry Points in interactiveExplain.vue}")
	lines.append("\\begin{enumerate}")
	for heading, details in interactive_points:
		lines.append("  \\item")
		lines.append(f"  \\textbf{{{latex_escape(heading)}}}")
		lines.extend(render_options_itemize(details, indent="  "))
	lines.append("\\end{enumerate}")

	lines.append("\\subsection*{4. User Entry Points in staticExplain.vue}")
	lines.append("\\begin{enumerate}")
	for heading, details in static_points:
		lines.append("  \\item")
		lines.append(f"  \\textbf{{{latex_escape(heading)}}}")
		lines.extend(render_options_itemize(details, indent="  "))
	lines.append("\\end{enumerate}")

	return "\n".join(lines) + "\n"


def main() -> None:
	tex = build_tex_document()
	OUTPUT_TEX_PATH.write_text(tex, encoding="utf-8")
	print(f"Wrote {OUTPUT_TEX_PATH}")


if __name__ == "__main__":
	main()
