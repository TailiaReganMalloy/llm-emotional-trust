from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
HYPOTHESES_MD_PATH = REPO_ROOT / "hypotheses.md"
HYPOTHESES_DIR = REPO_ROOT / "hypotheses"
OUTPUT_TEX_PATH = REPO_ROOT / "hypotheses.tex"


@dataclass
class HypothesisEntry:
    identifier: str
    statement: str
    notes: list[str]


def latex_escape(text: str) -> str:
    escaped = text
    escaped = escaped.replace("\\", r"\\textbackslash{}")
    escaped = escaped.replace("&", r"\\&")
    escaped = escaped.replace("%", r"\\%")
    escaped = escaped.replace("$", r"\\$")
    escaped = escaped.replace("#", r"\\#")
    escaped = escaped.replace("_", r"\\_")
    escaped = escaped.replace("{", r"\\{")
    escaped = escaped.replace("}", r"\\}")
    escaped = escaped.replace("~", r"\\textasciitilde{}")
    escaped = escaped.replace("^", r"\\textasciicircum{}")
    return escaped


def id_sort_key(identifier: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)([a-zA-Z]?)$", identifier)
    if not match:
        return (10**9, identifier.lower())
    num = int(match.group(1))
    suffix = match.group(2).lower()
    return (num, suffix)


def parse_hypotheses_markdown(markdown_path: Path) -> list[HypothesisEntry]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"^Hypothesis\s+(\d+[a-zA-Z]?)\s*:\s*(.+)$", re.IGNORECASE)

    entries: list[HypothesisEntry] = []
    current: HypothesisEntry | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            if current is not None:
                entries.append(current)
            identifier = match.group(1).lower()
            statement = match.group(2).strip()
            current = HypothesisEntry(identifier=identifier, statement=statement, notes=[])
            continue

        if current is not None:
            current.notes.append(line)

    if current is not None:
        entries.append(current)

    if not entries:
        raise ValueError(f"No hypotheses found in {markdown_path}")

    entries.sort(key=lambda entry: id_sort_key(entry.identifier))
    return entries


def hypothesis_paths(identifier: str) -> tuple[Path, Path, Path]:
    folder = HYPOTHESES_DIR / f"hypothesis{identifier}"
    script = folder / f"hypothesis{identifier}.py"
    txt = folder / f"hypothesis{identifier}.txt"
    png = folder / f"hypothesis{identifier}.png"
    return script, txt, png


def run_hypothesis_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing hypothesis script: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (
            f"Failed running {script_path.name}\\n"
            f"stdout:\\n{result.stdout}\\n"
            f"stderr:\\n{result.stderr}"
        )
        raise RuntimeError(message)


def strip_leading_subsubsection(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("\\subsubsection{"):
        return "\n".join(lines[1:]).lstrip()
    return text


def append_result_section(out: list[str], identifier: str, title: str | None = None) -> None:
    _, txt_path, png_path = hypothesis_paths(identifier)
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing generated report: {txt_path}")
    if not png_path.exists():
        raise FileNotFoundError(f"Missing generated figure: {png_path}")

    report_text = txt_path.read_text(encoding="utf-8").strip()
    cleaned_report = strip_leading_subsubsection(report_text)

    section_title = title or f"Hypothesis {identifier}"
    out.append("")
    out.append(f"\\subsubsection{{{section_title}}}")
    out.append(cleaned_report)
    out.append("")
    out.append("\\begin{figure}[h]")
    out.append("\\centering")
    out.append(f"\\includegraphics[width=0.85\\linewidth]{{hypotheses/hypothesis{identifier}/hypothesis{identifier}.png}}")
    out.append(f"\\caption{{Visualization for {section_title}.}}")
    out.append(f"\\label{{fig:hypothesis{identifier}}}")
    out.append("\\end{figure}")


def generate_latex(entries: list[HypothesisEntry]) -> str:
    out: list[str] = []

    out.append("\\subsection{Hypotheses}")
    out.append("\\begin{enumerate}")
    for entry in entries:
        display_id = entry.identifier.upper() if entry.identifier.endswith("a") else entry.identifier
        out.append(f"\\item \\textbf{{Hypothesis {display_id}.}} {latex_escape(entry.statement)}")
        for note in entry.notes:
            out.append(f"{latex_escape(note)}")
    out.append("\\end{enumerate}")
    out.append("")

    out.append("\\section{Results}")
    out.append("\\subsection{Hypotheses And Statistical Results}")

    for entry in entries:
        display_id = entry.identifier.upper() if entry.identifier.endswith("a") else entry.identifier
        append_result_section(out, entry.identifier, title=f"Hypothesis {display_id}")

    out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    entries = parse_hypotheses_markdown(HYPOTHESES_MD_PATH)

    print(f"Found {len(entries)} hypotheses in {HYPOTHESES_MD_PATH}")

    for entry in entries:
        script_path, _, _ = hypothesis_paths(entry.identifier)
        print(f"Running hypothesis {entry.identifier}: {script_path}")
        run_hypothesis_script(script_path)

    tex_content = generate_latex(entries)
    OUTPUT_TEX_PATH.write_text(tex_content, encoding="utf-8")

    print(f"Generated: {OUTPUT_TEX_PATH}")


if __name__ == "__main__":
    main()
