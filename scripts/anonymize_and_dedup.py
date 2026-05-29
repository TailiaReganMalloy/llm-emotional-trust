#!/usr/bin/env python3
"""Redact identifying information and deduplicate sensitive CSV exports.

Targets:
- .env
- SiteCode/public/api-config.js
- Dataset/raw/Responses.csv
- Dataset/raw/Submissions.csv
- Dataset/raw/databases/public.student_responses.csv
- Dataset/raw/databases/public.interactive_submissions.csv
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSV_TARGETS = [
    ROOT / "Dataset/raw/Responses.csv",
    ROOT / "Dataset/raw/Submissions.csv",
    ROOT / "Dataset/raw/databases/public.student_responses.csv",
    ROOT / "Dataset/raw/databases/public.interactive_submissions.csv",
]

ENV_FILE = ROOT / ".env"
API_CONFIG = ROOT / "SiteCode/public/api-config.js"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
NAME_TOKEN_RE = re.compile(r"tailia|malloy|tailiamalloy|tailia-malloy", re.IGNORECASE)


def anonymize_email(email: str) -> str:
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:10]
    return f"participant+{digest}@anon.invalid"


def anonymize_text(value: str) -> str:
    value = EMAIL_RE.sub(lambda m: anonymize_email(m.group(0)), value)
    value = NAME_TOKEN_RE.sub("redacted", value)
    return value


def process_csv(path: Path) -> tuple[int, int]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    if not rows:
        return 0, 0

    header, body = rows[0], rows[1:]

    transformed: list[list[str]] = []
    for row in body:
        transformed.append([anonymize_text(cell) for cell in row])

    seen: set[tuple[str, ...]] = set()
    deduped: list[list[str]] = []
    duplicate_count = 0

    for row in transformed:
        key = tuple(row)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(deduped)

    return len(body), duplicate_count


def process_env(path: Path) -> None:
    if not path.exists():
        return

    replacements = {
        "DB_USER": "REDACTED_USER",
        "DB_PASSWORD": "REDACTED_PASSWORD",
        "DB_HOST": "REDACTED_HOST",
        "DB_NAME": "REDACTED_DB",
        "DB_PORT": "5432",
        "DB_URI": "postgres://REDACTED_USER:REDACTED_PASSWORD@REDACTED_HOST:5432/REDACTED_DB",
        "HEROKU_CLI": "heroku pg:psql REDACTED_DATABASE --app anonymized-app",
    }

    lines = path.read_text(encoding="utf-8").splitlines()
    output = []
    for line in lines:
        if "=" not in line:
            output.append(line)
            continue
        key, _value = line.split("=", 1)
        if key in replacements:
            output.append(f"{key}={replacements[key]}")
        else:
            output.append(f"{key}=REDACTED")

    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def process_api_config(path: Path) -> None:
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"https://[^'\"]+",
        "https://anonymized-app.herokuapp.com",
        text,
    )
    text = anonymize_text(text)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    print("Applying anonymization and deduplication...")

    process_env(ENV_FILE)
    process_api_config(API_CONFIG)

    for target in CSV_TARGETS:
        if not target.exists():
            print(f"SKIP {target.relative_to(ROOT)} (missing)")
            continue
        total_rows, duplicate_rows = process_csv(target)
        print(
            f"OK   {target.relative_to(ROOT)} rows={total_rows} duplicates_removed={duplicate_rows}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
