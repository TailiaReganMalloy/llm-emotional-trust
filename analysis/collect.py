from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def db_env_from_values(values: dict[str, str]) -> dict[str, str]:
    db_uri = values.get("DB_URI")
    if db_uri:
        parsed = urlparse(db_uri)
        query = parse_qs(parsed.query)
        sslmode = query.get("sslmode", ["require"])[0]
        db_name = parsed.path.lstrip("/")
        if not (parsed.hostname and parsed.username and db_name):
            raise ValueError("DB_URI is missing required components.")
        return {
            "PGHOST": parsed.hostname,
            "PGPORT": str(parsed.port or 5432),
            "PGUSER": parsed.username,
            "PGPASSWORD": parsed.password or "",
            "PGDATABASE": db_name,
            "PGSSLMODE": sslmode,
        }

    required = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing = [k for k in required if not values.get(k)]
    if missing:
        raise ValueError(f"Missing DB values in .env: {', '.join(missing)}")

    return {
        "PGHOST": values["DB_HOST"],
        "PGPORT": values.get("DB_PORT", "5432"),
        "PGUSER": values["DB_USER"],
        "PGPASSWORD": values["DB_PASSWORD"],
        "PGDATABASE": values["DB_NAME"],
        "PGSSLMODE": values.get("DB_SSLMODE", "require"),
    }


def run_psql(sql: str, db_env: dict[str, str]) -> str:
    env = os.environ.copy()
    env.update(db_env)
    command = ["psql", "-X", "-A", "-F", "\t", "-q", "-t", "-v", "ON_ERROR_STOP=1", "-c", sql]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "psql command failed")
    return completed.stdout


def list_objects(db_env: dict[str, str]) -> list[tuple[str, str, str]]:
    sql = """
    SELECT n.nspname AS schema_name,
           c.relname AS object_name,
           CASE c.relkind
               WHEN 'r' THEN 'table'
               WHEN 'p' THEN 'partitioned_table'
               WHEN 'v' THEN 'view'
               WHEN 'm' THEN 'materialized_view'
               WHEN 'f' THEN 'foreign_table'
               ELSE 'other'
           END AS object_type
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
    ORDER BY n.nspname, c.relname;
    """
    rows = []
    for line in run_psql(sql, db_env).splitlines():
        line = line.strip()
        if not line:
            continue
        schema, name, object_type = line.split("\t")
        if schema in SYSTEM_SCHEMAS:
            continue
        rows.append((schema, name, object_type))
    return rows


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def export_object(schema: str, name: str, out_path: Path, db_env: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(db_env)

    qualified = f"{quote_ident(schema)}.{quote_ident(name)}"
    sql = f"\\copy (SELECT * FROM {qualified}) TO '{str(out_path).replace("'", "''")}' WITH (FORMAT CSV, HEADER true)"
    command = ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", sql]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"Failed exporting {schema}.{name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all non-system Postgres tables/views to CSV files in data/all."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file with DB credentials (default: ./.env).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/all",
        help="Output directory for CSV exports (default: data/all).",
    )
    parser.add_argument(
        "--skip-views",
        action="store_true",
        help="Skip exporting views and materialized views.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each object as it is exported.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_file).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}", file=sys.stderr)
        return 1

    if shutil_which("psql") is None:
        print(
            "Error: 'psql' command not found. Install PostgreSQL client tools first.",
            file=sys.stderr,
        )
        return 1

    values = load_env_file(env_path)
    db_env = db_env_from_values(values)

    out_dir.mkdir(parents=True, exist_ok=True)
    objects = list_objects(db_env)
    if not objects:
        print("No exportable tables or views found.")
        return 0

    exported = 0
    failed = 0

    for schema, name, object_type in objects:
        is_view = object_type in {"view", "materialized_view"}
        if args.skip_views and is_view:
            continue

        filename = f"{sanitize_filename(schema)}.{sanitize_filename(name)}.csv"
        out_path = out_dir / filename

        if args.verbose:
            print(f"Exporting {schema}.{name} ({object_type}) -> {out_path}")

        try:
            export_object(schema, name, out_path, db_env)
            exported += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed: {schema}.{name}: {exc}", file=sys.stderr)

    print(f"Export complete. Exported: {exported}, Failed: {failed}, Output: {out_dir}")
    return 0 if failed == 0 else 2


def shutil_which(command: str) -> str | None:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
