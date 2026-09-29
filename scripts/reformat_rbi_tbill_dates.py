"""Reformat RBI 91-day T-bill CSV dates and reverse chronological order."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "data" / "processed" / "risk_free" / "rbi_91d_tbill.csv"


def main() -> None:
    with SRC.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "date" not in fieldnames:
            raise SystemExit(f"Expected a 'date' column in {SRC}")
        rows = list(reader)

    for row in rows:
        row["date"] = datetime.strptime(row["date"], "%d-%b-%Y").strftime("%Y-%m-%d")

    rows.reverse()

    with SRC.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {SRC}")
    print(rows[0])
    print(rows[-1])


if __name__ == "__main__":
    main()
