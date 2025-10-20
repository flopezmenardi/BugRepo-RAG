import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import requests

BASE_URL = "https://bugzilla.mozilla.org/rest"
OUTPUT_CSV = "test_bugs_2025_20.csv"
START_DATE = "2025-01-01"
LIMIT = 20  # default number of bugs to collect; adjust as needed
TIMEOUT = 60

CSV_FIELDS = [
    "bug_id",
    "flag_type_ids",
    "resolution",
    "version",
    "status",
    "summary",
    "platform",
    "url",
    "classification",
    "priority",
    "creation_time",
    "component",
    "severity",
    "product",
    "blocks",
]

INCLUDE_FIELDS = [
    "id",
    "flags",
    "resolution",
    "version",
    "status",
    "summary",
    "platform",
    "url",
    "classification",
    "priority",
    "is_confirmed",
    "creation_time",
    "dupe_of",
    "component",
    "severity",
    "product",
    "blocks",
]

HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


def _parse_start_date(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _filter_bug(bug: Dict) -> bool:
    return bool(bug.get("is_confirmed")) and bug.get("dupe_of") in (None, "")


def _extract_row(bug: Dict) -> Dict[str, str]:
    flags = [str(flag.get("type_id")) for flag in bug.get("flags") or [] if flag.get("type_id") is not None]
    blocks = bug.get("blocks") or []
    return {
        "bug_id": bug.get("id", ""),
        "flag_type_ids": ",".join(flags),
        "resolution": bug.get("resolution", ""),
        "version": bug.get("version", ""),
        "status": bug.get("status", ""),
        "summary": (bug.get("summary") or "").replace("\n", " ").strip(),
        "platform": bug.get("platform", ""),
        "url": bug.get("url", ""),
        "classification": bug.get("classification", ""),
        "priority": bug.get("priority", ""),
        "creation_time": bug.get("creation_time", ""),
        "component": bug.get("component", ""),
        "severity": bug.get("severity", ""),
        "product": bug.get("product", ""),
        "blocks": ",".join(str(b) for b in blocks),
    }


def main(limit: int = LIMIT) -> None:
    start_iso = _parse_start_date(START_DATE)
    params = {
        "creation_time": start_iso,
        "include_fields": ",".join(INCLUDE_FIELDS),
        "limit": str(limit),
    }

    response = requests.get(
        f"{BASE_URL}/bug", params=params, headers=HEADERS, timeout=TIMEOUT
    )
    response.raise_for_status()
    bugs = response.json().get("bugs", [])

    filtered_rows: List[Dict[str, str]] = []
    for bug in bugs:
        if len(filtered_rows) >= limit:
            break
        if not _filter_bug(bug):
            continue
        filtered_rows.append(_extract_row(bug))

    output_path = Path(OUTPUT_CSV)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(filtered_rows[:limit])

    print(f"Wrote {len(filtered_rows[:limit])} bugs to {output_path.resolve()}")


if __name__ == "__main__":
    main()
