import csv
import time
import requests
from datetime import datetime, timedelta, timezone

# ---------- Config ----------
BASE_URL = "https://bugzilla.mozilla.org/rest"
OUTPUT_CSV = "bugs_since.csv"
START_DATE = "2025-01-01"  # UTC; can also be "2025-01-01T00:00:00Z"
RATE_LIMIT = 0.4  # seconds between requests to be polite
CHUNK_CAP = 10_000  # Bugzilla hard cap that triggers time-splitting
TIMEOUT = 60

# ---------- Fields we want (and will write to CSV) ----------
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

# Exactly match Bugzilla field names for include_fields
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

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}

def parse_start_date(s: str) -> datetime:
    """Accept YYYY-MM-DD or ISO8601; return timezone-aware UTC datetime."""
    try:
        # Try full ISO 8601 with trailing 'Z'
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        # Fallback: just a date
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt

def iso_utc(dt: datetime) -> str:
    """Return strict UTC ISO8601 with Z."""
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def fetch_chunk(session: requests.Session, start_iso: str) -> list[dict]:
    """
    Fetch up to CHUNK_CAP bugs created at or after start_iso.
    Limit to include_fields; return list of bug dicts.
    """
    params = {
        "creation_time": start_iso,               # >= this moment
        "include_fields": ",".join(INCLUDE_FIELDS),
        "limit": str(CHUNK_CAP),
        # Note: Bugzilla will cap anyway; explicit limit makes intent clear.
    }
    url = f"{BASE_URL}/bug"
    r = session.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("bugs", [])

def filter_bug(b: dict) -> bool:
    """Keep only is_confirmed == true and dupe_of == null."""
    return bool(b.get("is_confirmed")) and (b.get("dupe_of") in (None, ""))

def extract_row(b: dict) -> dict:
    """Map a Bugzilla bug to our CSV row schema."""
    # flags: collect all type_id values
    flag_type_ids = []
    for fl in (b.get("flags") or []):
        tid = fl.get("type_id")
        if tid is not None:
            flag_type_ids.append(str(tid))

    # blocks is an array of ints; join as comma-separated
    blocks = b.get("blocks") or []
    blocks_str = ",".join(str(x) for x in blocks) if blocks else ""

    return {
        "bug_id": b.get("id", ""),
        "flag_type_ids": ",".join(flag_type_ids),
        "resolution": b.get("resolution", ""),
        "version": b.get("version", ""),
        "status": b.get("status", ""),
        "summary": (b.get("summary") or "").replace("\n", " ").strip(),
        "platform": b.get("platform", ""),
        "url": b.get("url", ""),
        "classification": b.get("classification", ""),
        "priority": b.get("priority", ""),
        "creation_time": b.get("creation_time", ""),
        "component": b.get("component", ""),
        "severity": b.get("severity", ""),
        "product": b.get("product", ""),
        "blocks": blocks_str,
    }

def main():
    start_dt = parse_start_date(START_DATE)
    seen_ids = set()          # de-dup across chunks, just in case
    total_kept = 0
    chunk_ix = 0

    with requests.Session() as session, open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
        writer.writeheader()

        current_start = start_dt

        while True:
            chunk_ix += 1
            start_iso = iso_utc(current_start)
            print(f"[Chunk {chunk_ix}] Querying from creation_time >= {start_iso} ...")
            bugs = fetch_chunk(session, start_iso)

            if not bugs:
                print("No more bugs returned. Done.")
                break

            # Sort by creation_time to get a reliable last timestamp
            # (strings like "2025-01-01T00:00:42Z" sort correctly)
            bugs.sort(key=lambda x: x.get("creation_time", ""))

            # Filter and write rows
            kept_in_chunk = 0
            for b in bugs:
                # Skip already seen
                bid = b.get("id")
                if bid in seen_ids:
                    continue
                # Filter: confirmed & not duplicate
                if not filter_bug(b):
                    continue
                row = extract_row(b)
                writer.writerow(row)
                seen_ids.add(bid)
                kept_in_chunk += 1

            total_kept += kept_in_chunk
            print(f"  Retrieved: {len(bugs):5d} | kept (filtered & unique): {kept_in_chunk:5d} | total kept: {total_kept}")

            # If we hit the cap, advance the start time to last bug's creation_time + 1s
            if len(bugs) >= CHUNK_CAP:
                last_ct = bugs[-1].get("creation_time")
                if not last_ct:
                    # safety: if last bug has no creation_time (shouldn't happen), stop
                    print("  Last bug missing creation_time. Stopping to avoid loop.")
                    break
                # Move start forward by 1 second to avoid re-fetching the same last record
                last_dt = parse_start_date(last_ct)
                current_start = last_dt + timedelta(seconds=1)
                print(f"  Cap reached ({CHUNK_CAP}). Advancing start to {iso_utc(current_start)}")
                time.sleep(RATE_LIMIT)
                continue

            # Otherwise, we’re done (final chunk < cap)
            print("Final chunk under cap. Completed.")
            break

if __name__ == "__main__":
    main()
