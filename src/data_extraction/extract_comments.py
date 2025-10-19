import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://bugzilla.mozilla.org/rest"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_CSV = os.path.join(DATA_DIR, "bugs_since.csv")
OUTPUT_COMMENTS = os.path.join(DATA_DIR, "bugs_comments.csv")

# Tuning
RATE_LIMIT = 0.0        # per-request sleep in main loop; concurrency handles throughput
TEST_LIMIT = 40000
START_OFFSET = 10000
APPEND_MODE = True
MAX_WORKERS = 16        
TIMEOUT = 60
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

def build_session() -> requests.Session:
    """Session with keep-alive, pooling, and robust retries."""
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS*2, pool_maxsize=MAX_WORKERS*2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

def fetch_bug_comments(session: requests.Session, bug_id: int):
    """
    Doc-compliant: GET /rest/bug/{id}/comment
    Returns a list of CSV-ready rows: [bug_id, comment_id, creation_time, text]
    """
    url = f"{BASE_URL}/bug/{bug_id}/comment"
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    bucket = (data.get("bugs") or {}).get(str(bug_id)) or {}
    comments = bucket.get("comments", [])
    rows = []
    for c in comments:
        rows.append([
            bug_id,
            c.get("id", ""),                     # globally unique comment id per docs
            c.get("creation_time", "") or c.get("time", ""),
            (c.get("text", "") or "").replace("\n", " ").strip(),
        ])
    return rows

def main():
    # Determine file mode + header behavior
    file_mode = "a" if APPEND_MODE else "w"
    write_header = not APPEND_MODE
    if APPEND_MODE:
        try:
            with open(OUTPUT_COMMENTS, "r", encoding="utf-8") as f:
                write_header = not bool(f.readline().strip())
        except FileNotFoundError:
            write_header = True

    with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_COMMENTS, mode=file_mode, newline="", encoding="utf-8") as outcsv:

        reader = csv.reader(infile)
        writer = csv.writer(outcsv)

        # Input header (first column must be bug_id)
        in_header = next(reader, None)

        if write_header:
            writer.writerow(["bug_id", "comment_id", "creation_time", "text"])

        # Skip offset safely
        for _ in range(START_OFFSET):
            try:
                next(reader)
            except StopIteration:
                print("Reached EOF while skipping; nothing to do.")
                return
        print(f"Skipped first {START_OFFSET} bugs")

        # Collect target bug IDs (respect TEST_LIMIT)
        bug_ids = []
        for row in reader:
            if TEST_LIMIT and len(bug_ids) >= TEST_LIMIT:
                break
            if row and row[0].isdigit():
                bug_ids.append(int(row[0]))

        if not bug_ids:
            print("No bug IDs to process.")
            return

        session = build_session()
        start = time.perf_counter()
        submitted = 0
        written = 0

        # Submit concurrent requests
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_to_bug = {ex.submit(fetch_bug_comments, session, bid): bid for bid in bug_ids}
            submitted = len(future_to_bug)

            # Consume as they complete; write rows in main thread
            for idx, fut in enumerate(as_completed(future_to_bug), 1):
                bid = future_to_bug[fut]
                try:
                    rows = fut.result()
                except requests.HTTPError as e:
                    # Log and continue
                    print(f"[{idx}/{submitted}] bug {bid}: HTTPError {e.response.status_code} — skipping")
                    continue
                except Exception as e:
                    print(f"[{idx}/{submitted}] bug {bid}: error {e!r} — skipping")
                    continue

                if rows:
                    writer.writerows(rows)
                    written += 1

                if idx % 100 == 0:
                    elapsed = time.perf_counter() - start
                    print(f"Processed {idx}/{submitted} bugs, wrote comments for {written}, "
                          f"elapsed {elapsed:.1f}s (~{elapsed/max(idx,1):.3f}s/bug)")
                if RATE_LIMIT:
                    time.sleep(RATE_LIMIT)

        elapsed = time.perf_counter() - start
        print(f"Done. Bugs processed: {submitted}, with comments: {written}. "
              f"Elapsed {elapsed:.1f}s (~{elapsed/max(submitted,1):.3f}s/bug)")

if __name__ == "__main__":
    main()
