import requests
import csv
import time

# Config
BASE_URL = "https://bugzilla.mozilla.org/rest"
INPUT_CSV = "data/sample_bugs.csv"   # existing metadata file
OUTPUT_COMMENTS = "data/bugs_comments.csv"
RATE_LIMIT = 0.2  # seconds between requests to be nice to API
TEST_LIMIT = 10   # Only process first N bugs for testing (set to None for all)


def get_bug_comments(bug_id):
    url = f"{BASE_URL}/bug/{bug_id}/comment"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if "bugs" in data and str(bug_id) in data["bugs"]:
            return data["bugs"][str(bug_id)]["comments"]
    return []


def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_COMMENTS, mode="w", newline="", encoding="utf-8") as comments_file:

        reader = csv.reader(infile)
        comments_writer = csv.writer(comments_file)

        # Skip header row of input CSV
        header = next(reader)
        bug_id_idx = 0  # first column is bug_id

        # Write header for comments CSV
        comments_writer.writerow(["bug_id", "comment_id", "author", "creation_time", "text"])

        processed_count = 0
        for row in reader:
            # Limit for testing
            if TEST_LIMIT and processed_count >= TEST_LIMIT:
                print(f"Reached test limit of {TEST_LIMIT} bugs. Stopping.")
                break
                
            bug_id = row[bug_id_idx]
            if not bug_id.isdigit():
                continue

            print(f"Fetching comments for bug {bug_id}... ({processed_count + 1}/{TEST_LIMIT if TEST_LIMIT else 'all'})")
            comments = get_bug_comments(bug_id)

            for i, c in enumerate(comments):
                comments_writer.writerow([
                    bug_id,
                    i,
                    c.get("author", ""),
                    c.get("creation_time", ""),
                    c.get("text", "").replace("\n", " ").strip()
                ])

            processed_count += 1
            time.sleep(RATE_LIMIT)


if __name__ == "__main__":
    main()