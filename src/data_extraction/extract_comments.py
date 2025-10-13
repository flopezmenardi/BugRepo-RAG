import requests
import csv
import time

# Config
BASE_URL = "https://bugzilla.mozilla.org/rest"
INPUT_CSV = "data/sample_bugs.csv"   # existing metadata file
OUTPUT_COMMENTS = "data/bugs_comments.csv"
RATE_LIMIT = 0.1  # seconds between requests to be nice to API
TEST_LIMIT = 1000   # Only process first N bugs for testing (set to None for all)
START_OFFSET = 9000  # Skip the first N bugs (0 = start from beginning)
APPEND_MODE = True   # True = append to existing CSV, False = overwrite


def get_bug_comments(bug_id):
    url = f"{BASE_URL}/bug/{bug_id}/comment"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if "bugs" in data and str(bug_id) in data["bugs"]:
            return data["bugs"][str(bug_id)]["comments"]
    return []


def main():
    # Determine file mode and whether to write header
    file_mode = "a" if APPEND_MODE else "w"
    write_header = not APPEND_MODE
    
    # If appending, check if file exists and has content
    if APPEND_MODE:
        try:
            with open(OUTPUT_COMMENTS, 'r') as f:
                first_line = f.readline()
                if not first_line.strip():
                    write_header = True  # File is empty, write header
        except FileNotFoundError:
            write_header = True  # File doesn't exist, write header
    
    with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_COMMENTS, mode=file_mode, newline="", encoding="utf-8") as comments_file:

        reader = csv.reader(infile)
        comments_writer = csv.writer(comments_file)

        # Skip header row of input CSV
        header = next(reader)
        bug_id_idx = 0  # first column is bug_id

        # Write header for comments CSV only if needed
        if write_header:
            comments_writer.writerow(["bug_id", "comment_id", "creation_time", "text"])

        # Skip bugs until we reach our starting offset
        skipped_count = 0
        for row in reader:
            if skipped_count < START_OFFSET:
                skipped_count += 1
                continue
            else:
                # We've skipped START_OFFSET bugs, now process from here
                break
        
        print(f"Skipped first {START_OFFSET} bugs, starting from bug #{START_OFFSET + 1}")
        if APPEND_MODE:
            print(f"Appending to existing file: {OUTPUT_COMMENTS}")
        else:
            print(f"Creating new file: {OUTPUT_COMMENTS}")

        processed_count = 0
        
        # Continue processing from the current position (START_OFFSET + 1)
        for row in reader:
            # Limit for testing
            if TEST_LIMIT and processed_count >= TEST_LIMIT:
                print(f"Reached test limit of {TEST_LIMIT} bugs. Stopping.")
                break
                
            bug_id = row[bug_id_idx]
            if not bug_id.isdigit():
                continue

            print(f"Fetching comments for bug {bug_id}... processed ({processed_count + 1})")
            comments = get_bug_comments(bug_id)

            for i, c in enumerate(comments):
                comments_writer.writerow([
                    bug_id,
                    i,
                    c.get("creation_time", ""),
                    c.get("text", "").replace("\n", " ").strip()
                ])

            processed_count += 1
            time.sleep(RATE_LIMIT)


if __name__ == "__main__":
    main()