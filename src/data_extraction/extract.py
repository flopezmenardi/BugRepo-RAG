import requests
import csv
import time

# Definimos un rango de id's de bugs para extraer:
# se extraen los metadatos y comentarios de cada bug, y se guardan 

# Configuración
BASE_URL = "https://bugzilla.mozilla.org/rest"
OUTPUT_BUGS = "bugs_metadata.csv"
OUTPUT_COMMENTS = "bugs_comments.csv"
START_BUG_ID = 1801500
END_BUG_ID = 1801510   
RATE_LIMIT = 1  # segundos entre requests para no saturar la API


def get_bug_details(bug_id):
    url = f"{BASE_URL}/bug/{bug_id}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if "bugs" in data and data["bugs"]:
            return data["bugs"][0]
    return None


def get_bug_comments(bug_id):
    url = f"{BASE_URL}/bug/{bug_id}/comment"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if "bugs" in data and str(bug_id) in data["bugs"]:
            return data["bugs"][str(bug_id)]["comments"]
    return []


def main():
    # Archivos CSV
    with open(OUTPUT_BUGS, mode="w", newline="", encoding="utf-8") as bugs_file, \
         open(OUTPUT_COMMENTS, mode="w", newline="", encoding="utf-8") as comments_file:

        bugs_writer = csv.writer(bugs_file)
        comments_writer = csv.writer(comments_file)

        # Cabeceras
        bugs_writer.writerow(["bug_id", "summary", "status", "resolution", "severity", "priority", "product", "component", "creation_time", "last_change_time", "description"])
        comments_writer.writerow(["bug_id", "comment_id", "author", "creation_time", "text"])

        for bug_id in range(START_BUG_ID, END_BUG_ID + 1):
            print(f"Fetching bug {bug_id}...")

            # Bug metadata
            bug = get_bug_details(bug_id)
            if bug:
                bugs_writer.writerow([
                    bug.get("id"),
                    bug.get("summary", ""),
                    bug.get("status", ""),
                    bug.get("resolution", ""),
                    bug.get("severity", ""),
                    bug.get("priority", ""),
                    bug.get("product", ""),
                    bug.get("component", ""),
                    bug.get("creation_time", ""),
                    bug.get("last_change_time", ""),
                    bug.get("description", "")
                ])

                # Bug comments
                comments = get_bug_comments(bug_id)
                for i, c in enumerate(comments):
                    comments_writer.writerow([
                        bug_id,
                        i,
                        c.get("author", ""),
                        c.get("creation_time", ""),
                        c.get("text", "").replace("\n", " ").strip()
                    ])

            time.sleep(RATE_LIMIT)


if __name__ == "__main__":
    main()