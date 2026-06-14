from bs4 import BeautifulSoup
import csv
import re
from pathlib import Path
from collections import defaultdict

INPUT_HTML = "halloffame.html"
OUT_ALL = "svp_50_122_all_records.csv"
OUT_SUMMARY = "svp_50_122_by_dimension_summary.csv"


def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", x.replace("\xa0", " ")).strip()


def parse_halloffame(html_path: str):
    html = Path(html_path).read_text(encoding="iso-8859-1", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    for tr in soup.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue

        position = int(clean_text(tds[0].get_text()))
        dimension = int(clean_text(tds[1].get_text()))
        norm = clean_text(tds[2].get_text())
        seed = clean_text(tds[3].get_text())
        contestant = clean_text(tds[4].get_text())
        algorithm = clean_text(tds[6].get_text())
        subm_date = clean_text(tds[7].get_text())
        approx_factor = clean_text(tds[8].get_text())
        title_note = clean_text(tr.get("title", ""))

        if 50 <= dimension <= 122:
            rows.append(
                {
                    "position": position,
                    "dimension": dimension,
                    "norm": norm,
                    "seed": seed,
                    "contestant": contestant,
                    "algorithm": algorithm,
                    "subm_date": subm_date,
                    "approx_factor": approx_factor,
                    "title_note": title_note,
                }
            )

    return rows


def write_all_records(rows):
    fieldnames = [
        "position",
        "dimension",
        "norm",
        "seed",
        "contestant",
        "algorithm",
        "subm_date",
        "approx_factor",
        "title_note",
    ]

    with open(OUT_ALL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary(rows):
    by_dim = defaultdict(list)
    for r in rows:
        by_dim[r["dimension"]].append(r)

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dimension",
                "num_records",
                "seeds",
                "contestant_algorithm_pairs",
                "top_record",
            ]
        )

        for d in sorted(by_dim.keys(), reverse=True):
            records = sorted(by_dim[d], key=lambda x: x["position"])
            seeds = sorted(
                set(r["seed"] for r in records),
                key=lambda x: int(x) if x.isdigit() else x,
            )

            pairs = []
            seen = set()
            for r in records:
                pair = f"{r['contestant']} :: {r['algorithm']}"
                if pair not in seen:
                    pairs.append(pair)
                    seen.add(pair)

            top = records[0]
            top_record = (
                f"pos={top['position']}, seed={top['seed']}, "
                f"norm={top['norm']}, author={top['contestant']}, "
                f"algorithm={top['algorithm']}"
            )

            w.writerow(
                [d, len(records), " ".join(seeds), " | ".join(pairs), top_record]
            )


def main():
    rows = parse_halloffame(INPUT_HTML)

    print(f"Extracted {len(rows)} records from dimension 50 to 122.")
    print(f"Highest dimension: {max(r['dimension'] for r in rows)}")
    print(f"Lowest dimension: {min(r['dimension'] for r in rows)}")

    write_all_records(rows)
    write_summary(rows)

    print(f"Saved full table to: {OUT_ALL}")
    print(f"Saved dimension summary to: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
