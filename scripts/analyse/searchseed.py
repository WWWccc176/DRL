import re
import csv
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

URL = "https://www.latticechallenge.org/svp-challenge/halloffame.php"


def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    # 这个页面可能有编码问题，所以不要直接用 r.text
    for enc in ["utf-8", "latin-1", "windows-1252"]:
        try:
            return r.content.decode(enc)
        except UnicodeDecodeError:
            pass
    return r.content.decode("utf-8", errors="replace")


def parse_rows(html: str):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # 优先按 HTML 表格解析
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        if cells[0].lower().startswith("position"):
            continue

        try:
            pos = int(cells[0])
            dim = int(cells[1])
            norm = float(cells[2])
            seed = int(cells[3])
        except Exception:
            continue

        rows.append(
            {
                "position": pos,
                "dimension": dim,
                "norm": norm,
                "seed": seed,
                "raw": cells,
            }
        )

    # 如果 HTML 表格解析失败，退化成纯文本正则解析
    if not rows:
        text = soup.get_text("\n")
        pattern = re.compile(
            r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(-?\d+)\s+(.+?)\s+vec\b", re.MULTILINE
        )
        for m in pattern.finditer(text):
            rows.append(
                {
                    "position": int(m.group(1)),
                    "dimension": int(m.group(2)),
                    "norm": float(m.group(3)),
                    "seed": int(m.group(4)),
                    "raw": m.group(0),
                }
            )

    return rows


def fibonacci_fill_order():
    """
    Generate seeds in the order:
    0, 1, 2, 3, 5, 8, 13, 21, 34, ...

    This starts from the user-specified sequence.
    """
    yield 0
    yield 1

    a, b = 1, 2
    while True:
        yield b
        a, b = b, a + b


def top_unique_seeds(rows, dim_min=50, dim_max=150, k=9):
    by_dim = defaultdict(list)

    for row in rows:
        d = row["dimension"]
        if dim_min <= d <= dim_max:
            by_dim[d].append(row)

    result = {}

    for d in range(dim_min, dim_max + 1):
        # 按网页原始 position 顺序，也就是“最靠前”
        candidates = sorted(by_dim.get(d, []), key=lambda x: x["position"])

        seen = set()
        chosen = []

        # 先选网页上最靠前的不同 seed
        for row in candidates:
            seed = row["seed"]
            if seed in seen:
                continue

            seen.add(seed)
            chosen.append(
                {
                    "seed": seed,
                    "source": "halloffame",
                    "position": row["position"],
                    "norm": row["norm"],
                }
            )

            if len(chosen) == k:
                break

        # 如果不足 9 个，按 0,1,2,3,5,8,13,21,... 补
        for fill_seed in fibonacci_fill_order():
            if len(chosen) == k:
                break

            if fill_seed in seen:
                continue

            seen.add(fill_seed)
            chosen.append(
                {
                    "seed": fill_seed,
                    "source": "filled",
                    "position": None,
                    "norm": None,
                }
            )

        result[d] = chosen

    return result


def main():
    html = fetch_html(URL)
    rows = parse_rows(html)
    result = top_unique_seeds(rows)

    with open(
        "svp_50_122_top9_unique_seeds.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["dimension", "top_unique_seeds", "count", "details_norm_seed_position"]
        )

        for d, chosen in result.items():
            seeds = [str(r["seed"]) for r in chosen]
            details = [
                f"norm={r['norm']},seed={r['seed']},pos={r['position']}" for r in chosen
            ]
            w.writerow([d, " ".join(seeds), len(chosen), " | ".join(details)])

    for d, chosen in result.items():
        seeds = [r["seed"] for r in chosen]
        print(f"{d}: {seeds}")


if __name__ == "__main__":
    main()
