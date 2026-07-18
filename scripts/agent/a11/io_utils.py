from __future__ import annotations

import glob
import os
import re


def matrix_to_string(basis):
    lines = [" ".join(str(x) for x in row) for row in basis]
    return "[" + "\n".join(f"[{line}]" for line in lines) + "]"


def parse_fplll(s: str):
    out = []
    for line in s.strip().splitlines():
        line = line.strip().lstrip("[").rstrip("]").strip()
        if line:
            out.append([int(x) for x in line.split()])
    return out


def parse_challenge_file(filepath: str):
    matrix = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().replace("[", "").replace("]", "")
        for line in content.strip().split("\n"):
            if line.strip():
                matrix.append([int(x) for x in line.split()])
    return matrix


def parse_dim_seed(path: str):
    basename = os.path.basename(path)
    dim = re.search(r"dim(\d+)", basename)
    seed = re.search(r"seed(\d+)", basename)
    return (int(dim.group(1)) if dim else 0, int(seed.group(1)) if seed else 0)


def gather_files(dataset_dir: str):
    """Load every valid dim/seed .txt file in dataset/."""
    files = []
    for filepath in glob.glob(os.path.join(dataset_dir, "*.txt")):
        dim, _seed = parse_dim_seed(filepath)
        if dim > 0 and re.search(r"seed\d+", os.path.basename(filepath)):
            files.append(filepath)
    return sorted(files, key=lambda p: (*parse_dim_seed(p), os.path.basename(p)))
