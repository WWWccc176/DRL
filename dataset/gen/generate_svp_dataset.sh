#!/usr/bin/env zsh

# Generate SVP Challenge lattice instances from the seed table.
# Output is a flat file list under ./dataset only: no per-dimension folders, no logs.
#
# Usage:
#   ./generate_svp_dataset.sh
#   ./generate_svp_dataset.sh path/to/seeds.csv
#   ./generate_svp_dataset.sh path/to/seeds.csv path/to/generate_random
#   ./generate_svp_dataset.sh path/to/seeds.csv path/to/generate_random path/to/dataset

emulate -L zsh
set -e
set -u
set -o pipefail

SCRIPT_DIR="${0:A:h}"
CSV_FILE="${1:-${SCRIPT_DIR}/svp_50_122_top9_unique_seeds.csv}"
GENERATOR="${2:-${SCRIPT_DIR}/generate_random}"
OUT_DIR="${3:-${SCRIPT_DIR:h}}"
if [[ ! -f "$CSV_FILE" ]]; then
    print -u2 "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

if [[ ! -x "$GENERATOR" ]]; then
    print -u2 "ERROR: generator is not executable: $GENERATOR"
    print -u2 "Build it first, for example: make generate_random"
    exit 1
fi

mkdir -p "$OUT_DIR"

awk -F',' '
NR == 1 { next }
{
    dim = $1
    seeds_field = $2
    gsub(/\r/, "", dim)
    gsub(/\r/, "", seeds_field)
    gsub(/^[ \t]+|[ \t]+$/, "", dim)
    gsub(/^"|"$/, "", seeds_field)

    n = split(seeds_field, seeds, /[[:space:]]+/)
    for (i = 1; i <= n; i++) {
        seed = seeds[i]
        gsub(/^[ \t]+|[ \t]+$/, "", seed)
        if (dim != "" && seed != "") {
            key = dim "_" seed
            if (!(key in seen)) {
                seen[key] = 1
                print dim, seed
            }
        }
    }
}
' "$CSV_FILE" | while read -r dim seed; do
    "$GENERATOR" --dim "$dim" --seed "$seed" > "${OUT_DIR}/svpchallengedim${dim}seed${seed}.txt"
done
