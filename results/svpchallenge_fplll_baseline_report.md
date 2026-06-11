# SVP Challenge FPLLL Baseline Report

- Dataset files scanned: 657
- Successful result rows: 2628
- Failed rows: 0
- Dimension range: 50–122
- Distinct dim+seed pairs: 657
- CSV: `svpchallenge_fplll_baseline_norm_gh.csv`

## Method summary

| method | count | mean norm/GH | median | min | max |
|---|---:|---:|---:|---:|---:|
| lll | 657 | 2.6168972 | 2.4247178 | 1.0952848 | 5.595272 |
| bkz20 | 657 | 2.3120955 | 2.1632316 | 1.0952848 | 4.7919677 |
| bkz20_30 | 657 | 2.1220778 | 1.9928244 | 1.0952848 | 3.929498 |
| bkz20_30_40 | 657 | 1.979908 | 1.8522528 | 1.0394398 | 3.6807745 |

## Figures

- `svpchallenge_fplll_lll_norm_over_gh.png`
- `svpchallenge_fplll_bkz20_norm_over_gh.png`
- `svpchallenge_fplll_bkz20_30_norm_over_gh.png`
- `svpchallenge_fplll_bkz20_30_40_norm_over_gh.png`

## Backend note

This script uses the same Python-facing Rust/C++ backend style as agent7. Because the current backend exposes `LOCAL_BKZ` rather than a separate global BKZ call, `bkz20`, `bkz20_30`, and `bkz20_30_40` are implemented as sequential full left-to-right LOCAL_BKZ sweeps, with LLL after each beta sweep.
