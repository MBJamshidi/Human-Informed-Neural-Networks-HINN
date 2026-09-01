# HINN Real-World Validation - AAPL (FNSPID)

- Substrate: real AAPL OHLCV, 10852 days (1980-12-12 to 2023-12-28)
- Human pathway: paper-exact (10% uniform noise + linear policy drift), mask coverage 46.6%
- Methodology: imported unmodified from src/hinn/simulation.py
- Runtime: 47.8s

## Comparative baseline

|                       |    Acc |   macro_F1 |    ECE |   Brier |   AUC_ovo |   lambda_bar |   post_var |      KL |   loss |
|:----------------------|-------:|-----------:|-------:|--------:|----------:|-------------:|-----------:|--------:|-------:|
| HINN (full)           | 0.2236 |     0.2248 | 0.0152 |  0.323  |    0.5437 |       0.6228 |     0.0002 | 149.166 | 1.2292 |
| Data-only (lambda=1)  | 0.2128 |     0.2105 | 0.0109 |  0.3181 |    0.5621 |       1      |     0      | 149.155 | 1.2005 |
| Human-only (lambda=0) | 0.3007 |     0.276  | 0.1371 |  0.3512 |    0.4808 |       0.0015 |     0      | 149.188 | 1.2108 |