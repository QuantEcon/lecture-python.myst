# Data — `hansen_singleton_1982`

Vendored input data for the **Hansen–Singleton (1982)** lecture
(`lectures/hansen_singleton_1982.md`).

The lecture reads `hansen_singleton_1982_data.csv` directly from GitHub instead
of querying the data providers while the book builds. This keeps the build
reproducible and removes its dependence on live FRED / Ken French endpoints
(which can be slow, rate-limited, or temporarily unavailable).

## Sources

| Quantity | Provider | Code / dataset |
| --- | --- | --- |
| Civilian noninstitutional population, 16+ | FRED | `CNP16OV` |
| Real PCE: nondurable goods (chain-type quantity index) | FRED | `DNDGRA3M086SBEA` |
| PCE: nondurable goods (chain-type price index) | FRED | `DNDGRG3M086SBEA` |
| Monthly market excess return `Mkt-RF` and risk-free rate `RF` | Ken French data library | `F-F_Research_Data_Factors` |

- FRED — <https://fred.stlouisfed.org/>
- Ken French data library — <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>

## Sample

Monthly, **1959-02 to 1978-12** (239 observations), matching the Hansen–Singleton
(1982) ND + VWR sample. One extra prior month is pulled to form the first
in-sample growth rate.

## File: `hansen_singleton_1982_data.csv`

Month-end `date` index plus:

| Column | Definition |
| --- | --- |
| `gross_real_return` | gross real market return, `(1 + (Mkt-RF + RF)/100) / gross_inflation` |
| `gross_cons_growth` | gross growth of per-capita real nondurable consumption |

## Regenerating

```bash
python make_data.py
```

`make_data.py` needs only the standard library and pandas (both ship with
Anaconda).

> **The committed CSV is a frozen snapshot.** FRED revises historical series and
> the Ken French library is updated over time, so regenerating may produce small
> differences from the committed values. Update the CSV deliberately, in its own
> commit, when you intend to refresh the data.
