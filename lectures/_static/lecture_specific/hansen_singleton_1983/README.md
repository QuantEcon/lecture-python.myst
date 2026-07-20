# Data — `hansen_singleton_1983`

Vendored input data for the **Hansen–Singleton (1983)** lecture
(`lectures/hansen_singleton_1983.md`).

The lecture reads `hansen_singleton_1983_data.csv` directly from GitHub instead
of querying the data providers while the book builds. This keeps the build
reproducible and removes its dependence on live FRED / Ken French endpoints
(which can be slow, rate-limited, or temporarily unavailable).

The companion lecture `hansen_singleton_1982` uses the same construction; this
dataset is a superset that additionally carries the T-bill return and the
intermediate consumption/inflation series the 1983 lecture reports.

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

Monthly, **1959-02 to 1978-12** (239 observations). One extra prior month is
pulled to form the first in-sample growth rate.

## File: `hansen_singleton_1983_data.csv`

Month-end `date` index plus:

| Column | Definition |
| --- | --- |
| `gross_real_return` | gross real market return, `(1 + (Mkt-RF + RF)/100) / gross_inflation_cons` |
| `gross_cons_growth` | gross growth of per-capita real nondurable consumption |
| `gross_inflation_cons` | gross consumption-deflator inflation |
| `consumption_per_capita` | per-capita real nondurable consumption level |
| `gross_real_tbill` | gross real T-bill return, `(1 + RF/100) / gross_inflation_cons` |

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
