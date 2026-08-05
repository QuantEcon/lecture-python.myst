#!/usr/bin/env python3
"""
Build the vendored dataset for the ``hansen_singleton_1983`` lecture.

Downloads the raw inputs from FRED and the Ken French data library, constructs
the monthly series used in the lecture (gross real market return, gross real
T-bill return, gross consumption growth, gross consumption inflation, and per
capita real consumption), and writes them to
``hansen_singleton_1983_data.csv`` next to this script.

Usage
-----
    python make_data.py

Requires only the standard library plus pandas (both ship with Anaconda).

Note
----
The committed CSV is a *frozen snapshot*. FRED revises historical series and
the Ken French library is updated over time, so re-running this script may
produce small differences from the committed data. See README.md.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FRED_CODES = {
    "population_16plus": "CNP16OV",
    "cons_nd_real_index": "DNDGRA3M086SBEA",
    "cons_nd_price_index": "DNDGRG3M086SBEA",
}
START = "1959-02-01"
END = "1978-12-01"
OUTPUT = Path(__file__).with_name("hansen_singleton_1983_data.csv")


def read_fred(codes, start, end):
    """Download FRED series as a date-indexed DataFrame (columns = codes)."""
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    columns = []
    for code in codes:
        url = f"{base}?id={code}&cosd={start:%Y-%m-%d}&coed={end:%Y-%m-%d}"
        columns.append(
            pd.read_csv(url, index_col=0, parse_dates=True, na_values="."))
    fred = pd.concat(columns, axis=1).astype("float64")
    fred.index.name = "DATE"
    return fred


def read_famafrench_factors(start, end):
    """Download the monthly Fama-French research factors (percent)."""
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_CSV.zip")
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8")

    # Preamble, then a monthly table (rows keyed by YYYYMM), then an annual
    # table (rows keyed by YYYY). Keep the contiguous monthly block.
    records = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        key = cells[0]
        if len(key) == 6 and key.isdigit():
            records.append([key] + [float(x) for x in cells[1:5]])
        elif records:
            break
    factors = pd.DataFrame(
        records, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    factors.index = pd.PeriodIndex(
        pd.to_datetime(factors["date"], format="%Y%m"), freq="M")
    factors = factors.drop(columns="date")
    window = ((factors.index >= pd.Period(start, "M"))
              & (factors.index <= pd.Period(end, "M")))
    return factors.loc[window]


def to_month_end(index):
    return pd.PeriodIndex(pd.DatetimeIndex(index), freq="M").to_timestamp("M")


def build(start=START, end=END):
    start_period = pd.Timestamp(start).to_period("M")
    end_period = pd.Timestamp(end).to_period("M")

    # Pull one extra month to build the first in-sample growth rate.
    fetch_start = (start_period - 1).to_timestamp(how="start")
    fetch_end = end_period.to_timestamp("M")
    sample_start = start_period.to_timestamp("M")
    sample_end = end_period.to_timestamp("M")

    fred = read_fred(list(FRED_CODES.values()), fetch_start, fetch_end)
    fred = fred.rename(columns={v: k for k, v in FRED_CODES.items()})
    fred.index = to_month_end(fred.index)
    fred["cons_real_level"] = fred["cons_nd_real_index"]
    fred["cons_price_index"] = fred["cons_nd_price_index"]
    fred["consumption_per_capita"] = (
        fred["cons_real_level"] / fred["population_16plus"])
    fred["gross_cons_growth"] = (
        fred["consumption_per_capita"]
        / fred["consumption_per_capita"].shift(1))
    fred["gross_inflation_cons"] = (
        fred["cons_price_index"] / fred["cons_price_index"].shift(1))

    ff = read_famafrench_factors(fetch_start, fetch_end).copy()
    ff.columns = [str(col).strip() for col in ff.columns]
    if ("Mkt-RF" not in ff.columns) or ("RF" not in ff.columns):
        raise KeyError(
            "Fama-French data missing required columns: 'Mkt-RF' and 'RF'.")
    # Mkt-RF and RF are reported in percent per month.
    ff["gross_nom_return"] = 1.0 + (ff["Mkt-RF"] + ff["RF"]) / 100.0
    ff["gross_nom_tbill"] = 1.0 + ff["RF"] / 100.0
    ff.index = ff.index.to_timestamp(how="end")
    ff.index = to_month_end(ff.index)
    market = ff[["gross_nom_return", "gross_nom_tbill"]]

    out = fred.join(market, how="inner")
    out["gross_real_return"] = (
        out["gross_nom_return"] / out["gross_inflation_cons"])
    out["gross_real_tbill"] = (
        out["gross_nom_tbill"] / out["gross_inflation_cons"])
    out = out.loc[sample_start:sample_end].dropna()

    frame = out[[
        "gross_real_return",
        "gross_cons_growth",
        "gross_inflation_cons",
        "consumption_per_capita",
        "gross_real_tbill",
    ]].copy()
    frame.index.name = "date"
    return frame


if __name__ == "__main__":
    frame = build()
    frame.to_csv(OUTPUT)
    print(f"wrote {OUTPUT.name}: {frame.shape[0]} rows x {frame.shape[1]} cols "
          f"({frame.index.min().date()} .. {frame.index.max().date()})")
