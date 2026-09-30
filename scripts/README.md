# Data download scripts

Run all commands from the **repo root** with the project venv active:

```bash
cd /path/to/Factor-Model
source .venv/bin/activate
```

Target layout:

```
data/raw/bhavcopy/       # NSE EOD zips
data/raw/universe/       # Nifty 500 snapshots
data/raw/risk_free/      # RBI / MIBOR CSV (normalized)
data/raw/fundamentals/   # ME / BTM panel CSV
data/processed/...       # Parquet / DuckDB outputs
```

---

## Orchestrator (recommended)

Downloads Bhavcopy + universe + fundamentals + adjustments for a date window.  
Risk-free is **not** fully auto-fetched (see [Risk-free gap](#risk-free-gap-rbi-91-day-t-bill) below).

### For example: Jan 2020 → Dec 2025

```bash
python scripts/download_all_inputs.py \
  --start 2020-01-01 \
  --end 2025-12-31 \
  --universe-only-adjustments \
  --fundamentals-mode snapshot
```

If Bhavcopy for this range is **already** downloaded (you already ran `download_bhavcopy.py`):

```bash
python scripts/download_all_inputs.py \
  --start 2020-01-01 \
  --end 2025-12-31 \
  --skip-bhavcopy \
  --universe-only-adjustments \
  --fundamentals-mode snapshot
```

Optional smoke limits (faster, fewer names):

```bash
python scripts/download_all_inputs.py \
  --start 2020-01-01 \
  --end 2025-12-31 \
  --skip-bhavcopy \
  --universe-only-adjustments \
  --fundamentals-limit 100 \
  --adjustments-limit 100
```

After you have a normalized risk-free CSV in `data/raw/risk_free/` (already present — no import required):

```bash
python scripts/download_all_inputs.py \
  --start 2020-01-01 \
  --end 2025-12-31 \
  --skip-bhavcopy \
  --universe-only-adjustments \
  --fundamentals-mode snapshot
```

Notes:
- Default fundamentals mode is **snapshot** (more reliable under Yahoo SSL/crumb issues).
- `quarterly` auto-falls back to snapshot if empty; fundamentals/adjustments failures no longer abort the orchestrator.
- Existing non-template CSVs in `data/raw/risk_free/` are detected; template is not overwritten.

---



## Individual scripts



### 1. `download_bhavcopy.py` — NSE EOD Bhavcopy

Pulls old `cm*bhav.csv.zip` and/or UDiFF `BhavCopy_NSE_CM_*` into `data/raw/bhavcopy/`.  
Weekends skipped; holidays 404 and are skipped.

```bash
python scripts/download_bhavcopy.py --start 2020-01-01 --end 2025-12-31
python scripts/download_bhavcopy.py -s 2020-01-01 -e 2025-12-31 --force   # re-download
python scripts/download_bhavcopy.py -s 2020-01-01 -e 2025-12-31 -v        # verbose
```



### 2. `download_universe.py` — Nifty 500 constituents

Fetches the **current** public constituent list (not true historical PIT).

```bash
python scripts/download_universe.py
python scripts/download_universe.py --as-of 2025-12-31
```

Outputs: `data/raw/universe/nifty500_YYYY-MM-DD.csv` and processed Parquet.

For point-in-time membership, re-run periodically and keep dated files, or supply archived historical lists.

### 3. `download_risk_free.py` — India $R_f$ (RBI 91d / MIBOR)

```bash
# Write a tiny CSV template to fill in
python scripts/download_risk_free.py --template

# Import a local normalized CSV (date + yield)
python scripts/download_risk_free.py --from-csv data/raw/risk_free/rbi_91d_tbill.csv

# Import from a direct CSV URL (if you host one)
python scripts/download_risk_free.py --from-url https://example.com/rbi_91d.csv
```

Expected columns (aliases accepted): `date` + `yield` / `annualized_yield` / `ytm` / `rate`.  
Percent (`6.75`) or decimal (`0.0675`) both work.

### 4a. `smoke_test_vendor_fundamentals.py` — EODHD / FMP probe

Smoke-tests paid/free vendor APIs into the same schema (`as_of_date`, `symbol`,
`market_cap`, `book_value`, `book_to_market`). Keys via env (never committed):

```bash
# EODHD public demo (AAPL.US) works with no signup
python scripts/smoke_test_vendor_fundamentals.py

# After signup:
export EODHD_API_TOKEN=...   # https://eodhd.com/register
export FMP_API_KEY=...       # https://site.financialmodelingprep.com/register
python scripts/smoke_test_vendor_fundamentals.py --symbols RELIANCE,TCS,INFY
```

Library: `src/data/fundamentals_vendors.py`.

### 4. `download_fundamentals.py` — Market cap & book-to-market

Uses **yfinance** (good for pipeline validation; not CMIE-grade PIT).

```bash
# Quarterly filings-linked panel for Nifty 500 (or local universe file)
python scripts/download_fundamentals.py --mode quarterly

# Faster current snapshot
python scripts/download_fundamentals.py --mode snapshot --limit 50

# Explicit tickers
python scripts/download_fundamentals.py --mode quarterly --symbols RELIANCE,TCS,INFY

# Book equity only from Screener.in (see 4b)
python scripts/download_fundamentals.py --mode screener-book --symbols RELIANCE,TCS
```

### 4b. `download_screener_book.py` — Screener.in book equity

Scrapes consolidated balance-sheet **Equity Capital + Reserves** (₹ crore → INR)
for 2020–2025 by default. Leaves `market_cap` / `book_to_market` null. HTML is
cached under `data/raw/fundamentals/screener_cache/`.

```bash
# Smoke: a few names
python scripts/download_screener_book.py --symbols RELIANCE,TCS,INFY -v

# Full universe (Nifty 500 snapshot), polite rate limit
python scripts/download_screener_book.py --start-year 2020 --end-year 2025 --pause 0.75

# Overlay book onto an existing ME panel and rewrite fundamentals
python scripts/download_screener_book.py \
  --merge-into data/processed/fundamentals/fundamentals.parquet \
  --merged-name fundamentals
```

Outputs: `data/processed/fundamentals/screener_book.parquet` (+ raw CSV).

### 4c. `download_me.py` — Market equity (Bhavcopy × Yahoo shares)

\[
\mathrm{ME}_{i,t} = \underbrace{\text{Bhavcopy close}_{i,t}}_{\text{unadjusted}}
\times \underbrace{\text{shares outstanding}_{i,t}}_{\text{Yahoo get\_shares\_full}}
\]

Attaches ME onto Screener book `as_of_date` rows and rewrites
`fundamentals.parquet` with `book_to_market` filled. Shares are cached at
`data/raw/fundamentals/shares_outstanding.parquet`.

```bash
# Smoke
python scripts/download_me.py --symbols RELIANCE,TCS,INFY -v

# Full universe (symbols taken from screener_book)
python scripts/download_me.py --pause 0.15

# Reuse cached shares
python scripts/download_me.py \
  --shares-from data/raw/fundamentals/shares_outstanding.parquet
```

Outputs:
- `data/processed/fundamentals/market_equity.parquet` (audit: close, shares, ME)
- `data/processed/fundamentals/fundamentals.parquet` (book + ME + BTM)

### 5. `download_adjustments.py` — Split / bonus factors

Joins yfinance `Adj Close / Close` onto Bhavcopy; writes year-partitioned Parquet under `data/processed/bhavcopy/`.

```bash
# Prefer universe-only (much faster than full CM)
python scripts/download_adjustments.py \
  --universe-only \
  --start 2020-01-01 \
  --end 2025-12-31

python scripts/download_adjustments.py --symbols RELIANCE,TCS --limit-symbols 50
python scripts/download_adjustments.py --from-processed   # use existing processed panel
```



### 6. `_util.py`

Internal helpers only — not run directly.

---

## Phase-2 pipeline

### `run_phase2.py` — Clean prices + returns

Loads the **processed adjusted** Bhavcopy panel, builds clean prices and
excess returns (vs RBI Rf), and as-of joins fundamentals when available.

```bash
python scripts/run_phase2.py
python scripts/run_phase2.py -v
python scripts/run_phase2.py --from-raw   # parse raw zips instead (slower)
```

Writes under `data/processed/returns/` (and refreshes processed bhavcopy).

**Note:** a Yahoo *snapshot* fundamentals file with `available_date` in the
future will not join onto 2020–2025 returns — Amihud/WML still work; SMB/HML
need a PIT fundamentals CSV.

### `run_phase3.py` — Factor returns (WML, ILLIQ, …)

Builds characteristic panel + daily long-short factors from processed returns.
SMB/HML are built only when non-null `market_cap` / `book_to_market` exist.

```bash
python scripts/run_phase3.py
python scripts/run_phase3.py -v --amihud-window 30
```

Writes:
- `data/processed/factors/characteristics/year=*/part.parquet`
- `data/processed/factors/factor_returns/year=*/part.parquet`

### `run_phase4.py` — Fama–MacBeth + diagnostics

Runs characteristic Fama–MacBeth (Newey–West SEs), factor time-series metrics,
and optional month-end quintile long–short backtests for momentum / Amihud.

```bash
python scripts/run_phase4.py
python scripts/run_phase4.py --skip-backtest
python scripts/run_phase4.py --chars mom_12_1_z,illiq_signal_z -v
```

Writes under `data/processed/reports/` (`fama_macbeth_*.md`, `factor_metrics.md`,
`bt_momentum_*`, `bt_illiq_*`).

### `run_phase5.py` — Research tearsheet

Builds equal-weight **MKT**, factor correlations, optional classical two-pass
Fama–MacBeth, wealth charts, and a consolidated `MASTER_REPORT.md`.

```bash
python scripts/run_phase5.py
python scripts/run_phase5.py --skip-two-pass   # faster
python scripts/run_phase5.py --no-plots
```

Primary output: `data/processed/reports/MASTER_REPORT.md`.

---

## Risk-free gap (RBI 91-Day T-Bill)

There is **no stable public bulk API** for historical RBI 91-Day T-Bill yields that this repo can call reliably end-to-end. `download_risk_free.py` therefore:

1. Writes a **template**, or
2. **Imports** a CSV you supply (`--from-csv` / `--from-url`).



### Practical options


| Source                                                               | Coverage                             | Notes                                                                               |
| -------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| **RBI DBIE** ([data.rbi.org.in/DBIE](https://data.rbi.org.in/DBIE/)) | Full history you need                | Preferred. Export auction / yield series → normalize to `date,yield` → `--from-csv` |
| **Kaggle** (India 91-Day T-Bill datasets)                            | Typically **only through ~May 2023** | Fine for 2020–early-2023, **not enough alone** for a 2020–2025 study window         |




### If you already have an RBI *auction* export

Normalize RBI data to:

```csv
date,yield
2020-01-03,5.20
2020-01-10,5.25
```

Use auction / issue **date** and the **implicit yield at cut-off (%)**. Save under `data/raw/risk_free/rbi_91d_tbill.csv`, then:

```bash
python scripts/download_risk_free.py --from-csv data/raw/risk_free/rbi_91d_tbill.csv
```

---

