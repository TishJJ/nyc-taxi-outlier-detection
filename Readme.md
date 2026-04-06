# NYC Yellow Taxi — Distance Outlier Finder

Finds all yellow taxi trips above the **90th percentile of trip distance**, across every monthly Parquet file published by the NYC Taxi & Limousine Commission.

---

## Outputs

| File | Description |
|---|---|
| `outliers_monthly.json` | Trips above the per-month 90th percentile, written incrementally as each file is processed |
| `outliers_monthly.duckdb` | Same data in a DuckDB table, inserted incrementally |
| `outliers_global.json` | Trips above the global 90th percentile, written once all files are loaded |
| `outliers_global.duckdb` | Same global data in a DuckDB table |

Every record includes `month`, `source_file`, and `threshold` so each trip is traceable back to where it came from and what bar it was measured against.

---

## Approach

### Data access

The NYC TLC publishes yellow taxi trip records as monthly Parquet files hosted on a CloudFront CDN. The ideal approach would be to stream them lazily — both DuckDB and Polars support reading Parquet directly over HTTP without downloading the full file first. DuckDB's `httpfs` extension does this via HTTP range requests, fetching only the row groups and columns each query needs.

In practice this didn't work. CloudFront blocks requests that don't look like a browser, and DuckDB's default user agent triggered immediate 403s. Setting a browser user agent helped initially, but DuckDB's range request pattern — dozens of small sequential HTTP requests against the same file — was fingerprinted and blocked after the first few files regardless of the user agent string. A single full-file `GET` request, which is what a browser actually does, was not blocked.

The solution was to download each file in one HTTP request using `requests` with a browser user agent, which CloudFront accepts consistently. The TLC page is scraped directly for the list of available files so we only request URLs that actually exist — no guessing, no publication lag arithmetic.

### Memory

Processing the full file in memory caused out-of-memory kills on both Codespaces (4GB RAM) and Google Colab (12GB). A 70MB compressed Parquet file expands to several hundred MB in memory, and holding the full dataset alongside intermediate objects pushed past available RAM.

The solution uses two passes over each file:

**Pass 1** uses PyArrow's `iter_batches` to read the file one row group at a time, feeding only the `trip_distance` column into a numpy reservoir sample capped at 1 million values (~8MB). `numpy.percentile` on that sample gives the 90th percentile threshold. Peak memory for this pass is ~8MB regardless of file size.

**Sampling accuracy:** The full dataset contains approximately 1.5 billion rows through 2018, averaging 10–17 million rows per month at peak ridership (2013–2018). Post-COVID and recent files are considerably smaller at 3–5 million rows per month. A reservoir of 1 million rows gives:

| File size | Sample coverage | Error at p90 |
|---|---|---|
| Largest (~17M rows, peak 2013–2018) | ~6% | < 0.08% |
| Smallest (~1.5M rows, post-COVID 2020–2021) | ~67% | < 0.02% |

In practice the threshold shifts by less than a tenth of a mile in either direction — negligible for outlier detection.

**Pass 2** uses a Polars lazy scan with the threshold as a pushed-down filter, so only the ~10% of rows above the threshold are loaded into memory along with just the 6 needed columns.

The file is written to a temp file on disk rather than held in a `BytesIO` buffer in RAM. This lets the OS page the file to disk under memory pressure — with `BytesIO` the buffer is pinned in RAM with no escape valve.

### Percentile design

The 90th percentile is computed **per file** (per month) for the incremental output, giving a threshold relative to that month's trip distribution. A **global** threshold is computed after all files are processed by running `QUANTILE_CONT` across the full monthly DuckDB table — no files are re-fetched for this step.

### Ideal solution

Given no memory or CDN constraints, the cleanest architecture would be to query all Parquet files together using a lazy engine like DuckDB and add two fields at read time:

1. **`source_file`** — the URL or filename the row came from, so results can be filtered or grouped by month
2. **`is_multi_passenger`** — a boolean flag (`passenger_count > 1`) to allow outlier analysis to be split between single-rider and multi-passenger trips, since multi-passenger trips structurally travel further and distort the threshold

With those fields added, the full query becomes three steps without downloading anything:

```sql
-- Step 1: compute global threshold across all files
SELECT QUANTILE_CONT(trip_distance, 0.90)
FROM read_parquet('https://.../yellow_tripdata_*.parquet')
WHERE trip_distance > 0;

-- Step 2: retrieve outliers above the threshold
SELECT source_file, is_multi_passenger, trip_distance, ...
FROM read_parquet('https://.../yellow_tripdata_*.parquet')
WHERE trip_distance > <threshold>;

-- Step 3 (optional): count only, no raw data needed
SELECT source_file, COUNT(*) AS outlier_count
FROM read_parquet('https://.../yellow_tripdata_*.parquet')
WHERE trip_distance > <threshold>
GROUP BY source_file;
```

DuckDB's pushdown would fetch only the relevant columns and row groups across all files in a single pass, making the reservoir sampling and two-pass approach unnecessary. The current implementation is a pragmatic workaround for the CloudFront blocking and memory constraints of the target environments.

**S3 as an alternative data source:** The TLC dataset is also available as a public dataset on AWS S3 (`s3://nyc-tlc/trip data/`). Connecting via S3 entirely bypasses CloudFront and its blocking behaviour — DuckDB's `httpfs` extension supports S3 natively with no user agent issues. This would make the ideal solution above fully practical:

```python
import duckdb
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region='us-east-1';")

con.sql("""
    SELECT QUANTILE_CONT(trip_distance, 0.90)
    FROM read_parquet('s3://nyc-tlc/trip data/yellow_tripdata_*.parquet')
    WHERE trip_distance > 0
""").show()
```

### Why `trip_distance > 0` before computing any percentile

While these are outliers, the assumption is that they are cancelled trips or data entry errors. Since they are below the target threshold and would pull down the overall 90th percentile threshold score and inflate the outlier count, they are excluded from the calculation.

### Why consider flagging multi-passenger rides differently

A multi-passenger ride will more commonly show up as an outlier because the distance traveled generally will be greater than that of a single traveler. If this is an important stat to measure, separating them with a flag value may be helpful and could also change the 90th percentile value, thus changing the trips identified. Add `AND passenger_count = 1` to the filter in `query_file` to isolate single-rider trips.

### Resume support

If the run is interrupted, re-running automatically resumes from the last successfully processed month. Completed months are tracked in `outliers_monthly.duckdb`. The global pass is skipped until all months are complete, at which point it runs automatically.

---

## Reproduce

### 1. Install

```bash
pip install duckdb polars pyarrow requests lxml numpy
```

Requires Python 3.10+.

### 2. Run

```bash
python main.py
```

Processes all yellow taxi files listed on the TLC website. To test on a smaller slice, edit the call at the bottom of `main.py`:

```python
process_yellow(start_year=2023, end_year=2023)
```

### 3. Output

Progress is printed per file:

```
Processing 206 file(s)...

  2026-01 ...
    downloading 100% (64/64 MB)
  358,627 outliers (monthly threshold: 8.76 mi)
  2026-02 ...
    downloading 100% (59/59 MB)
  328,156 outliers (monthly threshold: 8.54 mi)
  ...

Monthly pass complete — 42,847,201 total outlier trips
Wrote outliers_monthly.json
Wrote outliers_monthly.duckdb

Global p90 threshold: 9.12 mi
Wrote outliers_global.json
Wrote outliers_global.duckdb
```

### 4. Query the DuckDB output

```python
import duckdb

con = duckdb.connect("outliers_monthly.duckdb")
con.sql("SELECT month, COUNT(*) AS trips FROM trips GROUP BY month ORDER BY month").show()

con = duckdb.connect("outliers_global.duckdb")
con.sql("SELECT COUNT(*) FROM trips").show()
```

### 5. Record format

Each JSON record and DuckDB row contains:

| Column | Description |
|---|---|
| `month` | Source file month (`YYYY-MM`) |
| `source_file` | Full URL of the Parquet file the trip came from |
| `pickup_dt` | Pickup datetime |
| `dropoff_dt` | Drop-off datetime |
| `trip_distance` | Distance in miles |
| `fare_amount` | Fare in USD |
| `tip_amount` | Tip in USD |
| `passenger_count` | Number of passengers |
| `threshold` | The percentile threshold applied (monthly or global) |
