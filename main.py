"""
NYC Yellow Taxi — Distance Outlier Finder
==========================================
Finds yellow taxi trips above the 90th percentile of trip_distance.

Four outputs:
  outliers_monthly.json    — trips above the per-month 90th percentile,
                             written incrementally as each file is processed.
  outliers_monthly.duckdb  — same data, inserted into a DuckDB table as each
                             file is processed.
  outliers_global.json     — trips above the global 90th percentile, written
                             once all files have been loaded.
  outliers_global.duckdb   — same global data in a DuckDB table.

Why two thresholds:
  The per-month threshold surfaces trips that were unusual relative to that
  month. The global threshold surfaces trips that were unusual across all of
  history — a single bar for the entire dataset.

Why trip_distance > 0 before computing any percentile:
  While these are outliers, the assumption is that they are cancelled trips
  or data entry errors. Since they are below the target threshold and would
  pull down the overall 90th percentile threshold score and inflate the
  outlier count, they are excluded from the calculation.

Why consider flagging multi-passenger rides differently:
  A multi-passenger ride will more commonly show up as an outlier because
  the distance traveled generally will be greater than that of a single
  traveler. If this is an important stat to measure, separating them with
  a flag value may be helpful and could also change the 90th percentile
  value, thus changing the trips identified.

Why scrape the TLC page instead of generating URLs:
  The TLC publishes files with an irregular lag and occasionally has gaps.
  Scraping the page means we only request files that actually exist.
"""

import gc
import os
import tempfile
import json
import numpy as np
import random
import time
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq
import requests
from lxml import html


# ── Constants ─────────────────────────────────────────────────────────────────

TLC_PAGE_URL    = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
PARQUET_XPATH   = '//a[contains(@href, "yellow_tripdata") and contains(@href, ".parquet")]'
FILENAME_SUFFIX = ".parquet"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

MONTHLY_JSON   = Path("outliers_monthly.json")
MONTHLY_DUCKDB = Path("outliers_monthly.duckdb")
GLOBAL_JSON    = Path("outliers_global.json")
GLOBAL_DUCKDB  = Path("outliers_global.duckdb")


# ── URL scraping ──────────────────────────────────────────────────────────────

def get_yellow_urls() -> list[str]:
    """Scrape the TLC page and return all yellow taxi Parquet URLs."""
    try:
        response = requests.get(TLC_PAGE_URL)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        links = tree.xpath(PARQUET_XPATH)
        return [link.get("href").strip() for link in links]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching TLC page: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


# ── DuckDB output store ───────────────────────────────────────────────────────

def init_duckdb(path: Path, resume: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Open or create a DuckDB output file with the trips table.
    resume=True: open existing file and append.
    resume=False: delete and recreate from scratch.
    """
    if not resume:
        wal_path = Path(str(path) + ".wal")
        try:
            if path.exists():
                path.unlink()
            if wal_path.exists():
                wal_path.unlink()
        except OSError as e:
            print(f"Warning: Could not delete '{path}': {e}\nConsider restarting the runtime.")

    con = duckdb.connect(str(path))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS trips (
                month           VARCHAR,
                source_file     VARCHAR,
                pickup_dt       TIMESTAMP,
                dropoff_dt      TIMESTAMP,
                trip_distance   DOUBLE,
                fare_amount     DOUBLE,
                tip_amount      DOUBLE,
                passenger_count INTEGER,
                threshold       DOUBLE
            )
        """)
    except Exception as e:
        con.close()
        raise RuntimeError(f"Error initialising DuckDB at '{path}': {e}")
    return con


def completed_months(path: Path) -> set[str]:
    """Return months already written to a DuckDB output file, or empty set."""
    if not path.exists():
        return set()
    try:
        con = duckdb.connect(str(path), read_only=True)
        months = {row[0] for row in con.execute("SELECT DISTINCT month FROM trips").fetchall()}
        con.close()
        return months
    except Exception:
        return set()


# ── Core: stream + reservoir + Polars ────────────────────────────────────────

def query_file(
    url:        str,
    percentile: float = 0.90,
) -> tuple[pl.DataFrame, float | None]:
    """
    Pass 1 — Download into BytesIO. Use pyarrow iter_batches to read only
             trip_distance one row group at a time into a numpy reservoir
             (capped at 1M values / 8MB). Compute threshold with np.percentile.
             Explicitly free the reservoir and ParquetFile before pass 2.

    Pass 2 — Seek BytesIO back to 0. Polars lazy scan with filter and column
             selection pushed down — only outlier rows and 6 columns loaded.
             Explicitly free the buffer after collect.

    Returns (empty DataFrame, None) if no usable distance data.
    """
    # Download to temp file — avoids holding buffer in RAM during both passes
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1_048_576):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    downloading {pct:.0f}% ({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB)",
                              end="", flush=True)
                print()

        # Pass 1: reservoir sampling from disk — peak mem ~8MB
        pf             = pq.ParquetFile(tmp_path)
        rng            = np.random.default_rng(42)
        reservoir      = []
        reservoir_size = 1_000_000
        seen           = 0

        for batch in pf.iter_batches(columns=["trip_distance"]):
            arr   = batch.column("trip_distance").to_numpy()
            valid = arr[np.isfinite(arr) & (arr > 0)]
            for v in valid:
                seen += 1
                if len(reservoir) < reservoir_size:
                    reservoir.append(v)
                else:
                    j = int(rng.integers(0, seen))
                    if j < reservoir_size:
                        reservoir[j] = v

        if not reservoir:
            return pl.DataFrame(), None

        threshold = float(np.percentile(reservoir, percentile * 100))

        # Free pass 1 memory before pass 2
        del pf, reservoir, arr, valid
        gc.collect()

        # Pass 2: Polars lazy scan from disk — OS can page if needed
        df = (
            pl.scan_parquet(tmp_path)
            .filter(pl.col("trip_distance") > threshold)
            .select([
                pl.col("tpep_pickup_datetime").alias("pickup_dt"),
                pl.col("tpep_dropoff_datetime").alias("dropoff_dt"),
                pl.col("trip_distance"),
                pl.col("fare_amount"),
                pl.col("tip_amount"),
                pl.col("passenger_count"),
            ])
            .sort("trip_distance", descending=True)
            .collect(engine="streaming")
        )

    finally:
        os.unlink(tmp_path)
        gc.collect()

    if df.is_empty():
        return pl.DataFrame(), None

    return df, threshold


# ── Writing outputs ───────────────────────────────────────────────────────────

def write_monthly(
    df:         pl.DataFrame,
    threshold:  float,
    month:      str,
    url:        str,
    json_path:  Path,
    duckdb_con: duckdb.DuckDBPyConnection,
) -> None:
    """Append one month of outliers to the JSON file and DuckDB table."""
    df = df.with_columns([
        pl.lit(month).alias("month"),
        pl.lit(url).alias("source_file"),
        pl.lit(threshold).alias("threshold"),
    ])

    try:
        existing = json.loads(json_path.read_text()) if json_path.exists() else []
        if not isinstance(existing, list):
            existing = []
    except (json.JSONDecodeError, ValueError):
        existing = []
    existing.extend(df.to_dicts())
    json_path.write_text(json.dumps(existing, indent=2, default=str))

    duckdb_con.execute("""
        INSERT INTO trips
        SELECT month, source_file, pickup_dt, dropoff_dt,
               trip_distance, fare_amount, tip_amount,
               passenger_count, threshold
        FROM df
    """)


def write_global(monthly_con: duckdb.DuckDBPyConnection, percentile: float = 0.90) -> None:
    """Compute the global percentile across all trips and write global outputs."""
    threshold = monthly_con.execute(f"""
        SELECT QUANTILE_CONT(trip_distance, {percentile}) FROM trips
    """).fetchone()[0]

    print(f"\nGlobal p{int(percentile * 100)} threshold: {threshold:.4f} mi")

    global_df = monthly_con.execute(f"""
        SELECT * FROM trips
        WHERE trip_distance > {threshold}
        ORDER BY trip_distance DESC
    """).pl()

    records = global_df.with_columns(pl.lit(threshold).alias("threshold")).to_dicts()
    GLOBAL_JSON.write_text(json.dumps(records, indent=2, default=str))
    print(f"Wrote {GLOBAL_JSON} ({len(records):,} records)")

    global_con = init_duckdb(GLOBAL_DUCKDB)
    try:
        global_con.execute("INSERT INTO trips SELECT * FROM global_df")
        print(f"Wrote {GLOBAL_DUCKDB}")
    except Exception as e:
        print(f"Error writing {GLOBAL_DUCKDB}: {e}")
    finally:
        global_con.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def process_yellow(
    start_year: int | None = None,
    end_year:   int | None = None,
    percentile: float = 0.90,
) -> None:
    done     = completed_months(MONTHLY_DUCKDB)
    resuming = len(done) > 0
    if resuming:
        print(f"Resuming — {len(done)} month(s) already done, skipping them.")
    else:
        MONTHLY_JSON.unlink(missing_ok=True)
    monthly_con = init_duckdb(MONTHLY_DUCKDB, resume=resuming)

    try:
        urls    = get_yellow_urls()
        pending = [u for u in urls
                   if u.split("yellow_tripdata_")[1].replace(FILENAME_SUFFIX, "") not in done]
        print(f"Processing {len(pending)} of {len(urls)} file(s)...")
        time.sleep(random.uniform(1.0, 3.0))

        for url in pending:
            try:
                month = url.split("yellow_tripdata_")[1].replace(FILENAME_SUFFIX, "")
                print(f"  {month} ... ", end="", flush=True)

                df, threshold = query_file(url, percentile)

                if df.is_empty():
                    print("no usable data, skipped")
                    time.sleep(random.uniform(0.75, 3.0))
                    continue

                write_monthly(df, threshold, month, url, MONTHLY_JSON, monthly_con)
                print(f"{len(df):,} outliers (monthly threshold: {threshold:.2f} mi)")

                # Explicitly free the DataFrame before next iteration
                del df
                gc.collect()

                time.sleep(random.uniform(0.75, 3.0))

            except Exception as e:
                print(f"Error processing URL '{url}': {e}, skipping file.")
                gc.collect()
                time.sleep(random.uniform(0.75, 3.0))

        total = monthly_con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
        print(f"\nMonthly pass complete — {total:,} total outlier trips")
        print(f"Wrote {MONTHLY_JSON}")
        print(f"Wrote {MONTHLY_DUCKDB}")

        done_count = monthly_con.execute("SELECT COUNT(DISTINCT month) FROM trips").fetchone()[0]
        if done_count < len(urls):
            print(
                f"\nSkipping global pass — {done_count} of {len(urls)} month(s) complete. "
                f"Re-run to process the remaining {len(urls) - done_count} "
                f"and then compute the global threshold."
            )
        else:
            write_global(monthly_con, percentile)

    except Exception as e:
        print(f"An error occurred during monthly processing: {e}")
    finally:
        monthly_con.close()

# ── Entry point ───────────────────────────────────────────────────────────────

def clean():
    # Delete existing output files if they exist
    MONTHLY_JSON.unlink(missing_ok=True)
    wal_path = Path(str(MONTHLY_DUCKDB) + ".wal")
    if MONTHLY_DUCKDB.exists():
        MONTHLY_DUCKDB.unlink()
    if wal_path.exists():
        wal_path.unlink()

    print("Cleaned up existing monthly output files.")

if __name__ == "__main__":
    #clean()
    process_yellow()