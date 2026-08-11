"""
Pull the full DOHMH restaurant inspection dataset (43nn-pn8j) from the NYC
Open Data SODA API and save it as one raw CSV.

One row per violation, not per inspection. Collapsing happens in features.py.

    python src/pull_data.py
"""
import io
import time

import pandas as pd
import requests

from config import PAGE_SIZE, RAW_CSV, SODA_URL


def pull_all(url=SODA_URL, page_size=PAGE_SIZE, pause=0.5):
    """Page through the API until it stops returning rows."""
    chunks, offset = [], 0
    while True:
        r = requests.get(
            url,
            params={
                "$limit": page_size,
                "$offset": offset,
                "$order": "camis,inspection_date",
            },
            timeout=120,
        )
        r.raise_for_status()
        part = pd.read_csv(io.StringIO(r.text), low_memory=False)
        if len(part) == 0:
            break
        chunks.append(part)
        offset += page_size
        print(f"  pulled {offset:,} rows...")
        time.sleep(pause)
    return pd.concat(chunks, ignore_index=True)


def main():
    print("pulling from SODA API (this takes a few minutes)")
    raw = pull_all()
    raw.to_csv(RAW_CSV, index=False)

    print(f"\nsaved {len(raw):,} violation rows -> {RAW_CSV}")
    dates = pd.to_datetime(raw.inspection_date, errors="coerce")
    print("\nrows by year:")
    print(dates.dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
