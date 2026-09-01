"""
Stream the FNSPID news feed and isolate AAPL headlines (the human/expert
side-channel). Faithful to the directive's intent, with a scan guard + progress
since the feed is sorted alphabetically by ticker (AAPL trails the A*/AA* block).
"""
import os
import pandas as pd
from datasets import load_dataset

DATA_DIR = "./real_world_fnspid/data"
OUT_PATH = os.path.join(DATA_DIR, "AAPL_news_filtered.csv")
TARGET = "AAPL"
MAX_HITS = 3000          # bound captured AAPL entries
MAX_SCAN = 1_000_000     # safety cap on rows scanned


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Filtering matching news stream from Hugging Face...")
    stream = load_dataset("Zihan1004/FNSPID", streaming=True)["train"]

    hits, scanned, passed_block = [], 0, False
    for row in stream:
        scanned += 1
        sym = row.get("Stock_symbol")
        if sym == TARGET:
            hits.append(row)
            passed_block = True
        elif passed_block:
            # alphabetical feed: once we exit the AAPL block we are done
            break
        if len(hits) >= MAX_HITS:
            break
        if scanned % 50_000 == 0:
            print(f"  scanned {scanned:,} rows | AAPL hits {len(hits)} | at symbol {sym}")
        if scanned >= MAX_SCAN:
            print(f"  scan cap {MAX_SCAN:,} reached.")
            break

    df = pd.DataFrame(hits)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df = df.sort_values("Date").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"News capture complete: {len(df)} targeted entries isolated "
          f"(scanned {scanned:,} rows) -> {OUT_PATH}")
    if not df.empty:
        print(f"  date range: {df['Date'].min()} -> {df['Date'].max()}")


if __name__ == "__main__":
    main()
