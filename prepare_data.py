#!/usr/bin/env python3
"""
prepare_data.py
---------------
ETL: Greenbyte Kelmarsh SCADA raw CSVs -> clean Parquet.

Reads all Turbine_Data_Kelmarsh_1_*.csv (Turbine 1 only by default),
extracts timestamp / wind_speed_ms / power_kw, drops NaN, clips negatives,
and writes data/kelmarsh_turbine1_all.parquet.

Usage:
    python prepare_data.py                   # Turbine 1, all years
    python prepare_data.py --turbine 1 2 3   # multiple turbines
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

RAW_DIR    = r"C:\Users\roy\kelmarsh"
OUT_DIR    = os.path.join(os.path.dirname(__file__), "data")

# Column positions in the Greenbyte export (0-indexed, after skiprows=9)
COL_WIND  = 1    # Wind speed (m/s)
COL_POWER = 61   # Power (kW)


def read_turbine_csv(path: str, turbine_id: int) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=9,
        encoding="latin1",
        usecols=[0, COL_WIND, COL_POWER],
    )
    # Strip Greenbyte comment prefix from first column name
    cols = list(df.columns)
    cols[0] = cols[0].replace("# ", "").strip()
    df.columns = ["timestamp", "wind_speed_ms", "power_kw"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["turbine"] = turbine_id
    return df


def load_turbine(turbine_id: int) -> pd.DataFrame:
    pattern = os.path.join(
        RAW_DIR, "**", f"Turbine_Data_Kelmarsh_{turbine_id}_*.csv"
    )
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files found for Turbine {turbine_id} in {RAW_DIR}")

    print(f"  Turbine {turbine_id}: found {len(files)} files")
    frames = []
    for f in files:
        yr = os.path.basename(f).split("_")[4]
        print(f"    reading {yr}...", end="", flush=True)
        chunk = read_turbine_csv(f, turbine_id)
        frames.append(chunk)
        print(f" {len(chunk):,} rows", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates("timestamp", inplace=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows where either signal is NaN
    df = df.dropna(subset=["wind_speed_ms", "power_kw"]).copy()
    # Clip non-physical values
    df["wind_speed_ms"] = df["wind_speed_ms"].clip(lower=0.01)
    df["power_kw"]      = df["power_kw"].clip(lower=0.0)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turbine", type=int, nargs="+", default=[1],
        help="Turbine IDs to process (default: 1)"
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    all_frames = []
    for tid in args.turbine:
        print(f"\n[Turbine {tid}] loading...")
        raw = load_turbine(tid)
        print(f"  Raw rows: {len(raw):,}")
        clean_df = clean(raw)
        print(f"  After cleaning: {len(clean_df):,} rows"
              f"  ({100 * len(clean_df) / len(raw):.1f}% retained)")
        print(f"  Wind: [{clean_df['wind_speed_ms'].min():.2f}, "
              f"{clean_df['wind_speed_ms'].max():.2f}] m/s")
        print(f"  Power: [{clean_df['power_kw'].min():.0f}, "
              f"{clean_df['power_kw'].max():.0f}] kW")
        all_frames.append(clean_df)

    df_all = pd.concat(all_frames, ignore_index=True)

    turbine_str = "_".join(str(t) for t in sorted(args.turbine))
    out_path = os.path.join(OUT_DIR, f"kelmarsh_turbine{turbine_str}_all.parquet")
    df_all.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[OK] Saved {len(df_all):,} rows -> {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
