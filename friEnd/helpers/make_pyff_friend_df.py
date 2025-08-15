#!/usr/bin/env python3
"""
Script to read a DataFrame from HDF5 or Parquet, process it, and save as Parquet.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a DataFrame from HDF5/Parquet and save as Parquet."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input DataFrame (.h5 or .parquet)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output Parquet file"
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to output Parquet file"
    )

    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Error: input file '{path}' does not exist.")
    
    suffix = path.suffix.lower()
    if suffix in [".h5", ".hdf", ".hdf5"]:
        return pd.read_hdf(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        sys.exit(f"Unsupported file format: {suffix}")


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: Implement your custom processing logic here.
    For now, this just returns the DataFrame unchanged.
    """
    return df


def main():
    args = parse_args()

    # Load
    df = load_dataframe(args.input)

    # Process
    df_processed = process_dataframe(df)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(args.output, index=False)
    print(f"Saved processed DataFrame to {args.output}")


if __name__ == "__main__":
    main()