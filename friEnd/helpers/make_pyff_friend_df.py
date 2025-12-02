#!/usr/bin/env python3
"""
Script to read a DataFrame from HDF5 or Parquet, process it, and save as Parquet.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

from friEnd import PyFF_Friend
import pyForwardFolding as pyFF

import jax.numpy as jnp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a DataFrame from HDF5/Parquet and save as Parquet."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to output Parquet file"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear previous created key"
    )

    return parser.parse_args()



def main():
    args = parse_args()

    config = args.config

    model_params, priors = pyFF.config.params_from_config(config)

    fr = PyFF_Friend(config,model_params,clear=args.clear)
    fr.add_weights()
    fr._save_output()
    


if __name__ == "__main__":
    main()
    print("Done")