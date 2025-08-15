#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import sys
import numpy as onp

from scipy.interpolate import interpn
import healpy as hp
from astropy.io import fits as pyfits

def parse_args():
    parser = argparse.ArgumentParser(description="Load a DataFrame and accept a FITS file path.")
    parser.add_argument('--df', required=True, type=str, help="Path to the DataFrame file (.parquet or .hdf)")
    parser.add_argument('--fits', required=True, type=str, help="Path to the FITS file")
    parser.add_argument('--out_key', required=True, type=str, help="Key where the new baseline weight will be saved")
    parser.add_argument('--energy_key', required=True, type=str)
    parser.add_argument('--ra_key', required=True, type=str)
    parser.add_argument('--dec_key', required=True, type=str)
    parser.add_argument('--fluxless_weight', required=True, type=str)
    return parser.parse_args()

def read_dataframe(df_path: str) -> pd.DataFrame:
    """Read a pandas DataFrame from a Parquet or HDF file."""
    if not os.path.isfile(df_path):
        raise FileNotFoundError(f"DataFrame file not found: {df_path}")

    if df_path.endswith('.parquet'):
        return pd.read_parquet(df_path)
    elif df_path.endswith('.hdf') or df_path.endswith('.h5'):
        return pd.read_hdf(df_path)
    else:
        raise ValueError("Unsupported DataFrame format. Use .parquet or .hdf")

def save_dataframe(df,df_path: str) -> pd.DataFrame:
    """Read a pandas DataFrame from a Parquet or HDF file."""
    if not os.path.isfile(df_path):
        raise FileNotFoundError(f"DataFrame file not found: {df_path}")

    if df_path.endswith('.parquet'):
        return df.to_parquet(df_path)
    elif df_path.endswith('.hdf') or df_path.endswith('.h5'):
        return df.to_hdf(df_path,key="data",mode="w")
    else:
        raise ValueError("Unsupported DataFrame format. Use .parquet or .hdf")

def load_map(file):
    #file = "/home/wecapstor3/capn/capn105h/data/cringe_fits/Neutrino_AAfrag_Galprop_Ferr_Fiducial_256.fits"
    maps = pyfits.open(
        file)[0].data  # these have a single hdulist entry
    return maps


def main():
    args = parse_args()
    try:
        df = read_dataframe(args.df)
        print("DataFrame loaded successfully:")
        print(df.head())
    except Exception as e:
        print(f"Error loading DataFrame: {e}", file=sys.stderr)
        sys.exit(1)

    

    maps = load_map(args.fits)
    maps *= 0.5

    rot = hp.Rotator(coord=["G","C"])

    # Skymaps have to be rotated one by one
    m = onp.array(maps)

    rotated_m = onp.empty_like(m)
    for i in range(m.shape[0]):
        rotated_m[i] = rot.rotate_map_pixel(m[i])
    
    maps = rotated_m


    nside = 256
    order = "RING"
    energies = onp.linspace(1, 8, 50)



    maps[maps == 0] += 1e-300 #avoid zeros

    log_maps = onp.log10(maps)

    log_E = onp.log10(df[args.energy_key])
    ra = df[args.ra_key]
    dec = df[args.dec_key]
    theta = onp.pi / 2 - dec
    phi = ra

    log_fluxes = hp.pixelfunc.get_interp_val(log_maps,
                                                 theta,
                                                 phi,
                                                 nest=False)
    
    numbers = onp.linspace(0, len(log_E) - 1, len(log_E))

    log_flux = interpn((numbers, energies),  # interpolate between map energies (and event numbers)
                       log_fluxes.T,       # fluxes for energies and event directions
                       xi=onp.array((numbers, log_E)).T, # return flux at event energies E (and event numbers)
                       bounds_error=False,  # enable linear extrapolation
                       fill_value=None)

    flux = 10**log_flux

    flux = flux * df[args.fluxless_weight]

    df[args.out_key] = flux

    

    save_dataframe(df,args.df)


if __name__ == "__main__":
    main()
    print("Done")