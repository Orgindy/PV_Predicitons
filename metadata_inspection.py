import os
import xarray as xr
import json
from pprint import pprint
from config import get_nc_dir

def show_metadata(nc_path: str):
    """
    Open the NetCDF file at `nc_path` and print:
      • global attributes
      • dimensions
      • coordinates (with their attrs)
      • each data variable (dims, dtype, shape, attrs)
    """
    ds = xr.open_dataset(nc_path)
    
    print("\n=== GLOBAL ATTRIBUTES ===")
    for key, val in ds.attrs.items():
        print(f"{key}: {val!r}")
    if 'crs' not in ds.attrs:
        print("⚠️ CRS metadata not found")

    print("\n=== DIMENSIONS ===")
    for dim, length in ds.dims.items():
        print(f"{dim}: {length}")

    print("\n=== COORDINATES ===")
    for coord in ds.coords:
        da = ds.coords[coord]
        print(f"\n– {coord}")
        print(f"   dims : {da.dims}")
        print(f"   length : {da.size}")
        if da.attrs:
            print("   attrs:")
            for k, v in da.attrs.items():
                print(f"     {k}: {v!r}")
        if 'units' not in da.attrs:
            print("     ⚠️ units attribute missing")

    print("\n=== DATA VARIABLES ===")
    for var in ds.data_vars:
        da = ds[var]
        print(f"\n– {var}")
        print(f"   dims  : {da.dims}")
        print(f"   dtype : {da.dtype}")
        print(f"   shape : {da.shape}")
        if da.attrs:
            print("   attrs:")
            for k, v in da.attrs.items():
                print(f"     {k}: {v!r}")

    ds.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect NetCDF metadata")
    parser.add_argument("nc_file", nargs="?", default=os.path.join(get_nc_dir(), "ERA5_daily.nc"),
                        help="Path to NetCDF file")
    args = parser.parse_args()

    nc_file = args.nc_file
    if not os.path.isabs(nc_file):
        nc_file = os.path.join(get_nc_dir(), nc_file)

    if not os.path.exists(nc_file):
        print(f"❌ File not found: {nc_file}")
    else:
        show_metadata(nc_file)
