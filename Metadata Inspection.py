import xarray as xr
import json
from pprint import pprint

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
    parser.add_argument("nc_file", help="Path to NetCDF file")
    args = parser.parse_args()

    if not os.path.exists(args.nc_file):
        print(f"❌ File not found: {args.nc_file}")
    else:
        show_metadata(args.nc_file)
