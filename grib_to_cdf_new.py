import xarray as xr
import os
import glob
import logging
import shutil
from tqdm import tqdm  # For progress bar
import sys
import cfgrib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded paths - CHANGE THESE TO YOUR ACTUAL PATHS
GRIB_FOLDER = os.getenv("GRIB_FOLDER", "grib_files")  # Directory containing GRIB files
NETCDF_OUTPUT_FOLDER = os.getenv("NETCDF_OUTPUT_FOLDER", "netcdf_files")  # Directory for individual NetCDF files
MERGED_NETCDF_FILE = os.getenv("MERGED_NETCDF_FILE", "era5_2023_merged.nc")  # Path for final merged file
TEMP_DIR = os.getenv("TEMP_DIR", "temp")  # Temporary directory for intermediate files

# Configuration
MAX_FILES_IN_MEMORY = 5  # Maximum number of files to open simultaneously for merging


def validate_metadata(nc_file):
    """
    Validates the metadata of a NetCDF file to ensure consistency.
    
    Parameters:
    - nc_file (str): Path to the NetCDF file to validate.
    
    Returns:
    - bool: True if the file passes validation, False otherwise.
    """
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(nc_file)
        
        # Check for critical dimensions
        required_dims = ["time", "latitude", "longitude"]
        missing_dims = [dim for dim in required_dims if dim not in ds.dims]
        
        if missing_dims:
            print(f"❌ Missing critical dimensions in {nc_file}: {missing_dims}")
            return False
        
        # Check for required global attributes
        required_attrs = ["Conventions", "title", "institution", "source"]
        missing_attrs = [attr for attr in required_attrs if attr not in ds.attrs]
        
        if missing_attrs:
            print(f"⚠️ Missing global attributes in {nc_file}: {missing_attrs}")
            return False
        
        # Check for variable consistency
        if not all(var in ds.data_vars for var in ["t2m", "sp", "u10", "v10"]):
            print(f"⚠️ Missing expected data variables in {nc_file}")
            return False
        
        print(f"✅ {nc_file} passed metadata validation.")
        return True
    
    except Exception as e:
        print(f"❌ Failed to validate {nc_file}: {e}")
        return False


def check_dependencies():
    """
    Check if all required dependencies are installed and properly configured.
    
    Returns:
    --------
    bool
        True if all dependencies are satisfied, False otherwise
    """
    try:

        logger.info("All required Python packages are installed")
        
        # Try to test if cfgrib is properly configured
        try:
            import cfgrib.messages
            # Try to modify the attribute if it exists
            if hasattr(cfgrib.messages, 'CHECK_ECMWF_CODES_PYTHON'):
                cfgrib.messages.CHECK_ECMWF_CODES_PYTHON = False
        except ImportError:
            pass  # It's okay if this specific module isn't found
            
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install xarray cfgrib tqdm")
        return False
    except Exception as e:
        logger.error(f"Error with dependency setup: {e}")
        logger.error("Make sure ecCodes C library is installed on your system.")
        logger.error("For Linux: 'apt-get install libeccodes-dev' or equivalent")
        logger.error("For macOS: 'brew install eccodes'")
        logger.error("For Windows: see https://github.com/ecmwf/cfgrib#installation")
        return False

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def merge_yearly_files(input_folder, output_file):
    """
    Merges multiple years of NetCDF files into a single dataset.
    
    Parameters:
    - input_folder (str): Directory containing yearly NetCDF files.
    - output_file (str): Path for the merged NetCDF output file.
    
    Returns:
    - Merged xarray.Dataset
    """
    # Find all NetCDF files in the input directory
    nc_files = sorted(glob.glob(os.path.join(input_folder, "*.nc")))
    
    if not nc_files:
        print("No NetCDF files found in the directory.")
        return
    
    # Open all NetCDF files as a single merged dataset
    try:
        merged_ds = xr.open_mfdataset(nc_files, combine='by_coords', parallel=True)
        merged_ds.to_netcdf(output_file)
        print(f"✅ Successfully merged {len(nc_files)} files into {output_file}")
    except Exception as e:
        print(f"❌ Failed to merge files: {e}")
        return
    
    return merged_ds

def auto_scan_directory(input_folder, output_file, validate=True):
    """
    Automatically scans a directory for NetCDF files, validates them, 
    and merges the valid files into a single output file.
    
    Parameters:
    - input_folder (str): Directory containing NetCDF files.
    - output_file (str): Path for the merged NetCDF output file.
    - validate (bool): Whether to validate files before merging (default True).
    
    Returns:
    - Merged xarray.Dataset (if successful), None otherwise.
    """
    # Find all NetCDF files
    nc_files = sorted([os.path.join(input_folder, f) 
                       for f in os.listdir(input_folder) if f.endswith(".nc")])
    
    if not nc_files:
        print("❌ No NetCDF files found in the directory.")
        return None
    
    valid_files = []
    
    # Validate each file if requested
    for nc_file in nc_files:
        if not validate or validate_metadata(nc_file):
            valid_files.append(nc_file)
    
    if not valid_files:
        print("❌ No valid NetCDF files found after validation.")
        return None
    
    # Merge valid files
    try:
        merged_ds = xr.open_mfdataset(valid_files, combine='by_coords', parallel=True)
        merged_ds.to_netcdf(output_file)
        print(f"✅ Successfully merged {len(valid_files)} files into {output_file}")
        return merged_ds
    except Exception as e:
        print(f"❌ Failed to merge files: {e}")
        return None


def grib_to_netcdf(input_file, output_folder):
    """
    Convert a GRIB file to NetCDF format.
    
    Parameters:
    -----------
    input_file : str
        Path to the input GRIB file
    output_folder : str
        Path to the folder where NetCDF files will be saved
    
    Returns:
    --------
    str or None
        Path to the output NetCDF file if successful, None otherwise
    """
    # Create output filename based on input filename
    base_filename = os.path.basename(input_file)
    base_name = os.path.splitext(base_filename)[0]
    output_file = os.path.join(output_folder, f"{base_name}.nc")
    
    try:
        # Open the GRIB file using cfgrib engine with additional options for better compatibility
        ds = xr.open_dataset(
            input_file, 
            engine='cfgrib',
            backend_kwargs={
                'errors': 'ignore',  # Skip unreadable messages
                'filter_by_keys': {}  # Don't filter any messages
            }
        )
        
        # Save to NetCDF with compression for smaller file size
        ds.to_netcdf(
            output_file,
            encoding={var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
        )
        
        # Close the dataset to free memory
        ds.close()
        
        logger.info(f"Converted {base_filename} to NetCDF successfully")
        return output_file
    
    except Exception as e:
        logger.error(f"Error converting {base_filename}: {e}")
        return None

def convert_all_grib_files(grib_folder, output_folder):
    """
    Convert all GRIB files in a folder to NetCDF format.
    
    Parameters:
    -----------
    grib_folder : str
        Path to the folder containing GRIB files
    output_folder : str
        Path to the folder where NetCDF files will be saved
    
    Returns:
    --------
    list
        List of successfully converted NetCDF file paths
    """
    # Ensure output directory exists
    ensure_dir_exists(output_folder)
    
    # Find all GRIB files (handling different possible extensions)
    grib_files = []
    for ext in ["*.grib", "*.grib2", "*.grb", "*.grb2"]:
        grib_files.extend(glob.glob(os.path.join(grib_folder, ext)))
    
    if not grib_files:
        logger.warning(f"No GRIB files found in {grib_folder}")
        return []
    
    logger.info(f"Found {len(grib_files)} GRIB files to convert")
    
    # Convert each GRIB file and collect successful conversions
    successful_conversions = []
    for grib_file in tqdm(grib_files, desc="Converting GRIB files"):
        result = grib_to_netcdf(grib_file, output_folder)
        if result:
            successful_conversions.append(result)
    
    logger.info(f"Successfully converted {len(successful_conversions)} out of {len(grib_files)} files")
    return successful_conversions

def identify_merge_dimension(netcdf_files, sample_size=3):
    """
    Identify the best dimension to merge along by examining a sample of files.
    
    Parameters:
    -----------
    netcdf_files : list
        List of NetCDF file paths
    sample_size : int
        Number of files to examine for dimension detection
    
    Returns:
    --------
    str or None
        Name of the dimension to merge along, or None if no suitable dimension is found
    """
    if len(netcdf_files) <= 1:
        return None
        
    # Take a sample of files to examine
    sample_files = netcdf_files[:min(sample_size, len(netcdf_files))]
    
    try:
        # Open sample datasets
        sample_datasets = [xr.open_dataset(file) for file in sample_files]
        
        # Common dimensions to check in order of preference
        dimensions_to_check = ['time', 'step', 'valid_time', 'forecast_time', 'realization']
        
        # Check each dimension
        for dim in dimensions_to_check:
            # Check if all datasets have this dimension
            if all(dim in ds.dims for ds in sample_datasets):
                # Check if values are different across datasets (makes sense to concatenate)
                
                # Get values from each dataset carefully
                dim_values = []
                for ds in sample_datasets:
                    # Handle different data formats safely
                    values = ds[dim].values
                    # Just store a hash or representative value to compare
                    if hasattr(values, 'size') and values.size > 0:
                        # For numpy arrays
                        dim_values.append(hash(values.tobytes()))
                    elif hasattr(values, '__iter__') and len(values) > 0:
                        # For regular iterables
                        dim_values.append(hash(str(values[0])))
                    else:
                        # For single values
                        dim_values.append(hash(str(values)))
                
                # Check if we have at least 2 different sets of values
                if len(set(dim_values)) > 1:
                    logger.info(f"Identified '{dim}' as suitable merge dimension")
                    
                    # Close all datasets
                    for ds in sample_datasets:
                        ds.close()
                        
                    return dim
        
        # If no suitable dimension found, use coordinate-based merge
        logger.info("No suitable dimension found for concatenation, will use coordinate-based merge")
        
        # Close all datasets
        for ds in sample_datasets:
            ds.close()
            
        return None
        
    except Exception as e:
        logger.error(f"Error identifying merge dimension: {e}")
        # Ensure datasets are closed
        try:
            for ds in sample_datasets:
                ds.close()
        except:
            pass
        return None

def merge_netcdf_files_chunked(netcdf_files, output_file, merge_dim=None, chunk_size=None):
    """
    Memory-efficient merging of NetCDF files by processing in chunks.
    
    Parameters:
    -----------
    netcdf_files : list
        List of NetCDF file paths to merge
    output_file : str
        Path to the output merged NetCDF file
    merge_dim : str or None
        Dimension to merge along, or None for automatic detection
    chunk_size : int or None
        Number of files to process at once, defaults to MAX_FILES_IN_MEMORY
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if not netcdf_files:
        logger.error("No NetCDF files to merge")
        return False
        
    if len(netcdf_files) == 1:
        logger.info("Only one file to process, copying instead of merging")
        shutil.copy2(netcdf_files[0], output_file)
        return True
    
    # Use default chunk size if not specified
    if chunk_size is None:
        chunk_size = MAX_FILES_IN_MEMORY
    
    # Create temp directory
    ensure_dir_exists(TEMP_DIR)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    ensure_dir_exists(output_dir)
    
    try:
        # Identify merge dimension if not provided
        if merge_dim is None:
            merge_dim = identify_merge_dimension(netcdf_files)
        
        # Divide files into manageable chunks
        chunks = [netcdf_files[i:i+chunk_size] for i in range(0, len(netcdf_files), chunk_size)]
        logger.info(f"Processing {len(netcdf_files)} files in {len(chunks)} chunks")
        
        # Process each chunk
        temp_files = []
        for i, chunk in enumerate(tqdm(chunks, desc="Merging chunks")):
            temp_file = os.path.join(TEMP_DIR, f"temp_merge_{i}.nc")
            
            # Open datasets in this chunk
            datasets = [xr.open_dataset(file) for file in chunk]
            
            # Merge datasets
            if merge_dim and all(merge_dim in ds.dims for ds in datasets):
                # Merge along specified dimension
                merged_ds = xr.concat(datasets, dim=merge_dim)
                logger.debug(f"Merged chunk {i+1}/{len(chunks)} along {merge_dim} dimension")
            else:
                # Fall back to coordinate-based merge
                merged_ds = xr.merge(datasets)
                logger.debug(f"Merged chunk {i+1}/{len(chunks)} using coordinate-based merge")
            
            # Save merged chunk
            merged_ds.to_netcdf(
                temp_file,
                encoding={var: {'zlib': True, 'complevel': 5} for var in merged_ds.data_vars}
            )
            
            # Close all datasets
            for ds in datasets:
                ds.close()
            
            # Close merged dataset
            merged_ds.close()
            
            temp_files.append(temp_file)
        
        # Now merge the temporary files
        if len(temp_files) == 1:
            # Only one temp file, just rename it
            shutil.move(temp_files[0], output_file)
        else:
            # Merge temp files using the same method
            logger.info(f"Merging {len(temp_files)} intermediate files")
            
            # Open all temp datasets
            temp_datasets = [xr.open_dataset(file) for file in temp_files]
            
            # Merge temp datasets
            if merge_dim and all(merge_dim in ds.dims for ds in temp_datasets):
                final_merged = xr.concat(temp_datasets, dim=merge_dim)
            else:
                final_merged = xr.merge(temp_datasets)
            
            # Save final merged file
            final_merged.to_netcdf(
                output_file,
                encoding={var: {'zlib': True, 'complevel': 5} for var in final_merged.data_vars}
            )
            
            # Close all temp datasets
            for ds in temp_datasets:
                ds.close()
                
            # Close final merged dataset
            final_merged.close()
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        logger.info(f"Successfully merged all files into {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        logger.exception("Detailed traceback:")
        return False
    finally:
        # Make sure we clean up temp files even if there was an error
        for temp_file in glob.glob(os.path.join(TEMP_DIR, "temp_merge_*.nc")):
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def main():
    """Main function to process GRIB files and create merged NetCDF."""
    logger.info("Starting GRIB to NetCDF conversion and merge process")
    
    # Step 0: Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Exiting.")
        sys.exit(1)
    
    # Step 1: Convert all GRIB files to NetCDF
    netcdf_files = convert_all_grib_files(GRIB_FOLDER, NETCDF_OUTPUT_FOLDER)
    
    # Step 2: Merge all NetCDF files
    if netcdf_files:
        merge_success = merge_netcdf_files_chunked(netcdf_files, MERGED_NETCDF_FILE)
        if merge_success:
            logger.info("Process completed successfully!")
            
            # Show summary
            file_size_mb = os.path.getsize(MERGED_NETCDF_FILE) / (1024 * 1024)
            logger.info(f"Merged file size: {file_size_mb:.2f} MB")
            logger.info(f"Number of original files processed: {len(netcdf_files)}")
        else:
            logger.error("Failed to merge NetCDF files")
            sys.exit(1)
    else:
        logger.error("No NetCDF files were created, cannot merge")
        sys.exit(1)
        
    # Step 3: Clean up if requested (commented out by default)
    # Un-comment these lines if you want to remove individual NetCDF files after merging
    # logger.info("Cleaning up individual NetCDF files")
    # for file in netcdf_files:
    #     if os.path.exists(file):
    #         os.remove(file)

if __name__ == "__main__":
    main()

