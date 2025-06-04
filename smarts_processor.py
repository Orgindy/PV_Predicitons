import os
import numpy as np
import pandas as pd
import re
from datetime import datetime
import netCDF4 as nc
import glob
import argparse

def extract_metadata(out_file_content):
    """Extract metadata from the SMARTS .out.txt file"""
    metadata = {}

    # Extract location from reference line
    ref_match = re.search(r'Reference for this run: (.+)', out_file_content)
    if ref_match:
        metadata['reference'] = ref_match.group(1).strip()
    
    # Extract date and time from filename if it's in the format YYYYMMDD_HHMM
    date_match = re.search(r'(\d{8})_(\d{4})', metadata.get('reference', ''))
    if date_match:
        date_str = date_match.group(1)
        time_str = date_match.group(2)
        try:
            date_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
            metadata['date'] = date_obj.strftime("%Y-%m-%d")
            metadata['time'] = date_obj.strftime("%H:%M")
        except ValueError:
            metadata['date'] = 'unknown'
            metadata['time'] = 'unknown'
    
    # Extract location name
    loc_match = re.search(r'(\w+) Coast', metadata.get('reference', ''))
    if loc_match:
        metadata['location'] = loc_match.group(0).strip()
    else:
        metadata['location'] = 'unknown'
    
    # Extract atmospheric model
    atm_match = re.search(r'\* ATMOSPHERE\s*:\s*(\w+)', out_file_content)
    if atm_match:
        metadata['atmosphere'] = atm_match.group(1).strip()
    
    # Extract pressure
    pressure_match = re.search(r'Pressure \(mb\) = (\d+\.\d+)', out_file_content)
    if pressure_match:
        metadata['pressure_mb'] = float(pressure_match.group(1))
    
    # Extract relative humidity
    rh_match = re.search(r'Relative Humidity \(%\) = (\d+\.\d+)', out_file_content)
    if rh_match:
        metadata['relative_humidity_percent'] = float(rh_match.group(1))
    
    # Extract temperature
    temp_match = re.search(r'Instantaneous at site\'s altitude = (\d+\.\d+)', out_file_content)
    if temp_match:
        metadata['temperature_K'] = float(temp_match.group(1))
    
    # Extract ground type
    ground_match = re.search(r'Spectral ZONAL albedo data: (\w+)', out_file_content)
    if ground_match:
        metadata['ground_type'] = ground_match.group(1).strip()
    
    # Extract solar position
    zenith_match = re.search(r'Zenith Angle \(apparent\) = (\d+\.\d+)', out_file_content)
    if zenith_match:
        metadata['solar_zenith_angle'] = float(zenith_match.group(1))
    
    azimuth_match = re.search(r'Azimuth \(from North\) = (\d+\.\d+)', out_file_content)
    if azimuth_match:
        metadata['solar_azimuth'] = float(azimuth_match.group(1))
    
    # Extract surface tilt information
    tilt_match = re.search(r'Surface Tilt =\s+(\d+\.\d+)', out_file_content)
    if tilt_match:
        metadata['surface_tilt'] = float(tilt_match.group(1))
    
    # Extract broadband irradiance values
    direct_beam_match = re.search(r'Direct Beam =\s+(\d+\.\d+)', out_file_content)
    if direct_beam_match:
        metadata['direct_beam_horizontal'] = float(direct_beam_match.group(1))
    
    diffuse_match = re.search(r'Diffuse = (\d+\.\d+)', out_file_content)
    if diffuse_match:
        metadata['diffuse_horizontal'] = float(diffuse_match.group(1))
    
    global_match = re.search(r'Global =\s+(\d+\.\d+)', out_file_content)
    if global_match:
        metadata['global_horizontal'] = float(global_match.group(1))
    
    # Extract clearness index
    kt_match = re.search(r'Clearness index, KT = (\d+\.\d+)', out_file_content)
    if kt_match:
        metadata['clearness_index'] = float(kt_match.group(1))
    
    return metadata

def parse_ext_file(ext_file_content):
    """Parse spectral data from the SMARTS .ext.txt file"""
    lines = ext_file_content.strip().split('\n')
    
    # Extract header line to get column names
    header = lines[0].split()
    
    # Convert rest of lines to DataFrame
    data = []
    for line in lines[1:]:
        values = line.split()
        if len(values) == len(header):
            # Convert scientific notation to float
            values = [float(v.replace('E+', 'e+').replace('E-', 'e-')) for v in values]
            data.append(values)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # Convert wavelength from nanometers to micrometers for easier band definition
    df['wavelength_um'] = df['Wvlgth'] / 1000
    
    return df

def define_spectral_bands(df):
    """Define and calculate the 5 spectral bands"""
    # Define band ranges in micrometers
    bands = {
        'UV': (0.28, 0.4),
        'Blue': (0.4, 0.5),
        'Green': (0.5, 0.6),
        'Red': (0.6, 0.7),
        'IR': (0.7, 4.0)
    }
    
    # Calculate irradiance in each band by spectral integration
    band_irradiance = {}
    for band_name, (wl_min, wl_max) in bands.items():
        band_data = df[(df['wavelength_um'] >= wl_min) & (df['wavelength_um'] <= wl_max)]
        
        # Check if we have data in this range
        if len(band_data) == 0:
            band_irradiance[band_name] = {
                'Global_tilted_irradiance': 0,
                'Beam_normal_+circumsolar': 0,
                'Difuse_horiz-circumsolar': 0,
                'Zonal_ground_reflectance': 0,
                'wavelength_range': f"{wl_min}-{wl_max} µm",
                'num_points': 0
            }
            continue
        
        # Calculate integrated values for each column of interest using trapezoidal integration
        band_irradiance[band_name] = {
            'Global_tilted_irradiance': np.trapz(band_data['Global_tilted_irradiance'], band_data['wavelength_um']),
            'Beam_normal_+circumsolar': np.trapz(band_data['Beam_normal_+circumsolar'], band_data['wavelength_um']),
            'Difuse_horiz-circumsolar': np.trapz(band_data['Difuse_horiz-circumsolar'], band_data['wavelength_um']),
            'Zonal_ground_reflectance': np.mean(band_data['Zonal_ground_reflectance']),  # Mean for reflectance
            'wavelength_range': f"{wl_min}-{wl_max} µm",
            'num_points': len(band_data),
            'min_wavelength': wl_min,
            'max_wavelength': wl_max
        }
    
    return band_irradiance

def process_smarts_files(out_file_path, ext_file_path):
    """Process a pair of SMARTS output files and return metadata and band irradiance"""
    # Read file contents
    with open(out_file_path, 'r') as f:
        out_file_content = f.read()
    
    with open(ext_file_path, 'r') as f:
        ext_file_content = f.read()
    
    # Extract metadata
    metadata = extract_metadata(out_file_content)
    
    # Parse spectral data
    df = parse_ext_file(ext_file_content)
    
    # Define and calculate spectral bands
    band_irradiance = define_spectral_bands(df)
    
    return metadata, band_irradiance, df

def save_to_csv(metadata, band_irradiance, output_dir, filename_base):
    """Save the processed data to a CSV file"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame from the band irradiance data
    data = []
    for band, values in band_irradiance.items():
        row = {
            'band': band,
            'min_wavelength_um': values['min_wavelength'],
            'max_wavelength_um': values['max_wavelength'],
            'global_tilted_irradiance_W_m2': values['Global_tilted_irradiance'],
            'beam_normal_irradiance_W_m2': values['Beam_normal_+circumsolar'],
            'diffuse_horizontal_irradiance_W_m2': values['Difuse_horiz-circumsolar'],
            'average_ground_reflectance': values['Zonal_ground_reflectance'],
            'num_spectral_points': values['num_points']
        }
        data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add metadata columns
    for key, value in metadata.items():
        df[key] = value
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"{filename_base}_bands.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved band data to {output_path}")
    
    return output_path

def save_to_netcdf(metadata, band_irradiance, spectral_df, output_dir, filename_base):
    """Save the processed data to a NetCDF file with metadata"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path
    output_path = os.path.join(output_dir, f"{filename_base}_bands.nc")
    
    # Create a new NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
        # Add global attributes (metadata)
        for key, value in metadata.items():
            ncfile.setncattr(key, value)
        
        # Add dimensions
        ncfile.createDimension('band', len(band_irradiance))
        
        # Define band names
        band_names = list(band_irradiance.keys())
        band_var = ncfile.createVariable('band_name', str, ('band',))
        band_var[:] = band_names
        band_var.long_name = 'Spectral band name'
        
        # Define band boundaries
        min_wl = ncfile.createVariable('min_wavelength', 'f4', ('band',))
        max_wl = ncfile.createVariable('max_wavelength', 'f4', ('band',))
        min_wl[:] = [band_irradiance[band]['min_wavelength'] for band in band_names]
        max_wl[:] = [band_irradiance[band]['max_wavelength'] for band in band_names]
        min_wl.units = 'micrometers'
        max_wl.units = 'micrometers'
        min_wl.long_name = 'Lower boundary of spectral band'
        max_wl.long_name = 'Upper boundary of spectral band'
        
        # Create variables for irradiance data
        global_var = ncfile.createVariable('global_tilted_irradiance', 'f4', ('band',))
        beam_var = ncfile.createVariable('beam_normal_irradiance', 'f4', ('band',))
        diffuse_var = ncfile.createVariable('diffuse_horizontal_irradiance', 'f4', ('band',))
        reflectance_var = ncfile.createVariable('average_ground_reflectance', 'f4', ('band',))
        
        # Set variable attributes
        global_var.units = 'W/m^2'
        global_var.long_name = 'Global tilted irradiance'
        
        beam_var.units = 'W/m^2'
        beam_var.long_name = 'Beam normal irradiance plus circumsolar'
        
        diffuse_var.units = 'W/m^2'
        diffuse_var.long_name = 'Diffuse horizontal irradiance minus circumsolar'
        
        reflectance_var.units = 'unitless'
        reflectance_var.long_name = 'Average zonal ground reflectance'
        
        # Fill variables with data
        global_var[:] = [band_irradiance[band]['Global_tilted_irradiance'] for band in band_names]
        beam_var[:] = [band_irradiance[band]['Beam_normal_+circumsolar'] for band in band_names]
        diffuse_var[:] = [band_irradiance[band]['Difuse_horiz-circumsolar'] for band in band_names]
        reflectance_var[:] = [band_irradiance[band]['Zonal_ground_reflectance'] for band in band_names]
        
        # Add more metadata
        ncfile.description = 'Spectral radiation data aggregated into 5 bands'
        ncfile.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        ncfile.source = 'SMARTS model output processed with smarts_processor.py'
    
    print(f"Saved NetCDF file to {output_path}")
    return output_path

def find_smarts_file_pairs(input_dir):
    """Find pairs of .out.txt and .ext.txt files that belong together"""
    out_files = glob.glob(os.path.join(input_dir, "*.out.txt"))
    pairs = []
    
    for out_file in out_files:
        # Get the base name without extension
        base_name = os.path.splitext(os.path.splitext(out_file)[0])[0]
        ext_file = f"{base_name}.ext.txt"
        
        # Check if the .ext.txt file exists
        if os.path.exists(ext_file):
            pairs.append((out_file, ext_file))
    
    return pairs

def process_directory(input_dir, output_dir):
    """Process all SMARTS file pairs in a directory"""
    # Find all matching pairs of .out.txt and .ext.txt files
    file_pairs = find_smarts_file_pairs(input_dir)
    
    if not file_pairs:
        print(f"No SMARTS file pairs found in {input_dir}")
        return
    
    print(f"Found {len(file_pairs)} SMARTS file pairs to process")
    
    # Process each pair
    for out_file, ext_file in file_pairs:
        print(f"Processing {os.path.basename(out_file)} and {os.path.basename(ext_file)}...")
        
        # Get the base filename without path and extension
        base_name = os.path.splitext(os.path.splitext(os.path.basename(out_file))[0])[0]
        
        # Process the files
        metadata, band_irradiance, spectral_df = process_smarts_files(out_file, ext_file)
        
        # Save to CSV and NetCDF
        csv_path = save_to_csv(metadata, band_irradiance, output_dir, base_name)
        nc_path = save_to_netcdf(metadata, band_irradiance, spectral_df, output_dir, base_name)
        print(f"✅ Saved: {csv_path}")
        print(f"✅ Saved: {nc_path}")

        print(f"Completed processing {base_name}")
        print("-" * 60)

def main():
    input_dir = r"C:\Users\gindi002\DATASET\smarts_out_files"
    output_dir = r"C:\Users\gindi002\DATASET\processed_spectral"

    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()
