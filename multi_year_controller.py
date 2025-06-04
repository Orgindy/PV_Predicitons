import os
from clustering import main_matching_pipeline  # <-- REQUIRED import
import pandas as pd

# Rest of your existing code remains the same...
def multi_year_matching_pipeline(years, base_input_path, output_dir, borders_path, k_range=range(2, 10)):
    """
    Run the matching pipeline across multiple years.
    """
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        input_file = os.path.join(base_input_path, f"clustered_dataset_{year}.csv")
        output_file = os.path.join(output_dir, f"matched_dataset_{year}.csv")

        print(f"\n=== Processing {year} ===")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        # Add your actual processing logic here
        # For now, just copy or process the file as needed
        try:
            # Placeholder - replace with your actual logic
            if os.path.exists(input_file):
                print(f"✅ Found input file for {year}")
                # Your processing code here
            else:
                print(f"❌ Input file not found for {year}")
        except Exception as e:
            print(f"❌ Error processing year {year}: {e}")

if __name__ == "__main__":
    years = [2020, 2021, 2022, 2023]
    base_input_path = "results/clusters/"
    output_dir = "results/matching/"
    borders_path = "data/borders/ne_10m_admin_0_countries.shp"

    multi_year_matching_pipeline(years, base_input_path, output_dir, borders_path)

