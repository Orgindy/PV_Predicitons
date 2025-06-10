import os
from clustering import main_matching_pipeline
import pandas as pd


def multi_year_matching_pipeline(
    years,
    base_input_path,
    output_dir,
    borders_path,
    k_range=range(2, 10),
):
    """Run ``main_matching_pipeline`` across multiple yearly cluster files.

    Parameters
    ----------
    years : list[int]
        List of years to process.
    base_input_path : str
        Directory where ``clustered_dataset_<year>.csv`` files are stored.
    output_dir : str
        Directory to write matched datasets to.
    borders_path : str
        Shapefile used for mapping in ``main_matching_pipeline``.
    k_range : range, optional
        Range of ``k`` values to evaluate for clustering.
    """

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for year in years:
        input_file = os.path.join(
            base_input_path,
            f"clustered_dataset_{year}.csv",
        )
        output_file = os.path.join(
            output_dir,
            f"matched_dataset_{year}.csv",
        )

        print(f"\n=== Processing {year} ===")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")

        if not os.path.exists(input_file):
            print(f"❌ Input file not found for {year}")
            continue

        try:
            df = main_matching_pipeline(
                clustered_data_path=input_file,
                shapefile_path=borders_path,
                output_file=output_file,
                k_range=k_range,
            )

            if df is not None:
                df["Year"] = year
                results.append(df)
        except Exception as e:
            print(f"❌ Error processing year {year}: {e}")

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined_file = os.path.join(
            output_dir,
            "matched_dataset_all_years.csv",
        )
        combined.to_csv(combined_file, index=False)
        print(f"\n✅ Combined multi-year dataset saved to {combined_file}")
        return combined

    print("⚠️ No results generated.")
    return None


if __name__ == "__main__":
    years = [2020, 2021, 2022, 2023]
    base_input_path = "results/clusters/"
    output_dir = "results/matching/"
    borders_path = "data/borders/ne_10m_admin_0_countries.shp"

    multi_year_matching_pipeline(
        years,
        base_input_path,
        output_dir,
        borders_path,
    )
