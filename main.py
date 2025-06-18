import os
import logging
from datetime import datetime
import pandas as pd
import argparse

from utils.resource_monitor import ResourceMonitor

# Import pipeline functions from the clustering module
from clustering import (
    prepare_clustered_dataset,
    main_matching_pipeline,
    plot_prediction_uncertainty_with_contours,
    compute_cluster_summary,
    compute_pv_potential_by_cluster_year,
    prepare_features_for_clustering,
)

from sklearn.model_selection import train_test_split
from train_models import train_all_models


def parse_args():
    parser = argparse.ArgumentParser(description="Run RC-PV pipeline")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "prep", "cluster"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "--input-file", default="clustered_dataset.csv", help="Path to input CSV file"
    )
    parser.add_argument(
        "--db-url", default=os.getenv("PV_DB_URL"), help="Optional database URL"
    )
    parser.add_argument(
        "--db-table",
        default=os.getenv("PV_DB_TABLE", "pv_data"),
        help="Table name if using DB",
    )
    return parser.parse_args()


def validate_environment(args: argparse.Namespace) -> bool:
    """Validate database configuration when DB usage is requested."""
    if args.db_url is None:
        return True

    if "://" not in args.db_url:
        logging.error("Invalid PV_DB_URL format: %s", args.db_url)
        return False

    if not args.db_table:
        logging.error("PV_DB_TABLE must be provided when using a database")
        return False

    return True


def check_required_files(input_file):
    """Check if required input files exist."""
    required_files = [input_file]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        logging.warning(f"Missing files: {missing_files}")
        return False
    return True


def main_rc_pv_pipeline(input_path, db_url=None, db_table="pv_data"):
    """Complete RC-PV pipeline using available functions."""

    processed_path = "data/clustered_dataset_enhanced.csv"
    output_path = "matched_dataset.csv"

    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/maps", exist_ok=True)

    if not ResourceMonitor.check_memory_usage():
        logging.error("Insufficient memory")
        return None
    stats = ResourceMonitor.get_memory_stats()
    logging.info(
        f"Memory usage: {stats['percent_used']:.1f}% of {stats['total_gb']:.1f} GB"
    )

    if db_url:
        from database_utils import read_table, write_dataframe

        try:
            df_db = read_table(db_table, db_url=db_url)
        except Exception as e:
            logging.error(f"Failed to read table {db_table}: {e}")
            return None
        input_path = "db_input.csv"
        df_db.to_csv(input_path, index=False)

    # Step 1: Prepare and enhance dataset
    logging.info("🧼 Step 1: Preparing enhanced dataset")
    try:
        df_prepared = prepare_clustered_dataset(
            input_path=input_path, output_path=processed_path
        )
        if df_prepared is not None:
            logging.info(f"✅ Dataset prepared with {len(df_prepared)} rows")
        else:
            logging.warning("Dataset preparation returned None, using original file")
            processed_path = input_path
    except Exception as e:
        logging.warning(f"Dataset preparation failed: {e}")
        logging.info("Continuing with original input file...")
        processed_path = input_path

    # Step 1b: Train ML models on prepared dataset
    logging.info("🤖 Step 1b: Training prediction models")
    try:
        if "PV_Potential_physics" in df_prepared.columns:
            target_col = "PV_Potential_physics"
        elif "PV_Potential" in df_prepared.columns:
            target_col = "PV_Potential"
        else:
            target_col = None

        if target_col:
            X_scaled, _, _ = prepare_features_for_clustering(
                df_prepared, use_predicted_pv=False
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                df_prepared[target_col].values,
                test_size=0.2,
                random_state=42,
            )
            _, perf = train_all_models(X_train, X_test, y_train, y_test)
            pd.DataFrame(perf).T.to_csv(
                "model_performance_summary.csv", index_label="Model"
            )
            logging.info("✅ Model training complete")
        else:
            logging.warning("No target column found for model training")
    except Exception as e:
        logging.warning(f"Model training failed: {e}")

    # Step 2: Run clustering and technology matching
    logging.info("🔗 Step 2: Running clustering & PV technology matching")
    try:
        df_result = main_matching_pipeline(
            clustered_data_path=processed_path,
            output_file=output_path,
            k_range=range(2, 8),  # Reduced range for faster execution
        )

        if df_result is not None:
            logging.info(f"✅ Clustering completed with {len(df_result)} locations")
        else:
            logging.error("❌ Clustering pipeline returned None")
            return None

    except Exception as e:
        logging.error(f"❌ Clustering failed: {e}")
        return None

    # Step 3: Compute cluster summaries
    logging.info("📊 Step 3: Computing cluster summaries")
    try:
        summary_df = compute_cluster_summary(df_result)
        logging.info("✅ Cluster summary computed")
    except Exception as e:
        logging.warning(f"Cluster summary failed: {e}")

    # Step 4: Compute PV potential by cluster and year
    logging.info("📈 Step 4: Computing PV potential by cluster/year")
    try:
        pv_summary = compute_pv_potential_by_cluster_year(df_result)
        logging.info("✅ PV potential summary computed")
    except Exception as e:
        logging.warning(f"PV potential computation failed: {e}")

    # Step 5: Generate uncertainty maps
    logging.info("🗺️ Step 5: Generating prediction uncertainty maps")
    try:
        plot_prediction_uncertainty_with_contours(
            df_result,
            use_hatching=False,
            output_path="results/maps/uncertainty_map.png",
        )
        logging.info("✅ Uncertainty map generated")
    except Exception as e:
        logging.warning(f"Map generation failed: {e}")

    if db_url and df_result is not None:
        from database_utils import write_dataframe

        try:
            write_dataframe(df_result, db_table, db_url=db_url, if_exists="replace")
            logging.info(f"✅ Results written to database table {db_table}")
        except Exception as e:
            logging.warning(f"Failed to write results to DB: {e}")

    logging.info("✅ Pipeline completed successfully")
    return df_result


def run_data_preparation_only():
    """Run only the data preparation step."""
    logging.info("🧼 Running data preparation only")

    input_path = "clustered_dataset.csv"
    output_path = "data/clustered_dataset_enhanced.csv"

    try:
        result = prepare_clustered_dataset(
            input_path=input_path, output_path=output_path
        )
        if result is not None:
            logging.info(f"✅ Data preparation complete: {len(result)} rows")
            return result
        else:
            logging.error("❌ Data preparation failed")
            return None
    except Exception as e:
        logging.error(f"❌ Data preparation error: {e}")
        return None


def run_clustering_only():
    """Run only the clustering step."""
    logging.info("🔗 Running clustering only")

    input_path = "data/clustered_dataset_enhanced.csv"
    if not os.path.exists(input_path):
        input_path = "clustered_dataset.csv"

    output_path = "matched_dataset.csv"

    try:
        result = main_matching_pipeline(
            clustered_data_path=input_path, output_file=output_path, k_range=range(2, 6)
        )
        if result is not None:
            logging.info(f"✅ Clustering complete: {len(result)} locations")
            return result
        else:
            logging.error("❌ Clustering failed")
            return None
    except Exception as e:
        logging.error(f"❌ Clustering error: {e}")
        return None


def main():
    """Main execution function with error handling and options."""

    args = parse_args()
    if not validate_environment(args):
        return
    mode = args.mode

    # Ensure output directories exist
    os.makedirs("results/maps", exist_ok=True)

    logging.info(f"🚀 Starting RC-PV pipeline in '{mode}' mode")

    # Check for required files
    if not check_required_files(args.input_file):
        logging.error("❌ Missing required input files")
        logging.info(
            "Please ensure 'clustered_dataset.csv' exists in your working directory"
        )
        return

    try:
        if mode == "full":
            result = main_rc_pv_pipeline(args.input_file, args.db_url, args.db_table)

        elif mode == "prep":
            # Run only data preparation
            result = run_data_preparation_only()

        elif mode == "cluster":
            # Run only clustering
            result = run_clustering_only()

        else:
            logging.error(f"❌ Unknown mode: {mode}")
            logging.info("Available modes: full, prep, cluster")
            return

        if result is not None:
            logging.info("✅ All requested steps completed successfully")
            logging.info(f"📊 Final dataset has {len(result)} rows")

            # Print summary statistics
            if "Cluster_ID" in result.columns:
                n_clusters = result["Cluster_ID"].nunique()
                logging.info(f"📈 Found {n_clusters} clusters")

            if "Best_Technology" in result.columns:
                tech_counts = result["Best_Technology"].value_counts()
                logging.info("🔧 Technology distribution:")
                for tech, count in tech_counts.items():
                    logging.info(f"   {tech}: {count} locations")
        else:
            logging.error("❌ Pipeline completed but returned no results")

    except KeyboardInterrupt:
        logging.info("⏹️ Pipeline interrupted by user")

    except FileNotFoundError as e:
        logging.error(f"❌ Required file not found: {e}")
        logging.info("Make sure all required input files exist")

    except Exception as e:
        logging.error(f"❌ Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler("pipeline.log"),  # File output
        ],
    )

    # Record start time
    start_time = datetime.now()
    logging.info("=" * 50)
    logging.info("🌞 RC-PV CLUSTERING PIPELINE STARTED")
    logging.info("=" * 50)

    try:
        # Run main function
        main()

    finally:
        # Calculate and log total runtime
        end_time = datetime.now()
        runtime = end_time - start_time
        logging.info("=" * 50)
        logging.info(f"⏱️ Total runtime: {runtime}")
        logging.info("🏁 PIPELINE FINISHED")
        logging.info("=" * 50)
