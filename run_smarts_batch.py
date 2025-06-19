import os
import time
import subprocess
from multiprocessing import Pool, cpu_count
import argparse
from pathlib import Path
from dataclasses import dataclass
import psutil
from config import PathConfig


@dataclass
class BatchConfig:
    smarts_exe: str
    inp_dir: Path
    out_dir: Path
    timeout: int = 300
    max_retries: int = 3
    min_disk_space: int = 100 * 1024 * 1024  # 100MB

    def validate(self) -> bool:
        return (
            Path(self.smarts_exe).exists()
            and self.inp_dir.exists()
            and self.check_disk_space()
        )

    def check_disk_space(self) -> bool:
        try:
            return psutil.disk_usage(self.out_dir).free >= self.min_disk_space
        except Exception:
            return False

    def ensure_dirs(self) -> bool:
        return ensure_directory(self.inp_dir) and ensure_directory(self.out_dir)


# === USER CONFIG (overridden by CLI) ===
def parse_args():
    """Return config-based arguments (CLI removed)."""
    cfg = PathConfig.from_yaml(Path("config.yaml"))
    return argparse.Namespace(
        smarts_exe="smarts295bat.exe",
        inp_dir=cfg.smarts_inp_path,
        out_dir=cfg.smarts_out_path,
        timeout=300,
    )


def ensure_directory(path: Path) -> bool:
    """Create directory if it doesn't exist and check write permission."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"❌ Permission denied creating directory: {path}")
        return False
    if not os.access(path, os.W_OK):
        print(f"❌ No write permission for directory: {path}")
        return False
    return True


def validate_smarts_executable(exe_path):
    """Validate that SMARTS executable exists and is accessible"""
    if not os.path.exists(exe_path):
        print(f"❌ SMARTS executable not found: {exe_path}")
        return False

    if not os.access(exe_path, os.X_OK):
        print(f"❌ SMARTS executable is not executable: {exe_path}")
        return False

    print(f"✅ SMARTS executable validated: {exe_path}")
    return True


def run_smarts_process(cfg: BatchConfig, input_file: Path) -> bool:
    """Run a single SMARTS simulation with error handling."""
    command = f"{cfg.smarts_exe} {input_file}"
    try:
        subprocess.run(command, check=True, shell=True, timeout=cfg.timeout)
        logging.info("✅ %s", input_file.name)
        return True
    except Exception as exc:
        logging.warning("SMARTS failed for %s: %s", input_file.name, exc)
        with open("smarts_batch_errors.log", "a") as log:
            log.write(f"{input_file}\n{exc}\n")
        return False


def retry_failed_runs(cfg: BatchConfig, max_retries=3, delay=5):
    """
    Retries failed SMARTS simulations up to a maximum number of attempts.

    Parameters:
    - input_folder (str): Directory containing SMARTS input files.
    - output_folder (str): Directory to save SMARTS output files.
    - max_retries (int): Maximum number of retry attempts (default 3).
    - delay (int): Delay (in seconds) between retry attempts (default 5).

    Returns:
    - None
    """
    input_files = [f for f in os.listdir(cfg.inp_dir) if f.endswith(".inp")]

    if not input_files:
        print("❌ No SMARTS input files found.")
        return

    for inp_file in input_files:
        base_name = os.path.splitext(inp_file)[0]
        output_file = cfg.out_dir / f"{base_name}.out"

        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"✅ {output_file} already exists. Skipping.")
            continue

        # Attempt to run SMARTS
        attempts = 0
        success = False

        while attempts < max_retries and not success:
            try:
                command = f"{cfg.smarts_exe} {os.path.join(cfg.inp_dir, inp_file)}"
                try:
                    subprocess.run(command, check=True, shell=True, timeout=cfg.timeout)
                    if os.path.exists(output_file):
                        print(f"✅ Successfully ran {inp_file} on attempt {attempts+1}")
                        success = True
                    else:
                        raise FileNotFoundError(
                            f"⚠️ Output file not found for {inp_file}"
                        )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"❌ SMARTS execution failed for {inp_file}: {e}")
                    with open("smarts_batch_errors.log", "a") as log:
                        log.write(f"[Execution Error] {inp_file}\n{command}\n{e}\n\n")
                except Exception as e:
                    print(f"❌ Unexpected error for {inp_file}: {e}")
                    with open("smarts_batch_errors.log", "a") as log:
                        log.write(f"[Unknown Error] {inp_file}\n{command}\n{e}\n\n")

                # Verify output file
                if os.path.exists(output_file):
                    print(f"✅ Successfully ran {inp_file} on attempt {attempts+1}")
                    success = True
                else:
                    raise FileNotFoundError(f"Output file not found for {inp_file}")

            except Exception as e:
                attempts += 1
                print(f"⚠️ Attempt {attempts} failed for {inp_file}: {e}")
                time.sleep(delay)

        if not success:
            print(f"❌ Failed to run {inp_file} after {max_retries} attempts.")
            if output_file.exists():
                output_file.unlink(missing_ok=True)


def parallel_process(cfg: BatchConfig, max_retries=3, delay=5, processes=None):
    """
    Runs SMARTS simulations in parallel for faster processing.

    Parameters:
    - input_folder (str): Directory containing SMARTS input files.
    - output_folder (str): Directory to save SMARTS output files.
    - max_retries (int): Maximum number of retry attempts (default 3).
    - delay (int): Delay (in seconds) between retry attempts (default 5).
    - processes (int): Number of parallel processes to use (default: number of CPU cores).

    Returns:
    - None
    """
    # Set number of processes to use
    if processes is None:
        processes = max(1, cpu_count() - 1)  # Leave one core free for system tasks

    # Get all input files
    input_files = [f for f in os.listdir(cfg.inp_dir) if f.endswith(".inp")]

    if not input_files:
        print("❌ No SMARTS input files found.")
        return

    # Helper function for parallel execution
    total = len(input_files)

    def run_smarts(idx, inp_file):
        base_name = os.path.splitext(inp_file)[0]
        output_file = cfg.out_dir / f"{base_name}.out"

        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"✅ {output_file} already exists. Skipping.")
            return

        attempts = 0
        success = False

        while attempts < max_retries and not success:
            try:
                command = f"{cfg.smarts_exe} {os.path.join(cfg.inp_dir, inp_file)}"
                subprocess.run(command, check=True, shell=True, timeout=cfg.timeout)

                # Verify output file
                if os.path.exists(output_file):
                    print(
                        f"[{idx}/{total}] ✅ Successfully ran {inp_file} on attempt {attempts+1}"
                    )
                    success = True
                else:
                    raise FileNotFoundError(f"Output file not found for {inp_file}")

            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                Exception,
            ) as e:
                attempts += 1
                print(
                    f"[{idx}/{total}] ⚠️ Attempt {attempts} failed for {inp_file}: {e}"
                )
                time.sleep(delay)

        if not success:
            print(
                f"[{idx}/{total}] ❌ Failed to run {inp_file} after {max_retries} attempts."
            )
            if output_file.exists():
                output_file.unlink(missing_ok=True)

    # Run in parallel
    print(f"🔄 Starting parallel processing with {processes} processes...")
    with Pool(processes=processes) as pool:
        pool.starmap(run_smarts, [(i, f) for i, f in enumerate(input_files, 1)])

    print("✅ Parallel processing complete.")


def verify_output_files(cfg: BatchConfig):
    """
    Verifies that all SMARTS output files are complete and valid.

    Parameters:
    - input_folder (str): Directory containing SMARTS input files.
    - output_folder (str): Directory to check for completed SMARTS output files.

    Returns:
    - None
    """
    input_files = [f for f in os.listdir(cfg.inp_dir) if f.endswith(".inp")]
    missing_files = []
    incomplete_files = []

    for inp_file in input_files:
        base_name = os.path.splitext(inp_file)[0]
        output_file = cfg.out_dir / f"{base_name}.out"

        # Check if the output file exists
        if not os.path.exists(output_file):
            missing_files.append(output_file)
            continue

        # Check if the output file is complete
        with open(output_file, "r") as f:
            lines = f.readlines()
            if not lines or "Program terminated normally" not in lines[-1]:
                incomplete_files.append(output_file)

    # Print results
    if missing_files:
        print("\n❌ Missing Output Files:")
        for file in missing_files:
            print(f"  - {file}")

    if incomplete_files:
        print("\n⚠️ Incomplete Output Files:")
        for file in incomplete_files:
            print(f"  - {file}")

    if not missing_files and not incomplete_files:
        print("\n✅ All output files are complete and valid.")
    else:
        print(
            "\n🚨 Some output files are missing or incomplete. Please check the log above."
        )


def main():
    args = parse_args()
    cfg = BatchConfig(
        smarts_exe=args.smarts_exe,
        inp_dir=Path(args.inp_dir),
        out_dir=Path(args.out_dir),
        timeout=args.timeout,
    )

    if not ensure_directory(cfg.inp_dir) or not ensure_directory(cfg.out_dir):
        return

    inp_files = list(cfg.inp_dir.glob("*.inp"))
    if not inp_files:
        print("❌ No .inp files found.")
        return

    print(f"🔍 Found {len(inp_files)} .inp files to process.\n")

    # Validate SMARTS executable first
    if not validate_smarts_executable(cfg.smarts_exe):
        print("❌ Cannot proceed without valid SMARTS executable")
        return

    parallel_process(cfg)
    verify_output_files(cfg)


if __name__ == "__main__":
    main()
