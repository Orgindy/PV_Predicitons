import os
import subprocess
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import argparse

# === USER CONFIG (overridden by CLI) ===
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SMARTS batch simulations")
    parser.add_argument(
        "--smarts-exe",
        default="smarts295bat.exe",
        help="Path to SMARTS executable",
    )
    parser.add_argument(
        "--inp-dir",
        default="smarts_inp_files",
        help="Directory containing SMARTS input files",
    )
    parser.add_argument(
        "--out-dir",
        default="smarts_out_files",
        help="Directory for SMARTS output files",
    )
    return parser.parse_args()

smarts_exe = None
inp_dir = None
out_dir = None

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

def retry_failed_runs(input_folder, output_folder, max_retries=3, delay=5):
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
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".inp")]
    
    if not input_files:
        print("❌ No SMARTS input files found.")
        return
    
    for inp_file in input_files:
        base_name = os.path.splitext(inp_file)[0]
        output_file = os.path.join(output_folder, f"{base_name}.out")
        
        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"✅ {output_file} already exists. Skipping.")
            continue
        
        # Attempt to run SMARTS
        attempts = 0
        success = False
        
        while attempts < max_retries and not success:
            try:
                command = f"{smarts_exe} {os.path.join(input_folder, inp_file)}"
                try:
                    subprocess.run(command, check=True, shell=True)
                    if os.path.exists(output_file):
                        print(f"✅ Successfully ran {inp_file} on attempt {attempts+1}")
                        success = True
                    else:
                        raise FileNotFoundError(f"⚠️ Output file not found for {inp_file}")
                except subprocess.CalledProcessError as e:
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


def parallel_process(input_folder, output_folder, max_retries=3, delay=5, processes=None):
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
        processes = cpu_count() - 1  # Leave one core free for system tasks
    
    # Get all input files
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".inp")]
    
    if not input_files:
        print("❌ No SMARTS input files found.")
        return
    
    # Helper function for parallel execution
    def run_smarts(inp_file):
        base_name = os.path.splitext(inp_file)[0]
        output_file = os.path.join(output_folder, f"{base_name}.out")
        
        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"✅ {output_file} already exists. Skipping.")
            return
        
        attempts = 0
        success = False
        
        while attempts < max_retries and not success:
            try:
                command = f"{smarts_exe} {os.path.join(input_folder, inp_file)}"
                subprocess.run(command, check=True, shell=True)
                
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
    
    # Run in parallel
    print(f"🔄 Starting parallel processing with {processes} processes...")
    with Pool(processes=processes) as pool:
        pool.map(run_smarts, input_files)
    
    print("✅ Parallel processing complete.")

def verify_output_files(input_folder, output_folder):
    """
    Verifies that all SMARTS output files are complete and valid.
    
    Parameters:
    - input_folder (str): Directory containing SMARTS input files.
    - output_folder (str): Directory to check for completed SMARTS output files.
    
    Returns:
    - None
    """
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".inp")]
    missing_files = []
    incomplete_files = []
    
    for inp_file in input_files:
        base_name = os.path.splitext(inp_file)[0]
        output_file = os.path.join(output_folder, f"{base_name}.out")
        
        # Check if the output file exists
        if not os.path.exists(output_file):
            missing_files.append(output_file)
            continue
        
        # Check if the output file is complete
        with open(output_file, 'r') as f:
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
        print("\n🚨 Some output files are missing or incomplete. Please check the log above.")

def main():
    global smarts_exe, inp_dir, out_dir
    args = parse_args()
    smarts_exe = args.smarts_exe
    inp_dir = Path(args.inp_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    inp_files = list(inp_dir.glob("*.inp"))
    if not inp_files:
        print("❌ No .inp files found.")
        return

    print(f"🔍 Found {len(inp_files)} .inp files to process.\n")

    # Validate SMARTS executable first
    if not validate_smarts_executable(smarts_exe):
        print("❌ Cannot proceed without valid SMARTS executable")
        return

    parallel_process(input_folder=str(inp_dir), output_folder=str(out_dir))
    verify_output_files(str(inp_dir), str(out_dir))

if __name__ == "__main__":
    main()



