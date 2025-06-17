import numpy as np
import os

def load_scenario_inputs(folder_path, scenario, year):
    """
    Load climate scenario input arrays for a specific model and year.

    Parameters:
        folder_path (str): Path to the base folder with input arrays
        scenario (str): Scenario label (e.g., 'ssp245', 'ERA5')
        year (int): Target year (e.g., 2023, 2050, 2100)

    Returns:
        dict: {
            'ghi': np.ndarray,
            'tair': np.ndarray,
            'ir': np.ndarray,
            'wind': np.ndarray,
            'zenith': np.ndarray
        }
    """

    base = os.path.join(folder_path, f"{scenario}_{year}")
    print(f"📂 Loading scenario data from: {base}")

    return {
        "ghi":    np.load(os.path.join(base, "ghi.npy")),
        "tair":   np.load(os.path.join(base, "tair.npy")),
        "ir":     np.load(os.path.join(base, "ir_down.npy")),
        "wind":   np.load(os.path.join(base, "wind.npy")),
        "zenith": np.load(os.path.join(base, "zenith.npy"))
    }
