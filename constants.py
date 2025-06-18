"""Central configuration for physical constants and parameters."""

PV_CONSTANTS = {
    'NOCT': 45,  # Nominal Operating Cell Temperature [°C]
    'PR_ref': 0.80,  # Reference performance ratio
    'Reference_Red_Fraction': 0.42,  # From AM1.5 standard
    'PR_bounds': (0.7, 0.9),  # Performance ratio bounds
    'temperature_coefficients': {
        'Silicon': -0.0045,
        'Perovskite': -0.0025,
        'Tandem': -0.0035,
        'CdTe': -0.0028,
    },
}

PHYSICAL_LIMITS = {
    'GHI': (0, 1500),  # W/m²
    'T_air': (-50, 60),  # °C
    'RC_potential': (-100, 300),  # W/m²
    'Red_band': (0, None),  # W/m²
    'Total_band': (0, None),  # W/m²
}
