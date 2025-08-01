"""
Shared constants and configurations for Fishing Line Flyback Impact Analysis

This module contains constants used by both impulse and legacy kinetic energy analysis.
"""

# Configuration weights for different fishing line configurations (kg)
CONFIG_WEIGHTS = {
    "STND": 0.045,  # Standard configuration
    "DF": 0.060,  # Dual Fixed
    "DS": 0.072,  # Dual Sliding
    "SL": 0.069,  # Sliding
    "BR": 0.045,  # Breakaway
}

# Line mass parameters
LINE_MASS_FRACTION = 0.70  # Effective line mass fraction (literature validated)
MEASURED_LINE_LENGTH_INCHES = 5.5  # Measured line length
MEASURED_LINE_MASS_GRAMS = 0.542  # Measured line mass

# Material name mappings for display
MATERIAL_NAMES = {
    "STND": "Standard",
    "DF": "Dual Fixed",
    "DS": "Dual Sliding",
    "SL": "Sliding",
    "BR": "Breakaway",
}

# Analysis parameters
DEFAULT_SAMPLING_RATE = 100000.0  # Hz
IMPACT_THRESHOLD_FACTOR = 0.02  # Threshold for impact detection

# Unit conversion factors
LBF_TO_N = 4.44822  # Pounds-force to Newtons
INCHES_TO_M = 0.0254  # Inches to meters
GRAMS_TO_KG = 0.001  # Grams to kilograms
