"""Shared components for Fishing Line Flyback Impact Analysis.

This package contains constants, data processing functions, and utilities
shared between impulse analysis and legacy kinetic energy analysis.
"""

from .constants import CONFIG_WEIGHTS
from .constants import DEFAULT_SAMPLING_RATE
from .constants import GRAMS_TO_KG
from .constants import IMPACT_THRESHOLD_FACTOR
from .constants import INCHES_TO_M
from .constants import LBF_TO_N
from .constants import LINE_MASS_FRACTION
from .constants import MATERIAL_NAMES
from .constants import MEASURED_LINE_LENGTH_INCHES
from .constants import MEASURED_LINE_MASS_GRAMS
from .data_processing import apply_baseline_correction
from .data_processing import calculate_total_force
from .data_processing import convert_lbf_to_n
from .data_processing import detect_force_columns
from .data_processing import extract_material_code
from .data_processing import extract_sample_number
from .data_processing import get_system_mass
from .data_processing import get_time_array
from .data_processing import load_csv_file


__all__ = [
    # Constants
    "CONFIG_WEIGHTS",
    "LINE_MASS_FRACTION",
    "MEASURED_LINE_LENGTH_INCHES",
    "MEASURED_LINE_MASS_GRAMS",
    "MATERIAL_NAMES",
    "DEFAULT_SAMPLING_RATE",
    "IMPACT_THRESHOLD_FACTOR",
    "LBF_TO_N",
    "INCHES_TO_M",
    "GRAMS_TO_KG",
    # Data processing functions
    "load_csv_file",
    "detect_force_columns",
    "convert_lbf_to_n",
    "apply_baseline_correction",
    "calculate_total_force",
    "get_time_array",
    "extract_material_code",
    "extract_sample_number",
    "get_system_mass",
]
