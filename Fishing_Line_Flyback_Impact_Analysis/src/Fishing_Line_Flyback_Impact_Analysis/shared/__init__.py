"""
Shared components for Fishing Line Flyback Impact Analysis

This package contains constants, data processing functions, and utilities
shared between impulse analysis and legacy kinetic energy analysis.
"""

from .constants import (
    CONFIG_WEIGHTS,
    LINE_MASS_FRACTION,
    MEASURED_LINE_LENGTH_INCHES,
    MEASURED_LINE_MASS_GRAMS,
    MATERIAL_NAMES,
    DEFAULT_SAMPLING_RATE,
    IMPACT_THRESHOLD_FACTOR,
    LBF_TO_N,
    INCHES_TO_M,
    GRAMS_TO_KG
)

from .data_processing import (
    load_csv_file,
    detect_force_columns,
    convert_lbf_to_n,
    apply_baseline_correction,
    calculate_total_force,
    get_time_array,
    extract_material_code,
    extract_sample_number,
    get_system_mass
)

__all__ = [
    # Constants
    'CONFIG_WEIGHTS',
    'LINE_MASS_FRACTION', 
    'MEASURED_LINE_LENGTH_INCHES',
    'MEASURED_LINE_MASS_GRAMS',
    'MATERIAL_NAMES',
    'DEFAULT_SAMPLING_RATE',
    'IMPACT_THRESHOLD_FACTOR',
    'LBF_TO_N',
    'INCHES_TO_M',
    'GRAMS_TO_KG',
    
    # Data processing functions
    'load_csv_file',
    'detect_force_columns',
    'convert_lbf_to_n',
    'apply_baseline_correction',
    'calculate_total_force', 
    'get_time_array',
    'extract_material_code',
    'extract_sample_number',
    'get_system_mass'
]
