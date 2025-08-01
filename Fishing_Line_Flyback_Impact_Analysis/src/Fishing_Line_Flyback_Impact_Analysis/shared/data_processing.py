"""
Shared data processing functions for Fishing Line Flyback Impact Analysis

This module contains common data processing functions used by both impulse 
and legacy kinetic energy analysis methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Union, Optional

from .constants import LBF_TO_N, CONFIG_WEIGHTS, LINE_MASS_FRACTION, DEFAULT_SAMPLING_RATE


def load_csv_file(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV file with error handling.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with CSV data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise pd.errors.EmptyDataError(f"CSV file is empty: {csv_path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {csv_path}: {e}")


def detect_force_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect force columns in CSV data.
    
    Args:
        df: DataFrame with sensor data
        
    Returns:
        List of column names containing force data
    """
    # Look for columns with 'AI' and 'lbf' in the name (common sensor naming)
    force_columns = [col for col in df.columns if 'AI' in col and 'lbf' in col.lower()]
    
    if not force_columns:
        # Fallback: look for numeric columns that aren't time
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        time_cols = [col for col in numeric_cols if 'time' in col.lower()]
        force_columns = [col for col in numeric_cols if col not in time_cols]
    
    return force_columns


def convert_lbf_to_n(force_lbf: np.ndarray) -> np.ndarray:
    """
    Convert force from pounds-force to Newtons.
    
    Args:
        force_lbf: Force data in lbf
        
    Returns:
        Force data in Newtons
    """
    return force_lbf * LBF_TO_N


def apply_baseline_correction(force: np.ndarray) -> np.ndarray:
    """
    Apply baseline correction to force data.
    
    Args:
        force: Raw force data
        
    Returns:
        Baseline-corrected force data
    """
    if len(force) < 100:
        return force
    
    # Use median of first portion as baseline
    baseline_window = min(1000, len(force) // 10)
    baseline = np.median(force[:baseline_window])
    return force - baseline


def calculate_total_force(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate total force from CSV data by summing all force columns.
    
    Args:
        df: DataFrame with sensor data
        
    Returns:
        Tuple of (total_force_in_N, force_column_names)
    """
    force_columns = detect_force_columns(df)
    
    if not force_columns:
        raise ValueError("No force columns detected in CSV file")
    
    # Sum all force columns (assumed to be in lbf)
    total_force_lbf = df[force_columns].sum(axis=1).values
    
    # Convert to Newtons and apply corrections
    total_force_n = convert_lbf_to_n(total_force_lbf)
    total_force_n = np.nan_to_num(total_force_n, nan=0.0)
    total_force_n = apply_baseline_correction(total_force_n)
    
    return total_force_n, force_columns


def get_time_array(df: pd.DataFrame, force_length: int, 
                   sampling_rate: float = DEFAULT_SAMPLING_RATE) -> Tuple[np.ndarray, float]:
    """
    Extract or generate time array from CSV data.
    
    Args:
        df: DataFrame with sensor data
        force_length: Length of force data array
        sampling_rate: Default sampling rate if no time column found
        
    Returns:
        Tuple of (time_array, actual_sampling_rate)
    """
    # Look for time columns
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    
    if time_cols:
        time_array = df[time_cols[0]].values[:force_length]
        if len(time_array) > 0:
            # Normalize to start at zero
            time_array = time_array - time_array[0]
            
            # Calculate actual sampling rate
            if len(time_array) > 10:
                dt_values = np.diff(time_array)
                actual_dt = np.median(dt_values[dt_values > 0])
                
                if actual_dt > 0:
                    actual_sampling_rate = 1.0 / actual_dt
                else:
                    actual_sampling_rate = sampling_rate
            else:
                actual_sampling_rate = sampling_rate
        else:
            time_array = np.arange(force_length) / sampling_rate
            actual_sampling_rate = sampling_rate
    else:
        # Generate time array based on sampling rate
        time_array = np.arange(force_length) / sampling_rate
        actual_sampling_rate = sampling_rate
    
    return time_array, actual_sampling_rate


def extract_material_code(filename: str) -> str:
    """
    Extract material configuration code from filename.
    
    Args:
        filename: CSV filename
        
    Returns:
        Material code (e.g., 'STND', 'DF', etc.)
    """
    try:
        return Path(filename).stem.split('-')[0]
    except:
        return 'UNKNOWN'


def extract_sample_number(filename: str) -> str:
    """
    Extract sample number from filename.
    
    Args:
        filename: CSV filename
        
    Returns:
        Sample identifier
    """
    try:
        parts = Path(filename).stem.split('-')
        return '-'.join(parts[1:])
    except:
        return 'UNKNOWN'


def get_system_mass(material_code: str, include_line_mass: bool = True,
                   line_mass_fraction: float = LINE_MASS_FRACTION) -> dict:
    """
    Calculate total system mass for given material configuration.
    
    Args:
        material_code: Configuration code (e.g., 'STND', 'DF')
        include_line_mass: Whether to include line mass
        line_mass_fraction: Effective line mass fraction
        
    Returns:
        Dictionary with mass breakdown
    """
    # Get hardware mass
    hardware_mass = CONFIG_WEIGHTS.get(material_code, CONFIG_WEIGHTS['STND'])
    
    # Calculate line mass
    if include_line_mass:
        # Convert measured line mass to kg and apply fraction
        line_mass_total_kg = 0.0388  # Total line mass in kg
        line_mass_effective = line_mass_total_kg * line_mass_fraction
    else:
        line_mass_effective = 0.0
    
    total_mass = hardware_mass + line_mass_effective
    
    return {
        'hardware_mass_kg': hardware_mass,
        'line_mass_effective_kg': line_mass_effective,
        'line_mass_fraction': line_mass_fraction if include_line_mass else 0.0,
        'total_mass_kg': total_mass,
        'material_code': material_code
    }
