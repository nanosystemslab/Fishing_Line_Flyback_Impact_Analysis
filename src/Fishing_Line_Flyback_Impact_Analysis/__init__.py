"""
Fishing Line Flyback Impact Analysis Package

CORRECTED VERSION v2.0: Fixed energy overestimation from rebound inclusion.

This package provides tools for analyzing fishing line flyback impact properties
from force sensor data, with corrected physics calculations that produce
realistic results by isolating the deceleration phase.

Key Features:
✓ Corrected energy calculation (excludes rebound phase)
✓ Automatic force unit conversion (lbf → N)
✓ AI sensor data handling (sums multiple channels)
✓ Baseline correction and noise filtering
✓ Material comparison and visualization
✓ Overestimation factor validation
✓ NEW: Impulse-based analysis (∫ F(t) dt)

Major Corrections in v2.0:
- Energy calculation: Isolates deceleration phase only (excludes rebound)
- Physics correction: Prevents 2-10x energy overestimation
- Methodology: Uses cumulative impulse to find velocity=0 turning point
- Validation: Provides overestimation factor comparison with legacy method
- Enhanced analysis: Better material differentiation and realistic energy values
- Impulse analysis: Direct momentum transfer measurement
"""

# Set global plotting style
import matplotlib.pyplot as plt

# Import required modules
import numpy as np
import pandas as pd
import seaborn as sns

# Core analysis and visualization classes
from .analysis import ImpactAnalyzer
from .analysis import analyze_single_file_with_config
from .analysis import run_comprehensive_analysis
from .impulse_analysis import ImpulseAnalyzer
from .impulse_analysis import analyze_single_file_with_impulse
from .impulse_analysis import run_impulse_analysis
from .visualization import ImpactVisualizer
from .visualization import create_summary_plots
from .visualization import plot_single_file_analysis


__version__ = "2.1.0"  # Updated to include impulse analysis
__author__ = "Nanosystems Lab"
__description__ = "Fishing Line Flyback Impact Analysis with Corrected Energy Calculation and Impulse Analysis"

__all__ = [
    # Main classes
    "ImpactAnalyzer",
    "ImpactVisualizer",
    "ImpulseAnalyzer",
    # Convenience functions - Kinetic Energy Analysis
    "analyze_single_file_with_config",
    "run_comprehensive_analysis",
    "create_summary_plots",
    "plot_single_file_analysis",
    # Convenience functions - Impulse Analysis
    "analyze_single_file_with_impulse",
    "run_impulse_analysis",
    # Package info
    "__version__",
    "__author__",
    "__description__",
]

# Set global plotting style
try:
    sns.set_style("whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    # Fallback if seaborn style not available
    plt.style.use("default")

# Package-level constants for validation (now with numpy imported)
REALISTIC_ENERGY_RANGES = {
    "very_low_J": (1e-6, 1e-3),  # μJ to mJ - typical for small impacts
    "low_J": (1e-3, 1e-1),  # mJ to 100 mJ - expected range
    "moderate_J": (1e-1, 1e1),  # 100 mJ to 10 J - higher energy
    "high_J": (1e1, 1e3),  # 10 J to 1 kJ - very high
    "extreme_J": (1e3, np.inf),  # > 1 kJ - likely overestimated
}

# Configuration masses in kg for different setups
CONFIG_MASSES_KG = {
    "STND": 0.049,  # Standard configuration (49g)
    "DF": 0.062,  # Dual Fixed (62g)
    "DS": 0.075,  # Dual Sliding (75g)
    "SL": 0.065,  # Sliding (65g)
    "BR": 0.045,  # Breakaway (45g)
}


def get_material_mass(material_code: str) -> float:
    """
    Get the appropriate mass for a material configuration.

    Args:
        material_code: Material code (e.g., 'BR', 'STND', etc.)

    Returns:
        Mass in kg
    """
    return CONFIG_MASSES_KG.get(material_code.upper(), 0.045)  # Default to 45g


def validate_energy_realistic(energy_J: float) -> tuple[bool, str]:
    """
    Validate if calculated energy is in realistic range.

    Args:
        energy_J: Calculated energy in Joules

    Returns:
        Tuple of (is_realistic, description)
    """
    if energy_J < REALISTIC_ENERGY_RANGES["very_low_J"][0]:
        return False, "Too low (< 1 μJ)"
    elif energy_J < REALISTIC_ENERGY_RANGES["low_J"][1]:
        return True, "Realistic range (< 100 mJ)"
    elif energy_J < REALISTIC_ENERGY_RANGES["moderate_J"][1]:
        return True, "Moderate range (100 mJ - 10 J)"
    elif energy_J < REALISTIC_ENERGY_RANGES["high_J"][1]:
        return False, "High range (10 J - 1 kJ) - check calculation"
    else:
        return False, "Extreme range (> 1 kJ) - likely overestimated"


def print_package_info():
    """Print package information and methodology overview."""
    print("\n" + "=" * 70)
    print("FISHING LINE FLYBACK IMPACT ANALYSIS v2.1")
    print("=" * 70)
    print("ANALYSIS METHODS:")
    print("1. KINETIC ENERGY ANALYSIS:")
    print("   ✓ Isolates deceleration phase only (excludes rebound)")
    print("   ✓ Uses cumulative impulse to find velocity = 0 point")
    print("   ✓ Converts force units: lbf → N (×4.448)")
    print("   ✓ Applies baseline correction for DC offset")
    print("   ✓ Calculates: E = J_decel² / (2m)")
    print("")
    print("2. IMPULSE ANALYSIS (NEW):")
    print("   ✓ Total momentum transfer: ∫ F(t) dt")
    print("   ✓ Direct measurement of impact effectiveness")
    print("   ✓ Simple integration of complete force curve")
    print("   ✓ More relevant to fishing line performance")
    print("")
    print("IMPROVEMENTS OVER LEGACY METHOD:")
    print("• Eliminates 2-10x energy overestimation")
    print("• Provides physically realistic energy values")
    print("• Better material differentiation")
    print("• Includes overestimation factor validation")
    print("• Direct momentum transfer measurement")
    print("")
    print("USAGE EXAMPLES:")
    print("  from Fishing_Line_Flyback_Impact_Analysis import *")
    print("  ")
    print("  # Kinetic Energy Analysis")
    print('  result = analyze_single_file_with_config("data/csv/BR-21-1.csv")')
    print("  print(f\"Energy: {result['kinetic_energy']:.6f} J\")")
    print("  ")
    print("  # Impulse Analysis (RECOMMENDED)")
    print('  result = analyze_single_file_with_impulse("data/csv/BR-21-1.csv")')
    print("  print(f\"Impulse: {result['total_impulse']:+.6f} N⋅s\")")
    print("  ")
    print("  # Batch analysis")
    print('  ke_results = run_comprehensive_analysis("data/csv", "ke_output")')
    print('  impulse_results = run_impulse_analysis("data/csv", "impulse_output")')
    print("  ")
    print("  # Create visualizations")
    print("  create_summary_plots(ke_results, 'ke_plots')")
    print("=" * 70 + "\n")


# Auto-print package info when imported (can be disabled)
import os


if os.environ.get("SHOW_PACKAGE_INFO", "false").lower() == "true":
    print_package_info()

# Methodology validation info
METHODOLOGY_INFO = {
    "version": "2.1.0",
    "methods": ["kinetic_energy", "impulse_analysis"],
    "kinetic_energy_method": "Deceleration-phase isolation",
    "impulse_method": "Total momentum transfer integration",
    "validation": "Overestimation factor comparison",
    "units": "lbf → N conversion, energy in Joules, impulse in N⋅s",
    "improvements": [
        "Excludes rebound phase from energy calculation",
        "Uses cumulative impulse maximum to find v=0",
        "Applies baseline correction for DC offset",
        "Provides realistic energy values (typically 0.001-1 J)",
        "Better material property differentiation",
        "Direct momentum transfer measurement via impulse",
        "Simple and robust impulse integration",
    ],
}
