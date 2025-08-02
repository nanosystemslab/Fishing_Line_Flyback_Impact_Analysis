"""Fishing Line Flyback Impact Analysis Package v3.0.

IMPULSE-FOCUSED VERSION: Streamlined for momentum transfer analysis.

This package provides tools for analyzing fishing line flyback impact properties
using impulse-based analysis (∫ F(t) dt), which directly measures momentum transfer
and is more relevant to fishing line performance than kinetic energy estimation.

Core Features:
✓ Impulse-based analysis (∫ F(t) dt) - PRIMARY METHOD
✓ Direct momentum transfer measurement
✓ Force curve visualization with boundary validation
✓ Configuration-specific hardware weights
✓ Measured line mass integration (70% effective)
✓ Material comparison and statistical reporting
✓ Publication-quality visualizations
✓ Shared data processing components

Legacy Features (available via .legacy module):
✓ Kinetic energy analysis methods
✓ Interactive windowing tools
✓ Method comparison utilities

Scientific Basis:
- Hardware masses: Measured for each configuration (STND=45g, DF=60g, etc.)
- Line mass: 5.5" = 0.542g measured → 38.8g total → 27g effective (70%)
- Analysis method: Direct impulse integration ∫ F(t) dt
- Focus: What the fish/lure actually experiences (momentum transfer)

Quick Start:
>>> from Fishing_Line_Flyback_Impact_Analysis import run_impulse_analysis
>>> results = run_impulse_analysis('data/csv', 'output')

CLI Usage:
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv
"""

# Core impulse analysis (primary interface)
from .impulse_analysis import ImpulseAnalyzer
from .impulse_analysis import analyze_single_file_with_impulse
from .impulse_analysis import create_impulse_boxplots
from .impulse_analysis import run_impulse_analysis

# Shared constants and utilities
from .shared import CONFIG_WEIGHTS
from .shared import LINE_MASS_FRACTION
from .shared import MATERIAL_NAMES
from .shared import MEASURED_LINE_LENGTH_INCHES
from .shared import MEASURED_LINE_MASS_GRAMS
from .shared import calculate_total_force
from .shared import extract_material_code
from .shared import get_system_mass
from .shared import get_time_array
from .shared import load_csv_file

# Essential visualization functions
from .visualization import plot_single_file_analysis
from .visualization import show_force_preview


# Import GUI components (with fallback for missing dependencies)
try:
    from .gui import FishingLineAnalyzerGUI
    from .gui_launcher import launch_gui

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False
    FishingLineAnalyzerGUI = None
    launch_gui = None

# Package metadata
__version__ = "3.0.0"  # Major version bump - impulse-focused architecture
__author__ = "Nanosystems Lab"
__description__ = "Fishing Line Flyback Impact Analysis - Impulse-Focused v3.0"

# Main public API (impulse-focused)
__all__ = [
    # Primary analysis functions
    "ImpulseAnalyzer",
    "analyze_single_file_with_impulse",
    "run_impulse_analysis",
    # Visualization functions
    "create_impulse_boxplots",
    "plot_single_file_analysis",
    "show_force_preview",
    # Data processing utilities
    "load_csv_file",
    "calculate_total_force",
    "get_time_array",
    "extract_material_code",
    "get_system_mass",
    # Constants and configuration
    "CONFIG_WEIGHTS",
    "LINE_MASS_FRACTION",
    "MEASURED_LINE_LENGTH_INCHES",
    "MEASURED_LINE_MASS_GRAMS",
    "MATERIAL_NAMES",
    # Package info
    "__version__",
    "__author__",
    "__description__"
    # GUI components (if available)
    "FishingLineAnalyzerGUI",
    "launch_gui",
]

# Set up plotting style for the package
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Configure matplotlib for consistent plots
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

except ImportError:
    # Plotting libraries not available - continue without styling
    pass


# Package-level convenience functions
def quick_analysis(file_path, show_plot=False):
    """Quick analysis of a single file with sensible defaults.

    Args:
        file_path: Path to CSV file
        show_plot: Whether to show validation plot

    Returns:
        Analysis results dictionary

    Example:
        >>> result = quick_analysis('data/STND-21-5.csv', show_plot=True)
        >>> print(f"Impulse: {result['total_impulse']:.6f} N⋅s")
    """
    return analyze_single_file_with_impulse(file_path, show_plot=show_plot)


def batch_analysis(data_directory, output_directory=None):
    """Batch analysis of all CSV files in a directory.

    Args:
        data_directory: Directory containing CSV files
        output_directory: Output directory (default: 'impulse_analysis')

    Returns:
        List of analysis results

    Example:
        >>> results = batch_analysis('data/csv')
        >>> valid_results = [r for r in results if 'error' not in r]
        >>> print(f"Analyzed {len(valid_results)} files successfully")
    """
    if output_directory is None:
        output_directory = "impulse_analysis"

    return run_impulse_analysis(data_directory, output_directory)


def get_configuration_info():
    """Get information about hardware configurations and mass calculations.

    Returns:
        Dictionary with configuration details

    Example:
        >>> info = get_configuration_info()
        >>> for config, details in info['configurations'].items():
        ...     print(f"{config}: {details['total_mass_kg']*1000:.0f}g total")
    """
    configurations = {}

    for config, hardware_mass in CONFIG_WEIGHTS.items():
        mass_info = get_system_mass(config, include_line_mass=True)
        configurations[config] = {
            "name": MATERIAL_NAMES.get(config, config),
            "hardware_mass_kg": hardware_mass,
            "line_mass_effective_kg": mass_info["line_mass_effective_kg"],
            "total_mass_kg": mass_info["total_mass_kg"],
            "description": f"{MATERIAL_NAMES.get(config, config)} configuration",
        }

    return {
        "configurations": configurations,
        "line_mass_fraction": LINE_MASS_FRACTION,
        "measured_line_length_inches": MEASURED_LINE_LENGTH_INCHES,
        "measured_line_mass_grams": MEASURED_LINE_MASS_GRAMS,
        "version": __version__,
        "analysis_method": "impulse_integration",
    }


# Add convenience functions to __all__
__all__.extend(["quick_analysis", "batch_analysis", "get_configuration_info"])
