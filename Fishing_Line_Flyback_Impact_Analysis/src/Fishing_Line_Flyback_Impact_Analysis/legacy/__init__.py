"""
Legacy Analysis Tools for Fishing Line Flyback Impact Analysis

This module contains the original kinetic energy analysis methods and
supporting tools that have been moved from the main package to maintain
backward compatibility while focusing the primary interface on impulse analysis.

Legacy Components:
✓ Kinetic energy analysis (ImpactAnalyzer class)
✓ Interactive windowing tools
✓ Method comparison utilities
✓ Legacy visualization functions
✓ Original CLI commands

Note: These tools are still functional but are no longer the primary
analysis method. The impulse-based analysis in the main package is
recommended for new analyses.

Usage:
>>> from Fishing_Line_Flyback_Impact_Analysis.legacy import kinetic_energy_analysis
>>> from Fishing_Line_Flyback_Impact_Analysis.legacy import windowing

CLI Access:
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis legacy ke-analysis data/csv
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis legacy window-tool file.csv
"""

# Note: Actual imports would be added here when the legacy modules are created
# For now, this serves as a placeholder structure

__version__ = "3.0.0-legacy"
__description__ = "Legacy kinetic energy analysis tools"

__all__ = [
    # Will be populated when legacy modules are created
    # "kinetic_energy_analysis",
    # "windowing", 
    # "method_comparison",
    # "legacy_visualization"
]

def _show_legacy_info():
    """Show information about available legacy tools."""
    print("🗂️  LEGACY ANALYSIS TOOLS")
    print("=" * 40)
    print("These tools have been moved from the main package")
    print("to maintain compatibility while focusing on impulse analysis.")
    print()
    print("Available legacy modules:")
    print("• kinetic_energy_analysis - Original KE analysis methods")
    print("• windowing - Interactive data windowing tools") 
    print("• method_comparison - Compare KE vs impulse methods")
    print("• legacy_visualization - Original plotting functions")
    print()
    print("CLI access via:")
    print("poetry run python -m Fishing_Line_Flyback_Impact_Analysis legacy --help")

if __name__ == "__main__":
    _show_legacy_info()
