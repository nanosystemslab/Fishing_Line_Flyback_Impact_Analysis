"""Fishing Line Flyback Impact Analysis - Impulse-Focused CLI v3.0.

Streamlined command-line interface focused on impulse analysis with
optional access to legacy kinetic energy analysis tools.
"""

from pathlib import Path

import click
import numpy as np

# Import core impulse analysis functions
from .impulse_analysis import analyze_single_file_with_impulse
from .impulse_analysis import create_impulse_boxplots
from .impulse_analysis import run_impulse_analysis

# Import shared constants
from .shared import CONFIG_WEIGHTS
from .shared import LINE_MASS_FRACTION
from .shared import MATERIAL_NAMES
from .shared import MEASURED_LINE_LENGTH_INCHES
from .shared import MEASURED_LINE_MASS_GRAMS


@click.group()
@click.version_option(version="1.0.0")
def main():
    """🎯 Fishing Line Flyback Impact Analysis v3.0 - Impulse-Focused.

    This version focuses on impulse analysis (∫ F(t) dt) as the primary method
    for measuring fishing line impact effectiveness through momentum transfer.

    Core Features:
    ✓ Impulse-based analysis (∫ F(t) dt)
    ✓ Direct momentum transfer measurement
    ✓ Force curve visualization with validation plots
    ✓ Configuration-specific weights and measured line mass
    ✓ Material comparison and statistical reporting
    ✓ Publication-quality box plots and visualizations

    Configuration Reference:
    • STND: Standard (45g) • DF: Dual Fixed (60g) • DS: Dual Sliding (72g)
    • SL: Sliding (69g) • BR: Breakaway (45g)

    Quick Start:
    poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv
    """
    pass


# =============================================================================
# CORE IMPULSE ANALYSIS COMMANDS
# =============================================================================


@main.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default="impulse_analysis",
    help="Output directory for analysis results",
)
@click.option(
    "--create-plots", is_flag=True, default=True, help="Create summary box plots"
)
def analyze_impulse(data_dir, output_dir, create_plots):
    """🎯 Run impulse analysis on all CSV files (PRIMARY COMMAND).

    This is the main analysis command that processes all measurement files
    in the specified directory using the impulse method (∫ F(t) dt).

    Results include:
    • Detailed CSV with all impulse metrics
    • Statistical summary by material type
    • Publication-quality box plots (optional)
    • Full results JSON for debugging

    DATA_DIR: Directory containing CSV measurement files
    """
    click.echo("🎯 IMPULSE-BASED FISHING LINE ANALYSIS v3.0")
    click.echo("=" * 70)
    click.echo(f"📁 Data directory: {data_dir}")
    click.echo(f"📊 Output directory: {output_dir}")
    click.echo("🧮 Method: Total momentum transfer via ∫ F(t) dt")
    click.echo()

    # Show configuration reference
    click.echo("⚖️  CONFIGURATION REFERENCE:")
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        line_mass_effective = 0.0388 * LINE_MASS_FRACTION
        total_mass = weight_kg + line_mass_effective
        material_name = MATERIAL_NAMES.get(config, config)
        click.echo(
            f"   {config}: {material_name:<12} "
            f"({weight_kg * 1000:2.0f}g + {line_mass_effective * 1000:.0f}g = {total_mass * 1000:.0f}g total)"  # noqa: B950
        )
    click.echo()

    try:
        # Run the analysis
        results = run_impulse_analysis(data_dir, output_dir)

        if results:
            valid_results = [r for r in results if "error" not in r]

            click.echo()
            click.echo("🎉 ANALYSIS COMPLETE!")
            click.echo(
                f"✅ Successfully analyzed: {len(valid_results)}/{len(results)} files"
            )

            if valid_results:
                impulses = [abs(r["total_impulse"]) for r in valid_results]
                click.echo(
                    f"📊 Impulse range: {min(impulses):.6f} to {max(impulses):.6f} N*s"
                )

                # Create plots if requested
                if create_plots:
                    click.echo("🎨 Creating visualization plots...")
                    output_path = Path(output_dir)
                    create_impulse_boxplots(valid_results, output_path, "SI")
                    create_impulse_boxplots(valid_results, output_path, "mixed")

            click.echo(f"📁 All results saved to: {output_dir}")
        else:
            click.echo("❌ No files were successfully analyzed")

    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--material",
    "-m",
    type=click.Choice(["STND", "DF", "DS", "SL", "BR"]),
    help="Material configuration code (auto-detected if not specified)",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for results (optional)"
)
@click.option("--show-plot", is_flag=True, help="Show boundary validation plot")
@click.option("--debug", is_flag=True, help="Show detailed debug information")
def analyze_single(file_path, material, output, show_plot, debug):
    """🔍 Analyze a single measurement file using impulse method.

    This command analyzes one CSV file and optionally displays a validation
    plot showing the integration boundaries and force curve details.

    Use --show-plot to visualize how the analysis identifies the impact
    region and validates the integration boundaries.

    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"🔍 Single File Impulse Analysis: {file_path.name}")
    click.echo("-" * 50)

    try:
        # Auto-detect material if not specified
        if material is None:
            material = file_path.name.split("-")[0]
            click.echo(f"🔍 Auto-detected material: {material}")

        # Run analysis
        result = analyze_single_file_with_impulse(
            file_path,
            material_code=material,
            include_line_mass=True,
            line_mass_fraction=LINE_MASS_FRACTION,
            show_plot=show_plot,
        )

        if "error" in result:
            click.echo(f"❌ Error: {result['error']}")
            return

        # Display key results
        impulse = result["total_impulse"]
        abs_impulse = result["total_abs_impulse"]
        peak_force = result["peak_force"]
        duration_ms = result["impact_duration"] * 1000

        click.echo("📊 IMPULSE ANALYSIS RESULTS:")
        click.echo(f"   Total impulse: {impulse:+.6f} N*s")
        click.echo(f"   Absolute impulse: {abs_impulse:.6f} N*s")
        click.echo(f"   Peak force: {peak_force:.0f} N")
        click.echo(f"   Impact duration: {duration_ms:.1f} ms")

        # Show equivalent metrics for reference
        if not np.isnan(result.get("equivalent_velocity", np.nan)):
            equiv_vel = result["equivalent_velocity"]
            equiv_ke = result["equivalent_kinetic_energy"]
            click.echo(f"   Equivalent velocity: {equiv_vel:.0f} m/s")
            click.echo(f"   Equivalent KE: {equiv_ke:.6f} J")

        # Debug information
        if debug:
            click.echo("\n🔧 DEBUG INFORMATION:")
            click.echo(
                f"   Material: {material} ({MATERIAL_NAMES.get(material, 'Unknown')})"
            )
            click.echo(f"   Mass breakdown: {result.get('mass_breakdown', 'N/A')}")
            click.echo(
                f"   Impact boundaries: {result['impact_start_idx']} to {result['impact_end_idx']}"  # noqa: B950
            )
            click.echo(
                f"   Sampling rate: {result.get('sampling_rate_hz', 'Unknown'):.0f} Hz"
            )
            click.echo(f"   Data points: {result.get('data_points', 'Unknown')}")

        # Save results if requested
        if output:
            import json

            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\n💾 Results saved to: {output_path}")

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if debug:
            import traceback

            click.echo("\n🔧 Full traceback:")
            click.echo(traceback.format_exc())
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--material",
    "-m",
    type=click.Choice(["STND", "DF", "DS", "SL", "BR"]),
    help="Material configuration code",
)
@click.option(
    "--interactive",
    is_flag=True,
    default=True,
    help="Use interactive plotting (default)",
)
@click.option(
    "--style",
    type=click.Choice(["explorer", "simple"]),
    default="explorer",
    help="Interaction style: explorer (zoom/pan), simple (static)",
)
def plot_file(file_path, material, interactive, style):
    """📊 Plot force curve for visual inspection and exploration.

    This command creates an interactive plot of the force vs time data
    for visual inspection, allowing you to examine the force curve
    characteristics before or after analysis.

    Interaction Styles:
    • explorer: Mouse wheel zoom, click+drag selection, real-time stats
    • simple: Static plot with zoom views

    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"📊 Force Curve Visualization: {file_path.name}")

    try:
        # Import visualization functions
        from .shared.data_processing import calculate_total_force
        from .shared.data_processing import get_time_array
        from .shared.data_processing import load_csv_file

        # Load and process data
        df = load_csv_file(file_path)
        force_data, force_columns = calculate_total_force(df)
        time_data, sampling_rate = get_time_array(df, len(force_data))

        # Auto-detect material if not specified
        if material is None:
            material = file_path.name.split("-")[0]

        click.echo(
            f"📈 Data loaded: {len(force_data)} points, {time_data[-1]:.4f}s duration"
        )
        click.echo(
            f"🔍 Force range: {np.min(force_data):.1f} to {np.max(force_data):.1f} N"
        )
        click.echo(f"📡 Force columns: {', '.join(force_columns)}")

        if interactive and style == "explorer":
            click.echo(f"🎛️ Creating interactive plot (style: {style})...")
            click.echo(
                "💡 Controls: Mouse wheel=zoom, Click+drag=select region, Reset button=restore"  # noqa: B950
            )

            from .visualization import show_force_preview_interactive

            show_force_preview_interactive(
                force_data, time_data, file_path.name, style=style
            )
        else:
            click.echo("🎨 Creating static plot...")
            from .visualization import show_force_preview

            show_force_preview(force_data, time_data, file_path.name)
            click.echo("📊 Static plot displayed")

        click.echo("✅ Plot complete")

    except Exception as e:
        click.echo(f"❌ Error creating plot: {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--material",
    "-m",
    type=click.Choice(["STND", "DF", "DS", "SL", "BR"]),
    help="Material configuration code",
)
@click.option("--show-analysis", is_flag=True, help="Show analysis boundaries overlay")
@click.option(
    "--style",
    type=click.Choice(["explorer"]),
    default="explorer",
    help="Interaction style",
)
def interactive_plot(file_path, material, show_analysis, style):
    """🎛️ Interactive force curve exploration with zoom and analysis preview.

    This command creates a fully interactive plot with:

    EXPLORER STYLE:
    • Mouse wheel: Zoom in/out centered on cursor
    • Click+drag on main plot: Select region to zoom into
    • Reset button: Return to full view
    • Real-time statistics for current view
    • Optional analysis boundary overlay

    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"🎛️ Interactive Force Analysis: {file_path.name}")
    click.echo("-" * 50)

    try:
        # Import visualization functions
        from .shared.data_processing import calculate_total_force
        from .shared.data_processing import extract_material_code
        from .shared.data_processing import get_time_array
        from .shared.data_processing import load_csv_file
        from .visualization import show_force_preview_interactive

        # Load and process data
        df = load_csv_file(file_path)
        force_data, force_columns = calculate_total_force(df)
        time_data, sampling_rate = get_time_array(df, len(force_data))

        # Auto-detect material if not specified
        if material is None:
            material = extract_material_code(file_path.name)
            click.echo(f"🔍 Auto-detected material: {material}")

        # Show data info
        click.echo(f"📈 Data loaded: {len(force_data):,} points")
        click.echo(
            f"⏱️  Duration: {time_data[-1]:.4f}s ({time_data[-1] * 1000:.1f} ms)"
        )  # noqa: B950
        click.echo(
            f"🔍 Force range: {np.min(force_data):.1f} to {np.max(force_data):.1f} N"
        )
        click.echo(f"📊 Peak |Force|: {np.max(np.abs(force_data)):.1f} N")
        click.echo(f"📡 Force columns: {', '.join(force_columns)}")
        click.echo(f"🎚️  Sampling rate: {sampling_rate:.0f} Hz")
        click.echo()

        # Show interactive plot with optional analysis
        click.echo(f"🎨 Creating interactive plot (style: {style})...")
        if show_analysis:
            click.echo("🔬 Including analysis boundary preview")

        click.echo("💡 EXPLORER CONTROLS:")
        click.echo("   • Mouse wheel: Zoom in/out")
        click.echo("   • Click+drag: Select region to zoom")
        click.echo("   • Reset button: Restore full view")
        click.echo("   • Statistics update in real-time")

        click.echo("📝 Close plot window when done exploring")
        click.echo()

        show_force_preview_interactive(
            force_data,
            time_data,
            file_path.name,
            show_analysis_preview=show_analysis,
            style=style,
        )

        click.echo("✅ Interactive exploration complete")

    except Exception as e:
        click.echo(f"❌ Error creating interactive plot: {e}")
        raise click.ClickException(str(e)) from e


# =============================================================================
# GUI COMMAND
# =============================================================================


@main.command()
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="CSV file to load on startup"
)
def gui(file):
    """🖥️  Launch interactive PyQt dashboard for data exploration.

    The GUI provides:
    • High-performance interactive plotting (handles 3M+ points smoothly)
    • Real-time zoom, pan, and crosshair cursor
    • Material selection and analysis controls
    • Live statistics and analysis results
    • Export capabilities for plots and analysis data
    • Much better user experience than matplotlib widgets

    Controls:
    • Mouse wheel: Zoom in/out
    • Mouse drag: Pan around
    • Right-click: Context menu with zoom options
    • Crosshair follows mouse for precise readings

    FILE: Optional CSV file to load on startup
    """
    click.echo("🖥️  Launching Interactive PyQt Dashboard...")

    try:
        from .gui import PYQT_AVAILABLE

        if not PYQT_AVAILABLE:
            click.echo("❌ PyQt5 and pyqtgraph are required for GUI mode")
            click.echo("💡 Install with: poetry add PyQt5 pyqtgraph")
            raise click.ClickException("Missing GUI dependencies")

        click.echo("✅ PyQt5 and pyqtgraph available")

        if file:
            click.echo(f"📁 Will load: {file}")

        click.echo("🚀 Starting GUI application...")

        # Import and launch GUI
        import sys

        from PyQt5 import QtWidgets

        from .gui.main_window import FishingLineAnalyzerGUI

        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("Fishing Line Impact Analyzer v3.0")
        app.setOrganizationName("Nanosystems Lab")
        app.setStyle("Fusion")  # Modern look

        window = FishingLineAnalyzerGUI()

        # Load file if specified
        if file:
            window.load_file(str(file))

        window.show()

        # This will block until GUI is closed
        sys.exit(app.exec_())

    except ImportError as e:
        click.echo(f"❌ Import error: {e}")
        click.echo("💡 Install PyQt dependencies:")
        click.echo("   poetry add PyQt5 pyqtgraph")
        raise click.ClickException("GUI dependencies not available") from e
    except Exception as e:
        click.echo(f"❌ Error launching GUI: {e}")
        raise click.ClickException(str(e)) from e


# =============================================================================
# UTILITY COMMANDS
# =============================================================================


@main.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--count", "-n", default=5, help="Number of files to check")
def quick_check(data_dir, count):
    """⚡ Quick impulse check of multiple files (first N files).

    Provides a rapid overview of impulse analysis results for the first
    few files in a directory. Useful for quick validation or debugging.

    DATA_DIR: Directory containing CSV files
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))[:count]

    click.echo(f"⚡ QUICK IMPULSE CHECK ({len(csv_files)} files)")
    click.echo("-" * 65)
    click.echo("File               Mat  Total Impulse    Direction  Peak Force")
    click.echo("-" * 65)

    success_count = 0

    for file_path in csv_files:
        try:
            material = file_path.name.split("-")[0]
            result = analyze_single_file_with_impulse(file_path, material_code=material)

            if "error" not in result:
                impulse = result["total_impulse"]
                peak_force = result["peak_force"]

                direction = "Forward →" if impulse > 0 else "Backward ←"
                click.echo(
                    f"{file_path.name:<18} {material:<4} {impulse:>+12.6f} N*s"
                    f"{direction:<10} {peak_force:>6.0f} N"
                )
                success_count += 1
            else:
                click.echo(f"{file_path.name:<18} ERROR: {result['error'][:30]}...")

        except Exception as e:
            click.echo(f"{file_path.name:<18} EXCEPTION: {str(e)[:30]}...")

    click.echo("-" * 65)
    if success_count > 0:
        click.echo(f"📊 Summary: {success_count}/{len(csv_files)} files successful")
        click.echo("💡 Use 'analyze-impulse' for complete analysis of all files")


@main.command()
def info():
    """ℹ️  Show package information and usage examples.

    Displays configuration details, usage examples, and scientific background
    for the impulse analysis method.
    """
    click.echo("🎯 FISHING LINE FLYBACK IMPACT ANALYSIS v3.0")
    click.echo("=" * 60)
    click.echo()

    click.echo("📖 SCIENTIFIC METHOD:")
    click.echo("   Impulse Analysis: ∫ F(t) dt")
    click.echo("   • Direct measurement of momentum transfer")
    click.echo("   • More relevant to fishing line performance than kinetic energy")
    click.echo("   • Simple integration of complete force curve")
    click.echo("   • No assumptions about energy conversion efficiency")
    click.echo()

    click.echo("⚖️  MEASURED PARAMETERS:")
    click.echo(
        f'Line length: {MEASURED_LINE_LENGTH_INCHES}"'
        f"({MEASURED_LINE_MASS_GRAMS}g measured)"
    )
    click.echo(
        f"   Effective line mass: {LINE_MASS_FRACTION * 100:.0f}% (literature validated)"  # noqa: B950
    )
    click.echo(
        f"   Total line mass: 38.8g → {LINE_MASS_FRACTION * 38.8:.0f}g effective"
    )  # noqa: B950
    click.echo()

    click.echo("🏗️  HARDWARE CONFIGURATIONS:")
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        material_name = MATERIAL_NAMES.get(config, config)
        click.echo(f"   {config}: {material_name:<12} {weight_kg * 1000:.0f}g hardware")
    click.echo()

    click.echo("🚀 USAGE EXAMPLES:")
    click.echo("   # Analyze all files in directory:")
    click.echo(
        "   poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv"  # noqa: B950
    )
    click.echo()
    click.echo("   # Analyze single file with validation plot:")
    click.echo(
        "   poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-single file.csv --show-plot"  # noqa: B950
    )
    click.echo()
    click.echo("   # Quick check of first 5 files:")
    click.echo(
        "   poetry run python -m Fishing_Line_Flyback_Impact_Analysis quick-check data/csv -n 5"  # noqa: B950
    )
    click.echo()
    click.echo("   # Interactive force curve exploration:")
    click.echo(
        "   poetry run python -m Fishing_Line_Flyback_Impact_Analysis interactive-plot data/csv/STND-21-5.csv"  # noqa: B950
    )
    click.echo()

    click.echo("🗂️  LEGACY ACCESS:")
    click.echo("   Kinetic energy analysis and windowing tools are available via:")
    click.echo(
        "   poetry run python -m Fishing_Line_Flyback_Impact_Analysis legacy --help"
    )


# =============================================================================
# LEGACY ACCESS GROUP
# =============================================================================


@main.group()
def legacy():
    """🗂️ Access legacy kinetic energy analysis and windowing tools.

    These commands provide access to the original kinetic energy analysis
    methods and interactive windowing tools. Use for comparison studies
    or when legacy functionality is specifically needed.
    """
    pass


@legacy.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="ke_analysis")
def ke_analysis(data_dir, output_dir):
    """Run legacy kinetic energy analysis."""
    click.echo("🔄 Loading legacy kinetic energy analysis...")
    try:
        from .legacy.kinetic_energy_analysis import run_comprehensive_analysis

        click.echo(f"📊 Running KE analysis on: {data_dir}")
        results = run_comprehensive_analysis(data_dir, output_dir)
        print(results)
        click.echo(f"✅ Legacy KE analysis complete: {output_dir}")
    except ImportError:
        click.echo("❌ Legacy KE analysis module not available")
        click.echo("💡 Kinetic energy analysis has been moved to legacy/")


@legacy.command()
@click.argument("file_path", type=click.Path(exists=True))
def window_tool(file_path):
    """Launch interactive windowing tool."""
    click.echo("🎛️  Loading interactive windowing tool...")
    try:
        from .legacy.windowing import interactive_window_tool

        interactive_window_tool(str(file_path))
    except ImportError:
        click.echo("❌ Windowing tool not available")
        click.echo("💡 Windowing tools have been moved to legacy/")


if __name__ == "__main__":
    main()
