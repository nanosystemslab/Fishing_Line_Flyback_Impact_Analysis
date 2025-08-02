"""Visualization Module for Fishing Line Flyback Impact Analysis v3.0.

This module provides visualization functions focused on impulse analysis,
with the essential plotting capabilities for force curves, single file analysis,
and interactive data exploration.

Key Features:
✓ Force curve visualization with zoom capability
✓ Single file analysis plots (impulse-focused)
✓ Interactive force preview for data exploration
✓ Boundary validation plots
✓ Publication-quality impulse box plots
✓ Shared data processing integration

Removed in v3.0:
- Legacy KE-specific visualization classes
- Unused methodology comparison plots
- Legacy compatibility functions
"""

import platform
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.widgets import Button

# Import impulse analysis for plotting
from .impulse_analysis import analyze_single_file_with_impulse

# Import shared components
from .shared import MATERIAL_NAMES
from .shared import calculate_total_force
from .shared import extract_material_code
from .shared import get_time_array
from .shared import load_csv_file


# Configure matplotlib for cross-platform compatibility
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
else:
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 10

# Configure matplotlib for consistent plotting
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


# =============================================================================
# CORE VISUALIZATION FUNCTIONS (KEPT)
# =============================================================================


def show_force_preview(
    force_data: np.ndarray,
    time_data: np.ndarray,
    filename: str,
    show_analysis_preview: bool = False,
    mass: float = 0.045,
) -> plt.Figure:
    """Show a preview of force vs time data for visual inspection.

    This function creates an interactive plot showing the complete force curve,
    allowing you to examine the data characteristics, identify impact regions,
    and assess data quality before analysis.

    Args:
        force_data: Force data in Newtons
        time_data: Time data in seconds
        filename: Filename for plot title
        show_analysis_preview: Whether to overlay analysis boundaries
        mass: Mass for analysis preview (kg)

    Returns:
        matplotlib Figure object
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Convert time to milliseconds for better readability
    time_ms = time_data * 1000

    # Top plot: Full force curve
    ax1.plot(time_ms, force_data, "b-", linewidth=1, alpha=0.8, label="Force")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Force (N)")
    ax1.set_title(f"Force vs Time: {filename}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add statistics text
    stats_text = f"""Data Statistics:
            Duration: {time_data[-1]:.4f} s ({time_ms[-1]:.1f} ms)
            Points: {len(force_data):,}
            Force Range: {np.min(force_data):.1f} to {np.max(force_data):.1f} N
            Peak |Force|: {np.max(np.abs(force_data)):.1f} N
            RMS Force: {np.sqrt(np.mean(force_data**2)):.1f} N"""

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
        fontfamily="monospace",
    )

    # Bottom plot: Zoomed view around peak force
    peak_idx = np.argmax(np.abs(force_data))
    peak_time_ms = time_ms[peak_idx]

    # Define zoom window (±50ms around peak, or ±500 samples, whichever is smaller)
    zoom_window_ms = 50  # ms
    zoom_samples = min(500, int(len(force_data) * zoom_window_ms / time_ms[-1]))

    zoom_start = max(0, peak_idx - zoom_samples)
    zoom_end = min(len(force_data), peak_idx + zoom_samples)

    zoom_time = time_ms[zoom_start:zoom_end]
    zoom_force = force_data[zoom_start:zoom_end]

    ax2.plot(zoom_time, zoom_force, "r-", linewidth=2, label="Force (Zoomed)")
    ax2.axvline(
        peak_time_ms,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Peak at {peak_time_ms:.1f} ms",
    )
    ax2.scatter(
        peak_time_ms,
        force_data[peak_idx],
        color="red",
        s=100,
        zorder=5,
        label=f"Peak: {force_data[peak_idx]:.1f} N",
    )

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Force (N)")
    ax2.set_title(f"Zoomed View: ±{zoom_window_ms} ms around peak")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Optional: Add analysis preview
    if show_analysis_preview:
        try:
            # Quick analysis to show boundaries
            material_code = extract_material_code(filename)
            from .impulse_analysis import ImpulseAnalyzer

            analyzer = ImpulseAnalyzer(material_code=material_code)
            impact_start, impact_end = analyzer.find_impact_boundaries(force_data)

            # Add boundaries to both plots
            for ax in [ax1, ax2]:
                ax.axvline(
                    time_ms[impact_start],
                    color="green",
                    linestyle=":",
                    alpha=0.8,
                    label="Impact Start",
                )
                ax.axvline(
                    time_ms[impact_end],
                    color="green",
                    linestyle=":",
                    alpha=0.8,
                    label="Impact End",
                )
                ax.legend()

            # Add analysis info
            duration_ms = (time_data[impact_end] - time_data[impact_start]) * 1000
            analysis_text = f"Analysis Preview:\nImpact Duration: {duration_ms:.1f} ms\nMaterial: {material_code}"  # noqa: B950
            ax2.text(
                0.98,
                0.02,
                analysis_text,
                transform=ax2.transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
                fontsize=9,
            )

        except Exception as e:
            print(f"Warning: Could not add analysis preview: {e}")

    plt.tight_layout()
    plt.show()

    return fig


def plot_single_file_analysis(
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> None:
    """Create analysis plot for a single CSV file using impulse method.

    This function loads a CSV file, performs impulse analysis, and creates
    a detailed plot showing the force curve with analysis boundaries and results.

    Args:
        csv_path: Path to CSV file
        output_dir: Output directory for saving plot (optional)
        show_plot: Whether to display the plot interactively
    """
    csv_path = Path(csv_path)

    try:
        # Load and process data using shared components
        df = load_csv_file(csv_path)
        force_data, force_columns = calculate_total_force(df)
        time_data, sampling_rate = get_time_array(df, len(force_data))

        # Perform impulse analysis
        result = analyze_single_file_with_impulse(csv_path, show_plot=False)

        if "error" in result:
            print(f"❌ Error analyzing {csv_path.name}: {result['error']}")
            return

        # Create detailed analysis plot
        fig = create_single_file_analysis_plot(
            force_data, time_data, result, csv_path.name
        )

        # Save plot if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"analysis_{csv_path.stem}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"📊 Analysis plot saved to: {plot_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        print(f"❌ Error creating plot for {csv_path.name}: {e}")


def create_single_file_analysis_plot(
    force_data: np.ndarray, time_data: np.ndarray, analysis_result: Dict, filename: str
) -> plt.Figure:
    """Create detailed analysis plot for a single file with impulse results.

    Args:
        force_data: Force data in Newtons
        time_data: Time data in seconds
        analysis_result: Analysis results from impulse analysis
        filename: Filename for plot title

    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Convert time to milliseconds
    time_ms = time_data * 1000

    # Extract analysis results
    impact_start = analysis_result.get("impact_start_idx", 0)
    impact_end = analysis_result.get("impact_end_idx", len(force_data) - 1)
    total_impulse = analysis_result.get("total_impulse", 0)
    peak_force = analysis_result.get("peak_force", 0)
    impact_duration = analysis_result.get("impact_duration", 0) * 1000  # Convert to ms

    # Plot 1: Full force curve with analysis boundaries
    ax1.plot(time_ms, force_data, "b-", linewidth=1, alpha=0.7, label="Force")
    ax1.axvline(
        time_ms[impact_start],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Impact Start",
    )
    ax1.axvline(
        time_ms[impact_end],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Impact End",
    )

    # Highlight impact region
    impact_mask = np.zeros_like(force_data, dtype=bool)
    impact_mask[impact_start : impact_end + 1] = True
    ax1.fill_between(
        time_ms,
        force_data,
        where=impact_mask,
        alpha=0.3,
        color="green",
        label="Impact Region",
    )

    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Force (N)")
    ax1.set_title(f"Force vs Time: {filename}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Zoomed impact region
    if impact_end > impact_start:
        zoom_start = max(0, impact_start - 50)
        zoom_end = min(len(force_data), impact_end + 50)

        zoom_time = time_ms[zoom_start:zoom_end]
        zoom_force = force_data[zoom_start:zoom_end]

        ax2.plot(zoom_time, zoom_force, "r-", linewidth=2, label="Force")
        ax2.axvline(
            time_ms[impact_start],
            color="green",
            linestyle="--",
            linewidth=2,
            label="Start",
        )
        ax2.axvline(
            time_ms[impact_end], color="green", linestyle="--", linewidth=2, label="End"
        )

        # Fill integration area
        impact_time_zoom = time_ms[impact_start : impact_end + 1]
        impact_force_zoom = force_data[impact_start : impact_end + 1]
        ax2.fill_between(
            impact_time_zoom,
            impact_force_zoom,
            alpha=0.4,
            color="green",
            label="Integration Area",
        )

        # Mark peak
        peak_idx = impact_start + np.argmax(
            np.abs(force_data[impact_start : impact_end + 1])
        )
        ax2.scatter(
            time_ms[peak_idx],
            force_data[peak_idx],
            color="red",
            s=100,
            zorder=5,
            label=f"Peak: {force_data[peak_idx]:.0f} N",
        )

        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Force (N)")
        ax2.set_title("Impact Region Detail")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # Plot 3: Cumulative impulse
    cumulative_impulse = np.cumsum(force_data) * np.mean(np.diff(time_data))
    ax3.plot(
        time_ms, cumulative_impulse, "purple", linewidth=2, label="Cumulative Impulse"
    )
    ax3.axhline(
        total_impulse,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Total: {total_impulse:+.6f} N⋅s",
    )
    ax3.axvline(time_ms[impact_start], color="red", linestyle="--", alpha=0.7)
    ax3.axvline(time_ms[impact_end], color="red", linestyle="--", alpha=0.7)

    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Cumulative Impulse (N⋅s)")
    ax3.set_title("Impulse Integration")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Analysis summary (text)
    ax4.axis("off")

    # Create summary text
    material_code = analysis_result.get("material_code", "Unknown")
    material_name = MATERIAL_NAMES.get(material_code, material_code)

    summary_text = f"""IMPULSE ANALYSIS RESULTS
        {'=' * 30}
        File: {filename}
        Material: {material_code} ({material_name})\n
        IMPULSE METRICS:
        Total Impulse: {total_impulse:+.6f} N⋅s
        Absolute Impulse: {analysis_result.get('total_abs_impulse', 0):.6f} N⋅s
        Impact Impulse: {analysis_result.get('impact_impulse', 0):+.6f} N⋅s\n
        FORCE CHARACTERISTICS:
        Peak Force: {peak_force:.0f} N
        Peak (+): {analysis_result.get('peak_force_positive', 0):.0f} N
        Peak (-): {analysis_result.get('peak_force_negative', 0):.0f} N
        RMS Force: {analysis_result.get('rms_force', 0):.0f} N\n
        TIMING:
        Impact Duration: {impact_duration:.1f} ms
        Total Duration: {time_ms[-1]:.1f} ms
        Sampling Rate: {analysis_result.get('sampling_rate_hz', 0):.0f} Hz \n
        EQUIVALENT METRICS:
        Velocity: {analysis_result.get('equivalent_velocity', 0):.0f} m/s
        Kinetic Energy: {analysis_result.get('equivalent_kinetic_energy', 0):.6f} J \n
        MASS BREAKDOWN:
        {analysis_result.get('mass_breakdown', {})}
        """

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    return fig


# =============================================================================
# IMPULSE-SPECIFIC VISUALIZATION FUNCTIONS
# =============================================================================


def create_impulse_material_comparison(
    results: List[Dict], output_dir: Union[str, Path]
) -> None:
    """Create material comparison plots for impulse analysis results.

    Args:
        results: List of impulse analysis results
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid results
    valid_results = [
        r for r in results if "error" not in r and "total_abs_impulse" in r
    ]

    if not valid_results:
        print("❌ No valid results for material comparison")
        return

    # Create DataFrame
    df = pd.DataFrame(valid_results)

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Impulse by material (box plot)
    if "material_type" in df.columns:
        sns.boxplot(data=df, x="material_type", y="total_abs_impulse", ax=ax1)
        ax1.set_title("Total Absolute Impulse by Material")
        ax1.set_xlabel("Material Type")
        ax1.set_ylabel("Total Absolute Impulse (N⋅s)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

    # 2. Peak force by material
    if "material_type" in df.columns:
        sns.boxplot(data=df, x="material_type", y="peak_force", ax=ax2)
        ax2.set_title("Peak Force by Material")
        ax2.set_xlabel("Material Type")
        ax2.set_ylabel("Peak Force (N)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

    # 3. Impact duration distribution
    ax3.hist(df["impact_duration"] * 1000, bins=20, alpha=0.7, edgecolor="black")
    ax3.set_xlabel("Impact Duration (ms)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Impact Duration Distribution")
    ax3.grid(True, alpha=0.3)

    # 4. Force vs Impulse scatter
    ax4.scatter(df["peak_force"], df["total_abs_impulse"], alpha=0.6, s=60)
    ax4.set_xlabel("Peak Force (N)")
    ax4.set_ylabel("Total Absolute Impulse (N⋅s)")
    ax4.set_title("Force vs Impulse Relationship")
    ax4.grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(df) > 2:
        corr = np.corrcoef(df["peak_force"], df["total_abs_impulse"])[0, 1]
        ax4.text(
            0.05,
            0.95,
            f"R = {corr:.3f}",
            transform=ax4.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "impulse_material_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"📊 Material comparison plot saved to: {plot_path}")
    plt.close()


def create_impulse_summary_statistics(
    results: List[Dict], output_dir: Union[str, Path]
) -> str:
    """Create a text summary of impulse analysis results.

    Args:
        results: List of analysis results
        output_dir: Output directory for saving summary

    Returns:
        Summary text string
    """
    if not results:
        return "No results to summarize."

    valid_results = [r for r in results if "error" not in r and "total_impulse" in r]

    if not valid_results:
        return "No valid impulse results to summarize."

    # Calculate statistics
    impulses = [r["total_impulse"] for r in valid_results]
    abs_impulses = [r["total_abs_impulse"] for r in valid_results]
    peak_forces = [r["peak_force"] for r in valid_results]
    durations = [r["impact_duration"] * 1000 for r in valid_results]  # Convert to ms

    summary = f"""
        IMPULSE ANALYSIS SUMMARY
        {'=' * 40}
        Total files processed: {len(results)}
        Successful analyses: {len(valid_results)}
        Success rate: {len(valid_results) / len(results) * 100:.1f}%\n
        IMPULSE STATISTICS:
        Total impulse range: {np.min(impulses):+.6f} to {np.max(impulses):+.6f} N⋅s
        Mean total impulse: {np.mean(impulses):+.6f} N⋅s
        Std total impulse: {np.std(impulses):.6f} N⋅s\n
        Absolute impulse range: {np.min(abs_impulses):.6f} to
        {np.max(abs_impulses):.6f} N⋅
        Mean absolute impulse: {np.mean(abs_impulses):.6f} N⋅s
        Std absolute impulse: {np.std(abs_impulses):.6f} N⋅s\n
        FORCE STATISTICS:
        Peak force range: {np.min(peak_forces):.0f} to {np.max(peak_forces):.0f} N
        Mean peak force: {np.mean(peak_forces):.0f} N
        Std peak force: {np.std(peak_forces):.0f} N\n
        DURATION STATISTICS:
        Duration range: {np.min(durations):.1f} to {np.max(durations):.1f} ms
        Mean duration: {np.mean(durations):.1f} ms
        Std duration: {np.std(durations):.1f} ms
        """

    # Add material breakdown if available
    if valid_results and "material_type" in valid_results[0]:
        df = pd.DataFrame(valid_results)
        summary += "\nMATERIAL BREAKDOWN:\n"
        for material in sorted(df["material_type"].unique()):
            material_data = df[df["material_type"] == material]
            count = len(material_data)
            avg_impulse = material_data["total_impulse"].mean()
            avg_abs_impulse = material_data["total_abs_impulse"].mean()
            avg_force = material_data["peak_force"].mean()

            summary += f"{material}: {count} samples | "
            summary += f"Impulse: {avg_impulse:+.6f} N⋅s | "
            summary += f"Abs: {avg_abs_impulse:.6f} N⋅s | "
            summary += f"Force: {avg_force:.0f} N\n"

    # Save summary to file
    if output_dir:
        output_dir = Path(output_dir)
        summary_file = output_dir / "impulse_analysis_summary.txt"
        with open(summary_file, "w") as f:
            f.write(summary)
        print(f"📄 Summary saved to: {summary_file}")

    return summary.strip()


# =============================================================================
# INTERACTIVE MODE
# =============================================================================


def create_interactive_force_explorer(  # noqa: C901
    force_data: np.ndarray,
    time_data: np.ndarray,
    filename: str,
    show_analysis_preview: bool = False,
) -> None:
    """Create interactive force data explorer.

    Capable of zoom, pan, and selection capabilities.

    Features:
    - Mouse wheel zoom
    - Click and drag to pan
    - Span selector to zoom to specific regions
    - Reset zoom button
    - Analysis boundary overlay (optional)
    - Real-time statistics display

    Args:
        force_data: Force data in Newtons
        time_data: Time data in seconds
        filename: Filename for plot title
        show_analysis_preview: Whether to overlay analysis boundaries
    """
    # Convert time to milliseconds for better readability
    time_ms = time_data * 1000

    # Create figure with subplots and make it interactive
    plt.ion()  # Turn on interactive mode
    fig, (ax_main, ax_zoom, ax_stats) = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"Interactive Force Explorer: {filename}", fontsize=14, fontweight="bold"
    )

    # Main plot with full data
    (line_main,) = ax_main.plot(
        time_ms, force_data, "b-", linewidth=1, alpha=0.8, label="Force"
    )
    ax_main.set_xlabel("Time (ms)")
    ax_main.set_ylabel("Force (N)")
    ax_main.set_title("Full Force Curve (Use mouse wheel to zoom, click+drag to pan)")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    # Store original limits
    original_xlim = ax_main.get_xlim()
    original_ylim = ax_main.get_ylim()

    # Zoom plot (will show selected region)
    (line_zoom,) = ax_zoom.plot([], [], "r-", linewidth=2, label="Zoomed Region")
    ax_zoom.set_xlabel("Time (ms)")
    ax_zoom.set_ylabel("Force (N)")
    ax_zoom.set_title("Zoomed View (Select region in main plot)")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend()

    # Statistics plot (text display)
    ax_stats.axis("off")

    # Add analysis boundaries if requested
    boundary_lines = []
    if show_analysis_preview:
        try:
            from ..impulse_analysis import ImpulseAnalyzer
            from ..shared import extract_material_code

            material_code = extract_material_code(filename)
            analyzer = ImpulseAnalyzer(material_code=material_code)
            impact_start, impact_end = analyzer.find_impact_boundaries(force_data)

            # Add boundary lines to main plot
            start_line = ax_main.axvline(
                time_ms[impact_start],
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Impact Start",
            )
            end_line = ax_main.axvline(
                time_ms[impact_end],
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Impact End",
            )
            boundary_lines = [start_line, end_line]

            print(boundary_lines)

            # Highlight impact region
            impact_region = patches.Rectangle(
                (time_ms[impact_start], ax_main.get_ylim()[0]),
                time_ms[impact_end] - time_ms[impact_start],
                ax_main.get_ylim()[1] - ax_main.get_ylim()[0],
                alpha=0.2,
                facecolor="yellow",
                label="Impact Region",
            )
            ax_main.add_patch(impact_region)
            ax_main.legend()

        except Exception as e:
            print(f"Warning: Could not add analysis preview: {e}")

    # Function to update statistics display
    def update_stats(xlim=None, ylim=None):
        """Update statistics display for current view."""
        if xlim is None:
            xlim = ax_main.get_xlim()
        if ylim is None:
            ylim = ax_main.get_ylim()

        # Find data points in current view
        time_mask = (time_ms >= xlim[0]) & (time_ms <= xlim[1])
        force_mask = (force_data >= ylim[0]) & (force_data <= ylim[1])
        view_mask = time_mask & force_mask

        if np.any(view_mask):
            view_time = time_ms[view_mask]
            view_force = force_data[view_mask]
            stats_text = f"""CURRENT VIEW STATISTICS
                {'=' * 30}
                Time Range: {xlim[0]:.1f} - {xlim[1]:.1f} ms
                Duration: {xlim[1] - xlim[0]:.1f} ms
                Data Points in View: {np.sum(view_mask):,}
                View Time: {view_time}
                Force Statistics:
                  Min: {np.min(view_force):.1f} N
                  Max: {np.max(view_force):.1f} N
                  Peak |Force|: {np.max(np.abs(view_force)):.1f} N
                  Mean: {np.mean(view_force):.1f} N
                  RMS: {np.sqrt(np.mean(view_force**2)):.1f} N
                  Std Dev: {np.std(view_force):.1f} N
                FULL DATA STATISTICS
                {'=' * 20}
                Total Duration: {time_ms[-1]:.1f} ms
                Total Points: {len(force_data):,}
                Global Min: {np.min(force_data):.1f} N
                Global Max: {np.max(force_data):.1f} N
                Global Peak |F|: {np.max(np.abs(force_data)):.1f} N
                CONTROLS:
                - Mouse wheel: Zoom in/out
                - Click + drag: Pan around
                - Select region below to zoom
                - Use Reset button to restore view"""

        else:
            stats_text = "No data points in current view"

        ax_stats.clear()
        ax_stats.axis("off")
        ax_stats.text(
            0.05,
            0.95,
            stats_text,
            transform=ax_stats.transAxes,
            verticalalignment="top",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    # Initialize statistics
    update_stats()

    # Span selector for zooming to specific regions
    def onselect(xmin, xmax):
        """Handle span selection for zooming."""
        # Update zoom plot
        time_mask = (time_ms >= xmin) & (time_ms <= xmax)
        if np.any(time_mask):
            zoom_time = time_ms[time_mask]
            zoom_force = force_data[time_mask]

            line_zoom.set_data(zoom_time, zoom_force)
            ax_zoom.set_xlim(xmin, xmax)
            ax_zoom.set_ylim(np.min(zoom_force) * 1.1, np.max(zoom_force) * 1.1)
            ax_zoom.set_title(
                f"Zoomed View: {xmin:.1f} - {xmax:.1f} ms ({xmax - xmin:.1f} ms duration)"  # noqa: B950
            )

            # Also zoom main plot
            ax_main.set_xlim(xmin, xmax)

            # Update statistics for zoomed region
            update_stats((xmin, xmax))

            fig.canvas.draw()

    # Reset zoom function
    def reset_zoom(event):
        """Reset to original zoom level."""
        ax_main.set_xlim(original_xlim)
        ax_main.set_ylim(original_ylim)
        ax_zoom.clear()
        ax_zoom.set_xlabel("Time (ms)")
        ax_zoom.set_ylabel("Force (N)")
        ax_zoom.set_title("Zoomed View (Select region in main plot)")
        ax_zoom.grid(True, alpha=0.3)
        (line_zoom,) = ax_zoom.plot([], [], "r-", linewidth=2, label="Zoomed Region")
        ax_zoom.legend()
        update_stats()
        fig.canvas.draw()

    # Add reset button
    ax_button = plt.axes([0.02, 0.02, 0.1, 0.04])
    button = Button(ax_button, "Reset Zoom")
    button.on_clicked(reset_zoom)

    # Mouse wheel zoom functionality
    def on_scroll(event):
        """Handle mouse wheel scrolling for zoom."""
        if event.inaxes != ax_main:
            return

        # Get current limits
        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()

        # Zoom factor
        zoom_factor = 1.2 if event.step < 0 else 1 / 1.2

        # Get mouse position
        x_mouse = event.xdata
        y_mouse = event.ydata

        if x_mouse is None or y_mouse is None:
            return

        # Calculate new limits centered on mouse position
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor

        new_xlim = [
            x_mouse - new_x_range * (x_mouse - xlim[0]) / x_range,
            x_mouse + new_x_range * (xlim[1] - x_mouse) / x_range,
        ]
        new_ylim = [
            y_mouse - new_y_range * (y_mouse - ylim[0]) / y_range,
            y_mouse + new_y_range * (ylim[1] - y_mouse) / y_range,
        ]

        ax_main.set_xlim(new_xlim)
        ax_main.set_ylim(new_ylim)
        update_stats(new_xlim, new_ylim)
        fig.canvas.draw()

    # Connect scroll event
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Instructions
    print("\n🎛️ INTERACTIVE CONTROLS:")
    print("• Mouse wheel: Zoom in/out")
    print("• Click + drag on main plot: Select region to zoom")
    print("• Reset Zoom button: Return to full view")
    print("• Close window when done exploring")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for button
    plt.show(block=True)  # Block until window is closed


def create_dual_view_explorer(  # noqa: C901
    force_data: np.ndarray, time_data: np.ndarray, filename: str
) -> None:
    """Create a dual-view interactive explorer with overview and detail views.

    Features:
    - Overview plot showing full data
    - Detail plot showing zoomed region
    - Interactive selection rectangle
    - Mouse controls for navigation
    """
    time_ms = time_data * 1000

    plt.ion()
    fig = plt.figure(figsize=(16, 10))

    # Create custom layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], width_ratios=[3, 2])

    # Overview plot (top left)
    ax_overview = fig.add_subplot(gs[0, :])
    ax_overview.plot(time_ms, force_data, "b-", linewidth=1, alpha=0.7)
    ax_overview.set_title(f"Overview: {filename}")
    ax_overview.set_xlabel("Time (ms)")
    ax_overview.set_ylabel("Force (N)")
    ax_overview.grid(True, alpha=0.3)

    # Detail plot (middle left)
    ax_detail = fig.add_subplot(gs[1, 0])
    (detail_line,) = ax_detail.plot([], [], "r-", linewidth=2)
    ax_detail.set_title("Detail View (Select region in overview)")
    ax_detail.set_xlabel("Time (ms)")
    ax_detail.set_ylabel("Force (N)")
    ax_detail.grid(True, alpha=0.3)

    # Info panel (right side)
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")

    # Selection rectangle
    selection_rect = patches.Rectangle(
        (0, 0), 0, 0, linewidth=2, edgecolor="red", facecolor="none", alpha=0.7
    )
    ax_overview.add_patch(selection_rect)

    def update_info(xlim=None):
        """Update information panel."""
        if xlim is None:
            # Show overall statistics
            info_text = f"""FILE INFORMATION
                    {'=' * 20}
                    Filename: {filename}
                    Duration: {time_ms[-1]:.1f} ms
                    Sample Rate: {len(time_data) / (time_data[-1]):.0f} Hz
                    Total Points: {len(force_data):,}\n
                    FORCE STATISTICS
                    {'=' * 16}
                    Min Force: {np.min(force_data):.1f} N
                    Max Force: {np.max(force_data):.1f} N
                    Peak |Force|: {np.max(np.abs(force_data)):.1f} N
                    Mean Force: {np.mean(force_data):.1f} N
                    RMS Force: {np.sqrt(np.mean(force_data ** 2)):.1f} N
                    Std Dev: {np.std(force_data):.1f} N\n
                    INSTRUCTIONS
                    {'=' * 12}
                    1. Click and drag on overview
                       to select a region
                    2. Selected region will appear
                       in detail view
                    3. Use mouse wheel to zoom
                    4. Right-click to reset\
                    SELECTION
                    {'=' * 9}
                    Click and drag to select region
                    for detailed analysis"""
        else:
            # Show selection statistics
            time_mask = (time_ms >= xlim[0]) & (time_ms <= xlim[1])
            if np.any(time_mask):
                sel_force = force_data[time_mask]
                duration = xlim[1] - xlim[0]
                energy_est = 0.5 * 0.072 * (np.max(np.abs(sel_force)) / 0.072) ** 2
                impuse_est = np.trapz(np.abs(sel_force), time_ms[time_mask]) / 1000

                info_text = f"""SELECTED REGION
                    {'=' * 15}
                    Time Range: {xlim[0]:.1f} - {xlim[1]:.1f} ms
                    Duration: {duration:.1f} ms
                    Points: {np.sum(time_mask):,}\n
                    FORCE IN SELECTION
                    {'=' * 18}
                    Min: {np.min(sel_force):.1f} N
                    Max: {np.max(sel_force):.1f} N
                    Peak |F|: {np.max(np.abs(sel_force)):.1f} N
                    Mean: {np.mean(sel_force):.1f} N
                    RMS: {np.sqrt(np.mean(sel_force**2)):.1f} N \n
                    FULL DATA REFERENCE
                    {'=' * 19}
                    Total Duration: {time_ms[-1]:.1f} ms
                    Global Min: {np.min(force_data):.1f} N
                    Global Max: {np.max(force_data):.1f} N \n
                    ANALYSIS POTENTIAL
                    {'=' * 17}
                    Energy Est: {energy_est:.6f} J
                    Impulse Est: {impuse_est:.6f} N⋅s\n
                    Right-click to reset selection"""
            else:
                info_text = "No data in selection"

        ax_info.clear()
        ax_info.axis("off")
        ax_info.text(
            0.05,
            0.95,
            info_text,
            transform=ax_info.transAxes,
            verticalalignment="top",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

    # Initialize info
    update_info()

    # Selection variables
    selecting = False
    start_x = None

    def on_press(event):
        """Handle mouse press for selection."""
        nonlocal selecting, start_x
        if event.inaxes == ax_overview and event.button == 1:  # Left click
            selecting = True
            start_x = event.xdata
            selection_rect.set_width(0)
            selection_rect.set_height(
                ax_overview.get_ylim()[1] - ax_overview.get_ylim()[0]
            )
            selection_rect.set_xy((start_x, ax_overview.get_ylim()[0]))

    def on_motion(event):
        """Handle mouse motion during selection."""
        if selecting and event.inaxes == ax_overview and event.xdata is not None:
            width = event.xdata - start_x
            selection_rect.set_width(width)
            fig.canvas.draw_idle()

    def on_release(event):
        """Handle mouse release to complete selection."""
        nonlocal selecting
        if selecting and event.inaxes == ax_overview and event.button == 1:
            selecting = False
            end_x = event.xdata

            if start_x is not None and end_x is not None:
                xlim = [min(start_x, end_x), max(start_x, end_x)]

                # Update detail plot
                time_mask = (time_ms >= xlim[0]) & (time_ms <= xlim[1])
                if np.any(time_mask):
                    detail_time = time_ms[time_mask]
                    detail_force = force_data[time_mask]

                    detail_line.set_data(detail_time, detail_force)
                    ax_detail.set_xlim(xlim)
                    ax_detail.set_ylim(
                        np.min(detail_force) * 1.1, np.max(detail_force) * 1.1
                    )
                    ax_detail.set_title(f"Detail: {xlim[0]:.1f} - {xlim[1]:.1f} ms")

                    # Update info panel
                    update_info(xlim)

                    fig.canvas.draw()
        elif event.button == 3:  # Right click to reset
            detail_line.set_data([], [])
            ax_detail.clear()
            ax_detail.set_title("Detail View (Select region in overview)")
            ax_detail.set_xlabel("Time (ms)")
            ax_detail.set_ylabel("Force (N)")
            ax_detail.grid(True, alpha=0.3)
            selection_rect.set_width(0)
            update_info()
            fig.canvas.draw()

    # Connect events
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.tight_layout()
    plt.show(block=True)


def show_force_preview_interactive(
    force_data: np.ndarray,
    time_data: np.ndarray,
    filename: str,
    show_analysis_preview: bool = False,
    style: str = "explorer",
) -> None:
    """Show interactive force preview with multiple interaction styles.

    Args:
        force_data: Force data array
        time_data: Time data array
        filename: Name of the file
        show_analysis_preview: Whether to show analysis boundaries
        style: Interaction style ("explorer" or other)

    Returns:
        None
    """
    if style != "explorer":
        # Fall back to simple static plot
        return show_force_preview(
            force_data, time_data, filename, show_analysis_preview
        )

    # Convert time to milliseconds for better readability
    time_ms = time_data * 1000

    # Create the interactive plot
    fig, axes_dict = _create_interactive_figure(filename, time_ms, force_data)

    # Add analysis boundaries if requested
    if show_analysis_preview:
        _add_analysis_boundaries(axes_dict["main"], time_ms, force_data, filename)

    # Set up interactivity
    _setup_interactive_controls(fig, axes_dict, time_ms, force_data)

    _print_usage_instructions()
    plt.show(block=True)


def _create_interactive_figure(
    filename: str, time_ms: np.ndarray, force_data: np.ndarray
):
    """Create the figure layout and initial plots."""
    # Create figure with proper scaling for large datasets
    plt.ion()  # Turn on interactive mode

    # Use a larger figure and adjust layout for better visibility
    fig = plt.figure(figsize=(16, 14))

    # Create custom grid layout with better proportions
    gs = fig.add_gridspec(
        4, 3, height_ratios=[3, 2, 2, 1], width_ratios=[4, 1, 1], hspace=0.3, wspace=0.3
    )

    # Main plot (top, spans all columns)
    ax_main = fig.add_subplot(gs[0, :])
    (line_main,) = ax_main.plot(
        time_ms, force_data, "b-", linewidth=0.5, alpha=0.8, label="Force"
    )
    ax_main.set_xlabel("Time (ms)", fontsize=12)
    ax_main.set_ylabel("Force (N)", fontsize=12)
    ax_main.set_title(
        f"Interactive Force Explorer: {filename}", fontsize=14, fontweight="bold"
    )
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    # Store original limits
    original_xlim = ax_main.get_xlim()
    original_ylim = ax_main.get_ylim()

    # Zoom plot (middle left)
    ax_zoom = fig.add_subplot(gs[1, :2])
    (line_zoom,) = ax_zoom.plot([], [], "r-", linewidth=2, label="Zoomed Region")
    ax_zoom.set_xlabel("Time (ms)", fontsize=10)
    ax_zoom.set_ylabel("Force (N)", fontsize=10)
    ax_zoom.set_title("Zoomed View (Select region in main plot)")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend()

    # Control panel (middle right)
    ax_controls = _create_controls_panel(fig, gs)

    # Statistics panel (bottom left and middle)
    ax_stats = fig.add_subplot(gs[2, :2])
    ax_stats.axis("off")

    # Info panel (bottom right)
    ax_info = _create_info_panel(fig, gs, force_data, time_ms)

    return fig, {
        "main": ax_main,
        "zoom": ax_zoom,
        "stats": ax_stats,
        "info": ax_info,
        "controls": ax_controls,
        "line_main": line_main,
        "line_zoom": line_zoom,
        "original_xlim": original_xlim,
        "original_ylim": original_ylim,
    }


def _create_controls_panel(fig, gs):
    """Create the controls panel with instructions."""
    ax_controls = fig.add_subplot(gs[1, 2])
    ax_controls.axis("off")
    ax_controls.text(
        0.1,
        0.9,
        "CONTROLS",
        fontweight="bold",
        fontsize=12,
        transform=ax_controls.transAxes,
    )
    ax_controls.text(
        0.1,
        0.7,
        "• Mouse wheel:\n  Zoom in/out",
        fontsize=9,
        transform=ax_controls.transAxes,
    )
    ax_controls.text(
        0.1,
        0.5,
        "• Click + drag:\n  Select region",
        fontsize=9,
        transform=ax_controls.transAxes,
    )
    ax_controls.text(
        0.1,
        0.3,
        "• Reset button:\n  Restore view",
        fontsize=9,
        transform=ax_controls.transAxes,
    )
    ax_controls.text(
        0.1, 0.1, "• Close window:\n  Exit", fontsize=9, transform=ax_controls.transAxes
    )
    return ax_controls


def _create_info_panel(fig, gs, force_data: np.ndarray, time_ms: np.ndarray):
    """Create the data info panel."""
    ax_info = fig.add_subplot(gs[2, 2])
    ax_info.axis("off")

    # Add data info that doesn't change
    data_info = f"""DATA INFO
{'=' * 12}
Points: {len(force_data):,}
Duration: {time_ms[-1]:.0f} ms
Range: {np.min(force_data):.0f} to
       {np.max(force_data):.0f} N
Peak: {np.max(np.abs(force_data)):.0f} N"""

    ax_info.text(
        0.1,
        0.9,
        data_info,
        fontsize=9,
        fontfamily="monospace",
        transform=ax_info.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )
    return ax_info


def _add_analysis_boundaries(
    ax_main, time_ms: np.ndarray, force_data: np.ndarray, filename: str
):
    """Add analysis boundaries to the main plot."""
    try:
        from .impulse_analysis import ImpulseAnalyzer

        material_code = extract_material_code(filename)
        analyzer = ImpulseAnalyzer(material_code=material_code)
        impact_start, impact_end = analyzer.find_impact_boundaries(force_data)

        # Add boundary lines to main plot
        ax_main.axvline(
            time_ms[impact_start],
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Impact Start",
        )
        ax_main.axvline(
            time_ms[impact_end],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Impact End",
        )

        # Highlight impact region
        original_ylim = ax_main.get_ylim()
        impact_region = patches.Rectangle(
            (time_ms[impact_start], original_ylim[0]),
            time_ms[impact_end] - time_ms[impact_start],
            original_ylim[1] - original_ylim[0],
            alpha=0.2,
            facecolor="yellow",
            label="Impact Region",
        )
        ax_main.add_patch(impact_region)
        ax_main.legend()

    except Exception as e:
        print(f"Warning: Could not add analysis preview: {e}")


def _create_stats_updater(axes_dict, time_ms: np.ndarray, force_data: np.ndarray):
    """Create the statistics update function."""

    def update_stats(xlim=None, ylim=None):
        ax_main = axes_dict["main"]
        ax_stats = axes_dict["stats"]

        if xlim is None:
            xlim = ax_main.get_xlim()
        if ylim is None:
            ylim = ax_main.get_ylim()

        # Find data points in current view
        time_mask = (time_ms >= xlim[0]) & (time_ms <= xlim[1])
        force_mask = (force_data >= ylim[0]) & (force_data <= ylim[1])
        view_mask = time_mask & force_mask

        if np.any(view_mask):
            view_force = force_data[view_mask]
            duration_view = xlim[1] - xlim[0]

            stats_text = f"""CURRENT VIEW STATISTICS
{'=' * 30}
Time Range: {xlim[0]:.0f} - {xlim[1]:.0f} ms
Duration: {duration_view:.0f} ms ({duration_view / 1000:.2f} s)
Data Points: {np.sum(view_mask):,}

Force Statistics:
  Min: {np.min(view_force):.0f} N
  Max: {np.max(view_force):.0f} N
  Peak |F|: {np.max(np.abs(view_force)):.0f} N
  Mean: {np.mean(view_force):.0f} N
  RMS: {np.sqrt(np.mean(view_force**2)):.0f} N
  Std: {np.std(view_force):.0f} N

Zoom Level: {(time_ms[-1] - time_ms[0]) / duration_view:.1f}x"""
        else:
            stats_text = "No data points in current view"

        ax_stats.clear()
        ax_stats.axis("off")
        ax_stats.text(
            0.05,
            0.95,
            stats_text,
            transform=ax_stats.transAxes,
            verticalalignment="top",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    return update_stats


def _create_zoom_selector(
    axes_dict, time_ms: np.ndarray, force_data: np.ndarray, update_stats
):
    """Create the zoom selection function."""

    def onselect(xmin, xmax):
        ax_zoom = axes_dict["zoom"]
        ax_main = axes_dict["main"]
        line_zoom = axes_dict["line_zoom"]

        # Update zoom plot
        time_mask = (time_ms >= xmin) & (time_ms <= xmax)
        if np.any(time_mask):
            zoom_time = time_ms[time_mask]
            zoom_force = force_data[time_mask]

            line_zoom.set_data(zoom_time, zoom_force)
            ax_zoom.set_xlim(xmin, xmax)

            # Set y-limits with some padding
            force_min, force_max = np.min(zoom_force), np.max(zoom_force)
            force_range = force_max - force_min
            padding = force_range * 0.1 if force_range > 0 else abs(force_max) * 0.1
            ax_zoom.set_ylim(force_min - padding, force_max + padding)

            duration_ms = xmax - xmin
            ax_zoom.set_title(f"Zoomed View: {duration_ms:.0f} ms duration")

            # Also zoom main plot
            ax_main.set_xlim(xmin, xmax)

            # Update statistics for zoomed region
            update_stats((xmin, xmax))

            axes_dict["fig"].canvas.draw()

    return onselect


def _create_button_handlers(
    axes_dict, time_ms: np.ndarray, force_data: np.ndarray, update_stats
):
    """Create button event handlers."""

    def reset_zoom(event):
        ax_main = axes_dict["main"]
        ax_zoom = axes_dict["zoom"]
        original_xlim = axes_dict["original_xlim"]
        original_ylim = axes_dict["original_ylim"]

        ax_main.set_xlim(original_xlim)
        ax_main.set_ylim(original_ylim)
        ax_zoom.clear()
        ax_zoom.set_xlabel("Time (ms)", fontsize=10)
        ax_zoom.set_ylabel("Force (N)", fontsize=10)
        ax_zoom.set_title("Zoomed View (Select region in main plot)")
        ax_zoom.grid(True, alpha=0.3)
        (line_zoom,) = ax_zoom.plot([], [], "r-", linewidth=2, label="Zoomed Region")
        axes_dict["line_zoom"] = line_zoom
        ax_zoom.legend()
        update_stats()
        axes_dict["fig"].canvas.draw()

    def zoom_to_peak(event):
        peak_idx = np.argmax(np.abs(force_data))
        peak_time = time_ms[peak_idx]

        # Zoom to ±5 seconds around peak
        zoom_window = 5000  # 5 seconds in ms
        xmin = max(time_ms[0], peak_time - zoom_window)
        xmax = min(time_ms[-1], peak_time + zoom_window)

        onselect = _create_zoom_selector(axes_dict, time_ms, force_data, update_stats)
        onselect(xmin, xmax)

    return reset_zoom, zoom_to_peak


def _create_scroll_handler(axes_dict, time_ms: np.ndarray, update_stats):
    """Create mouse wheel scroll handler."""

    def on_scroll(event):
        ax_main = axes_dict["main"]

        if event.inaxes != ax_main:
            return

        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()

        zoom_factor = 1.5 if event.step < 0 else 1 / 1.5  # Faster zoom

        x_mouse = event.xdata
        y_mouse = event.ydata

        if x_mouse is None or y_mouse is None:
            return

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor

        new_xlim = [
            x_mouse - new_x_range * (x_mouse - xlim[0]) / x_range,
            x_mouse + new_x_range * (xlim[1] - x_mouse) / x_range,
        ]
        new_ylim = [
            y_mouse - new_y_range * (y_mouse - ylim[0]) / y_range,
            y_mouse + new_y_range * (ylim[1] - y_mouse) / y_range,
        ]

        ax_main.set_xlim(new_xlim)
        ax_main.set_ylim(new_ylim)
        update_stats(new_xlim, new_ylim)
        axes_dict["fig"].canvas.draw()

    return on_scroll


def _setup_interactive_controls(
    fig, axes_dict, time_ms: np.ndarray, force_data: np.ndarray
):
    """Set up all interactive controls and event handlers."""
    # Store figure reference
    axes_dict["fig"] = fig

    # Create update function
    update_stats = _create_stats_updater(axes_dict, time_ms, force_data)

    # Initialize statistics
    update_stats()

    # Create zoom selector
    onselect = _create_zoom_selector(axes_dict, time_ms, force_data, update_stats)

    # Create span selector (try/except for matplotlib version compatibility)
    try:
        from matplotlib.widgets import SpanSelector

        # Store span selector to prevent garbage collection
        axes_dict["span_selector"] = SpanSelector(
            axes_dict["main"],
            onselect,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="red"),
        )
    except TypeError:
        # Fall back to older matplotlib API
        axes_dict["span_selector"] = SpanSelector(
            axes_dict["main"], onselect, direction="horizontal", useblit=True
        )

    # Create button handlers
    reset_zoom, zoom_to_peak = _create_button_handlers(
        axes_dict, time_ms, force_data, update_stats
    )

    # Add buttons
    from matplotlib.widgets import Button

    button_axes = plt.axes([0.1, 0.02, 0.15, 0.06])
    button = Button(button_axes, "Reset Zoom", color="lightcoral", hovercolor="red")
    button.on_clicked(reset_zoom)

    peak_button_axes = plt.axes([0.3, 0.02, 0.15, 0.06])
    peak_button = Button(
        peak_button_axes, "Zoom to Peak", color="lightgreen", hovercolor="green"
    )
    peak_button.on_clicked(zoom_to_peak)

    # Connect scroll event
    on_scroll = _create_scroll_handler(axes_dict, time_ms, update_stats)
    fig.canvas.mpl_connect("scroll_event", on_scroll)


def _print_usage_instructions():
    """Print usage instructions to console."""
    print("\n🎛️ INTERACTIVE CONTROLS:")
    print("• Mouse wheel: Zoom in/out")
    print("• Click + drag on main plot: Select region to zoom")
    print("• Reset Zoom button: Return to full view")
    print("• Zoom to Peak button: Quick zoom to maximum force")
    print("• Close window when done exploring")


# =============================================================================
# LEGACY COMPATIBILITY (MINIMAL)
# =============================================================================


def create_summary_plots(results: List[Dict], output_dir: Union[str, Path]) -> None:
    """Legacy compatibility function - creates impulse-focused summary plots.

    Note: This function now focuses on impulse analysis results rather than
    kinetic energy. For KE-specific plots, use legacy.visualization module.
    """
    print("📊 Creating impulse-focused summary plots...")
    create_impulse_material_comparison(results, output_dir)
    create_impulse_summary_statistics(results, output_dir)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("📊 Fishing Line Flyback Impact Analysis - Visualization Module v3.0")
    print("=" * 70)
    print("This module provides impulse-focused visualization capabilities.")
    print()
    print("Key functions:")
    print("• show_force_preview() - Interactive force curve exploration")
    print("• plot_single_file_analysis() - Detailed single file analysis plots")
    print("• create_impulse_material_comparison() - Material comparison plots")
    print("• create_impulse_summary_statistics() - Statistical summaries")
    print()
    print("Usage examples:")
    print(">>> from visualization import show_force_preview, plot_single_file_analysis")
    print(">>> show_force_preview(force_data, time_data, 'test.csv')")
    print(">>> plot_single_file_analysis('data/STND-21-5.csv')")
    print()
    print("For legacy kinetic energy visualization, use:")
    print(">>> from legacy.visualization import ImpactVisualizer")
