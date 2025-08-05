"""
Impulse Analysis Module for Fishing Line Flyback Impact Analysis

This module provides impulse-based analysis (∫ F(t) dt) as an alternative
to kinetic energy estimation. It focuses on total momentum transfer,
which is more direct and relevant for fishing line performance evaluation.

Key Features:
- Direct momentum transfer measurement via ∫ F(t) dt
- Compatible with existing configuration weights and line mass
- Simple and robust analysis method
- More relevant to fishing line impact effectiveness
- NEW: Boundary validation plotting with --show-plot flag
- NEW: SI box plots in publication style
- FIXED: Font compatibility and seaborn warnings
"""

import platform
import warnings
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate
from scipy import stats
from scipy.signal import savgol_filter


# Configure matplotlib for cross-platform compatibility
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
else:
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 10


# Clear font cache to ensure changes take effect
def clear_matplotlib_cache():
    """Clear matplotlib font cache to pick up new font settings."""
    try:
        import os

        import matplotlib as mpl

        cache_dir = mpl.get_cachedir()
        cache_files = ["fontlist-v330.json", "fontlist-v320.json", "fontlist-v310.json"]
        for cache_file in cache_files:
            cache_path = os.path.join(cache_dir, cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"Cleared font cache: {cache_file}")
    except Exception as e:
        print(f"Could not clear font cache: {e}")


# Call this once when module loads
clear_matplotlib_cache()

# Import existing project components
from .analysis import CONFIG_WEIGHTS
from .analysis import LINE_MASS_FRACTION
from .analysis import get_system_mass


class ImpulseAnalyzer:
    """
    Impulse-based analyzer that integrates with the existing project structure.

    Focuses on total momentum transfer: Impulse = ∫ F(t) dt
    This provides a cleaner, more direct measurement of fishing line
    impact effectiveness compared to kinetic energy estimation.
    """

    def __init__(
        self,
        material_code: Optional[str] = None,
        include_line_mass: bool = True,
        line_mass_fraction: float = LINE_MASS_FRACTION,
        sampling_rate: float = 100000.0,
        impact_threshold_factor: float = 0.02,
    ):
        """
        Initialize impulse analyzer with project configuration.

        Args:
            material_code: Configuration code (STND, DF, DS, SL, BR)
            include_line_mass: Whether to include line mass (for compatibility)
            line_mass_fraction: Line mass fraction (for compatibility)
            sampling_rate: Data acquisition sampling rate in Hz
            impact_threshold_factor: Fraction of peak force to define impact boundaries
        """
        self.material_code = material_code
        self.include_line_mass = include_line_mass
        self.line_mass_fraction = line_mass_fraction
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.threshold_factor = impact_threshold_factor

        # Get mass information for compatibility with existing system
        if material_code:
            self.mass_info = get_system_mass(
                material_code, include_line_mass, line_mass_fraction
            )
            self.total_mass = self.mass_info["total_system_mass_kg"]
        else:
            self.mass_info = None
            self.total_mass = None

    def _detect_force_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect AI sensor force columns (compatible with existing method)."""
        force_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if ("ai" in col_lower and "lbf" in col_lower) or ("force" in col_lower):
                force_columns.append(col)
        return force_columns

    def _convert_lbf_to_n(self, force_lbf: np.ndarray) -> np.ndarray:
        """Convert force from pounds-force to Newtons."""
        return force_lbf * 4.44822

    def _apply_baseline_correction(self, force: np.ndarray) -> np.ndarray:
        """Apply baseline correction (compatible with existing method)."""
        if len(force) < 100:
            return force
        baseline_window = min(1000, len(force) // 10)
        baseline = np.median(force[:baseline_window])
        return force - baseline

    def _calculate_total_force(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate total force from CSV data (compatible with existing method).

        Args:
            df: DataFrame with sensor data

        Returns:
            Tuple of (total_force_in_N, force_column_names)
        """
        force_columns = self._detect_force_columns(df)

        if not force_columns:
            # Fallback: find numeric columns that aren't time
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            time_cols = [col for col in numeric_cols if "time" in col.lower()]
            force_columns = [col for col in numeric_cols if col not in time_cols]

            if not force_columns:
                raise ValueError("No force columns detected in CSV file")

        # Sum all force columns (in lbf)
        total_force_lbf = df[force_columns].sum(axis=1).values

        # Convert to Newtons and apply corrections
        total_force_n = self._convert_lbf_to_n(total_force_lbf)
        total_force_n = np.nan_to_num(total_force_n, nan=0.0)
        total_force_n = self._apply_baseline_correction(total_force_n)

        return total_force_n, force_columns

    def _get_time_array(self, df: pd.DataFrame, force_length: int) -> np.ndarray:
        """Get time array (compatible with existing method)."""
        time_cols = [col for col in df.columns if "time" in col.lower()]

        if time_cols:
            time_array = df[time_cols[0]].values[:force_length]
            if len(time_array) > 0:
                time_array = time_array - time_array[0]

                # Update sampling rate if time data available
                if len(time_array) > 10:
                    dt_values = np.diff(time_array)
                    actual_dt = np.median(dt_values[dt_values > 0])

                    if actual_dt > 0:
                        self.dt = actual_dt
                        self.sampling_rate = 1.0 / actual_dt
        else:
            time_array = np.arange(force_length) * self.dt

        return time_array

    def plot_impulse_boundaries(
        self,
        force: np.ndarray,
        time: np.ndarray,
        impact_start: int,
        impact_end: int,
        filename: str = "unknown",
    ):
        """
        Plot force data with integration boundaries for validation (auto-scaling).

        Args:
            force: Force array (N)
            time: Time array (s)
            impact_start: Start index of impact region
            impact_end: End index of impact region
            filename: Filename for plot title
        """
        # Auto-detect screen size and scale accordingly
        try:
            import tkinter as tk

            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            # Calculate appropriate figure size (use 80% of screen)
            max_width = screen_width * 0.8 / 100  # Convert pixels to inches (approx)
            max_height = screen_height * 0.8 / 100

            # Set reasonable bounds
            fig_width = min(max(8, max_width), 20)  # Between 8-20 inches
            fig_height = min(max(6, max_height), 12)  # Between 6-12 inches

        except:
            # Fallback if screen detection fails
            fig_width, fig_height = 12, 8

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
        fig.suptitle(
            f"Impulse Integration Boundary Validation: {filename}",
            fontsize=max(10, min(16, fig_width)),
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )

        # Top plot: Full force trace with boundaries
        ax1.plot(time, force, "b-", linewidth=1, alpha=0.7, label="Force")

        # Highlight integration region
        if impact_start < len(time) and impact_end < len(time):
            integration_time = time[impact_start : impact_end + 1]
            integration_force = force[impact_start : impact_end + 1]
            ax1.fill_between(
                integration_time,
                integration_force,
                alpha=0.3,
                color="green",
                label=f"Integration Region ({(impact_end - impact_start) * self.dt * 1000:.1f} ms)",
            )

            # Mark boundaries
            ax1.axvline(
                time[impact_start],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Start: {time[impact_start]:.4f} s",
            )
            ax1.axvline(
                time[impact_end],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"End: {time[impact_end]:.4f} s",
            )

        # Mark peak
        peak_idx = np.argmax(np.abs(force))
        ax1.axvline(
            time[peak_idx],
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"Peak: {np.max(np.abs(force)):.0f} N",
        )

        ax1.set_xlabel("Time (s)", fontfamily="DejaVu Sans")
        ax1.set_ylabel("Force (N)", fontfamily="DejaVu Sans")
        ax1.set_title(
            "Full Force Trace with Integration Boundaries", fontfamily="DejaVu Sans"
        )
        ax1.legend(fontsize=max(8, min(10, fig_width * 0.8)))
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Zoomed view around integration region
        if impact_start < len(time) and impact_end < len(time):
            # Add padding
            padding = max(100, (impact_end - impact_start) // 5)
            zoom_start = max(0, impact_start - padding)
            zoom_end = min(len(force) - 1, impact_end + padding)

            zoom_time = time[zoom_start : zoom_end + 1]
            zoom_force = force[zoom_start : zoom_end + 1]

            ax2.plot(zoom_time, zoom_force, "b-", linewidth=2, label="Force")
            ax2.fill_between(
                integration_time,
                integration_force,
                alpha=0.3,
                color="green",
                label="Integration Region",
            )
            ax2.axvline(time[impact_start], color="red", linestyle="--", linewidth=2)
            ax2.axvline(time[impact_end], color="red", linestyle="--", linewidth=2)

            ax2.set_xlabel("Time (s)", fontfamily="DejaVu Sans")
            ax2.set_ylabel("Force (N)", fontfamily="DejaVu Sans")
            ax2.set_title("Zoomed View of Integration Region", fontfamily="DejaVu Sans")
            ax2.legend(fontsize=max(8, min(10, fig_width * 0.8)))
            ax2.grid(True, alpha=0.3)

            # Add statistics
            integration_impulse = integrate.trapezoid(integration_force, dx=self.dt)
            duration_ms = (impact_end - impact_start) * self.dt * 1000

            stats_text = f"""
Duration: {duration_ms:.1f} ms
Samples: {impact_end - impact_start + 1}
Impulse: {integration_impulse:+.6f} N⋅s
Peak: {np.max(np.abs(integration_force)):.1f} N
            """

            ax2.text(
                0.02,
                0.98,
                stats_text.strip(),
                transform=ax2.transAxes,
                verticalalignment="top",
                fontfamily="monospace",
                fontsize=max(8, min(10, fig_width * 0.8)),
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
            )

        # Auto-adjust layout and show
        plt.tight_layout()

        # Try to maximize window if possible
        try:
            mng = plt.get_current_fig_manager()
            if hasattr(mng, "window"):
                if hasattr(mng.window, "wm_state"):
                    mng.window.wm_state("zoomed")  # Windows
                elif hasattr(mng.window, "showMaximized"):
                    mng.window.showMaximized()  # Qt backends
                elif hasattr(mng, "full_screen_toggle"):
                    pass  # Don't auto-maximize for some backends
        except:
            pass  # Ignore if window management fails

        plt.show()

        # Print validation guidance
        if impact_start < len(time) and impact_end < len(time):
            duration_ms = (impact_end - impact_start) * self.dt * 1000
            print(f"\n📊 BOUNDARY VALIDATION RESULTS:")
            print(f"   Integration duration: {duration_ms:.1f} ms")
            print(
                f"   Percentage of total data: {(impact_end - impact_start + 1) / len(force) * 100:.1f}%"
            )

            if 2 <= duration_ms <= 50:
                print(f"   ✅ Duration looks reasonable for fishing line impact")
            elif duration_ms < 2:
                print(f"   ⚠️  Duration very short - may be missing impact data")
                print(f"   💡 Try lower threshold: impact_threshold_factor=0.005")
            elif duration_ms > 100:
                print(f"   ❌ Duration very long - likely includes rebound/noise")
                print(f"   💡 Try higher threshold: impact_threshold_factor=0.05")
            else:
                print(f"   🟡 Duration borderline - review plot carefully")

    def find_impact_boundaries(
        self,
        force: np.ndarray,
        debug: bool = False,
        show_plot: bool = False,
        filename: str = "unknown",
    ) -> Tuple[int, int]:
        """
        Detect complete impact spike from rise above baseline to full return to baseline.
        No arbitrary duration limits - follows the physics of the impact.
        """

        # STEP 1: Apply smoothing to reduce noise
        try:
            from scipy.signal import savgol_filter

            window_length = min(21, len(force) // 10)
            if window_length % 2 == 0:
                window_length += 1
            if window_length < 5:
                window_length = 5

            force_smooth = savgol_filter(force, window_length, 3)

            if debug:
                print(
                    f"   🔧 Applied Savitzky-Golay smoothing (window={window_length})"
                )
        except:
            # Fallback: simple moving average
            window = 7
            force_smooth = np.convolve(force, np.ones(window) / window, mode="same")
            if debug:
                print(f"   🔧 Applied moving average smoothing (window={window})")

        peak_force = np.max(np.abs(force_smooth))
        peak_idx = np.argmax(np.abs(force_smooth))

        if debug:
            print(f"   🎯 Complete return-to-baseline detection:")
            print(
                f"      Peak: {peak_force:.1f} N at index {peak_idx} ({peak_idx * self.dt:.4f} s)"
            )

        # STEP 2: Calculate true baseline from quiet regions
        # Use first and last 5% of data to get baseline
        baseline_samples = max(
            100, len(force) // 20
        )  # At least 100 samples, or 5% of data

        baseline_start_region = force[:baseline_samples]
        baseline_end_region = force[-baseline_samples:]

        # Use median for robustness against outliers
        baseline_start = np.median(baseline_start_region)
        baseline_end = np.median(baseline_end_region)
        baseline = (baseline_start + baseline_end) / 2

        # Calculate noise level in baseline regions to set thresholds
        noise_level_start = np.std(baseline_start_region - baseline_start)
        noise_level_end = np.std(baseline_end_region - baseline_end)
        noise_level = max(noise_level_start, noise_level_end)

        # Set threshold as 3x noise level (statistically significant)
        significance_threshold = max(
            3 * noise_level, peak_force * 0.01
        )  # At least 1% of peak

        if debug:
            print(f"      Baseline: {baseline:.1f} N")
            print(f"      Noise level: {noise_level:.1f} N")
            print(f"      Significance threshold: {significance_threshold:.1f} N")

        # STEP 3: Find impact START - first significant rise above baseline
        impact_start = 0
        for i in range(
            max(0, peak_idx - 5000), peak_idx
        ):  # Look back further if needed
            if abs(force_smooth[i] - baseline) > significance_threshold:
                # Found first significant rise - look back for the actual start
                for j in range(i, max(0, i - 200), -1):  # Look back up to 200 samples
                    if abs(force_smooth[j] - baseline) <= significance_threshold * 0.5:
                        impact_start = j
                        break
                else:
                    impact_start = max(0, i - 100)  # Fallback
                break

        # STEP 4: Find impact END - complete return to baseline
        impact_end = len(force) - 1  # Default to end of data

        # Start looking after the peak
        search_start = peak_idx + 50  # Give some buffer after peak

        for i in range(search_start, len(force)):
            # Check if we're back to baseline
            force_deviation = abs(force_smooth[i] - baseline)

            if force_deviation <= significance_threshold:
                # Found potential end - verify it stays at baseline
                samples_to_check = min(
                    100, len(force) - i
                )  # Check next 100 samples or to end

                if samples_to_check < 20:  # If near end of data, accept it
                    impact_end = i
                    break

                # Check if force stays near baseline for the next samples
                baseline_count = 0
                for j in range(i, min(len(force), i + samples_to_check)):
                    if abs(force_smooth[j] - baseline) <= significance_threshold * 1.5:
                        baseline_count += 1

                # If at least 70% of the next samples are at baseline, we're done
                if baseline_count / samples_to_check >= 0.7:
                    impact_end = i
                    if debug:
                        print(f"      Found stable return to baseline at index {i}")
                    break

        # STEP 5: Final validation (but don't impose arbitrary limits)
        duration_ms = (impact_end - impact_start) * self.dt * 1000

        # Only impose limits if truly unreasonable (> 1 second suggests something is wrong)
        if duration_ms > 1000:
            if debug:
                print(
                    f"   ⚠️  Duration very long ({duration_ms:.1f} ms), may include multiple events"
                )
            # In this case, limit to a reasonable window around peak
            max_samples = int(0.2 / self.dt)  # 200ms max
            impact_end = min(impact_end, impact_start + max_samples)
            duration_ms = (impact_end - impact_start) * self.dt * 1000

        # Ensure minimum duration
        if duration_ms < 1:
            min_samples = int(0.001 / self.dt)  # 1ms minimum
            impact_end = max(impact_end, impact_start + min_samples)
            duration_ms = (impact_end - impact_start) * self.dt * 1000

        if debug:
            print(f"   📏 Complete impact boundaries:")
            print(
                f"      Start: index {impact_start}, time {impact_start * self.dt:.4f} s"
            )
            print(f"      End: index {impact_end}, time {impact_end * self.dt:.4f} s")
            print(f"      Duration: {duration_ms:.1f} ms")
            print(
                f"      Force at start: {force[impact_start]:.1f} N (deviation: {abs(force[impact_start] - baseline):.1f} N)"
            )
            print(
                f"      Force at end: {force[impact_end]:.1f} N (deviation: {abs(force[impact_end] - baseline):.1f} N)"
            )
            print(f"      Peak force: {force[peak_idx]:.1f} N")

        if show_plot:
            time_array = np.arange(len(force)) * self.dt
            self.plot_impulse_boundaries_with_smoothing(
                force, force_smooth, time_array, impact_start, impact_end, filename
            )

        return impact_start, impact_end

    def plot_impulse_boundaries_with_smoothing(
        self,
        force_orig: np.ndarray,
        force_smooth: np.ndarray,
        time: np.ndarray,
        impact_start: int,
        impact_end: int,
        filename: str = "unknown",
    ):
        """
        Plot force data showing both original and smoothed signals with boundaries.
        """
        # Auto-detect screen size and scale accordingly
        try:
            import tkinter as tk

            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            # Calculate appropriate figure size (use 80% of screen)
            max_width = screen_width * 0.8 / 100
            max_height = screen_height * 0.8 / 100

            fig_width = min(max(8, max_width), 20)
            fig_height = min(max(6, max_height), 12)
        except:
            fig_width, fig_height = 12, 8

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
        fig.suptitle(
            f"Impulse Integration with Signal Smoothing: {filename}",
            fontsize=max(10, min(16, fig_width)),
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )

        # Top plot: Full trace with both original and smoothed
        ax1.plot(
            time, force_orig, "b-", linewidth=0.5, alpha=0.6, label="Original Force"
        )
        ax1.plot(
            time, force_smooth, "r-", linewidth=1.5, alpha=0.8, label="Smoothed Force"
        )

        # Highlight integration region
        if impact_start < len(time) and impact_end < len(time):
            integration_time = time[impact_start : impact_end + 1]
            integration_force = force_orig[
                impact_start : impact_end + 1
            ]  # Use original for integration
            ax1.fill_between(
                integration_time,
                integration_force,
                alpha=0.3,
                color="green",
                label=f"Integration Region ({(impact_end - impact_start) * self.dt * 1000:.1f} ms)",
            )

            # Mark boundaries
            ax1.axvline(
                time[impact_start],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Start: {time[impact_start]:.4f} s",
            )
            ax1.axvline(
                time[impact_end],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"End: {time[impact_end]:.4f} s",
            )

        # Mark peak
        peak_idx = np.argmax(np.abs(force_orig))
        ax1.axvline(
            time[peak_idx],
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"Peak: {np.max(np.abs(force_orig)):.0f} N",
        )

        ax1.set_xlabel("Time (s)", fontfamily="DejaVu Sans")
        ax1.set_ylabel("Force (N)", fontfamily="DejaVu Sans")
        ax1.set_title(
            "Full Force Trace (Original + Smoothed) with Integration Boundaries",
            fontfamily="DejaVu Sans",
        )
        ax1.legend(fontsize=max(8, min(10, fig_width * 0.8)))
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Zoomed view of integration region
        if impact_start < len(time) and impact_end < len(time):
            # Add padding
            padding = max(100, (impact_end - impact_start) // 3)
            zoom_start = max(0, impact_start - padding)
            zoom_end = min(len(force_orig) - 1, impact_end + padding)

            zoom_time = time[zoom_start : zoom_end + 1]
            zoom_force_orig = force_orig[zoom_start : zoom_end + 1]
            zoom_force_smooth = force_smooth[zoom_start : zoom_end + 1]

            ax2.plot(
                zoom_time,
                zoom_force_orig,
                "b-",
                linewidth=1,
                alpha=0.7,
                label="Original Force",
            )
            ax2.plot(
                zoom_time,
                zoom_force_smooth,
                "r-",
                linewidth=2,
                alpha=0.8,
                label="Smoothed Force",
            )
            ax2.fill_between(
                integration_time,
                integration_force,
                alpha=0.3,
                color="green",
                label="Integration Region",
            )
            ax2.axvline(time[impact_start], color="red", linestyle="--", linewidth=2)
            ax2.axvline(time[impact_end], color="red", linestyle="--", linewidth=2)

            ax2.set_xlabel("Time (s)", fontfamily="DejaVu Sans")
            ax2.set_ylabel("Force (N)", fontfamily="DejaVu Sans")
            ax2.set_title(
                "Zoomed View: Full Impact Spike (Start to Return to Baseline)",
                fontfamily="DejaVu Sans",
            )
            ax2.legend(fontsize=max(8, min(10, fig_width * 0.8)))
            ax2.grid(True, alpha=0.3)

            # Add statistics
            integration_impulse = integrate.trapezoid(integration_force, dx=self.dt)
            duration_ms = (impact_end - impact_start) * self.dt * 1000

            stats_text = f"""
Duration: {duration_ms:.1f} ms
Samples: {impact_end - impact_start + 1}
Impulse: {integration_impulse:+.6f} N⋅s
Peak: {np.max(np.abs(integration_force)):.1f} N
Start Force: {force_orig[impact_start]:.1f} N
End Force: {force_orig[impact_end]:.1f} N
            """

            ax2.text(
                0.02,
                0.98,
                stats_text.strip(),
                transform=ax2.transAxes,
                verticalalignment="top",
                fontfamily="monospace",
                fontsize=max(8, min(10, fig_width * 0.8)),
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
            )

        plt.tight_layout()
        plt.show()

    def calculate_impulse_metrics(
        self,
        force: np.ndarray,
        time: np.ndarray,
        debug: bool = False,
        show_plot: bool = False,
        filename: str = "unknown",
    ) -> Dict:
        """
        Calculate comprehensive impulse metrics.

        Args:
            force: Force array (N)
            time: Time array (s)
            debug: Print debug information
            show_plot: Show boundary validation plot
            filename: Filename for plot title

        Returns:
            Dictionary of impulse metrics
        """
        if debug:
            print(f"\n🎯 IMPULSE ANALYSIS:")
            print(f"   Data points: {len(force):,}")
            print(f"   Sampling rate: {self.sampling_rate:.0f} Hz")
            print(f"   Duration: {time[-1]:.4f} s")
            print(f"   Force range: {np.min(force):.1f} to {np.max(force):.1f} N")
            if self.material_code:
                print(f"   Material: {self.material_code}")
                print(f"   Total mass: {self.total_mass*1000:.1f}g")

        # Find impact boundaries with optional plotting
        impact_start, impact_end = self.find_impact_boundaries(
            force, debug, show_plot, filename
        )
        impact_force = force[impact_start : impact_end + 1]
        impact_time = time[impact_start : impact_end + 1]
        impact_duration = (impact_end - impact_start + 1) * self.dt

        # PRIMARY METRIC: Total Impulse = ∫ F(t) dt
        total_impulse = integrate.trapezoid(force, dx=self.dt)
        impact_impulse = integrate.trapezoid(impact_force, dx=self.dt)

        # Energy-like metrics
        total_abs_impulse = integrate.trapezoid(np.abs(force), dx=self.dt)
        impact_abs_impulse = integrate.trapezoid(np.abs(impact_force), dx=self.dt)

        # Force characteristics
        peak_force = np.max(np.abs(force))
        peak_force_positive = np.max(force) if len(force) > 0 else 0
        peak_force_negative = np.min(force) if len(force) > 0 else 0

        # Derived metrics
        rms_force = np.sqrt(np.mean(force**2)) if len(force) > 0 else 0

        # For compatibility with kinetic energy analysis
        if self.total_mass and self.total_mass > 0:
            # Equivalent velocity from impulse
            equivalent_velocity = abs(total_impulse) / self.total_mass
            # Equivalent kinetic energy
            equivalent_kinetic_energy = 0.5 * self.total_mass * equivalent_velocity**2
        else:
            equivalent_velocity = np.nan
            equivalent_kinetic_energy = np.nan

        if debug:
            print(f"\n   📊 IMPULSE RESULTS:")
            print(f"      Total impulse: {total_impulse:+.6f} N⋅s")
            print(f"      Total abs impulse: {total_abs_impulse:.6f} N⋅s")
            print(f"      Impact impulse: {impact_impulse:+.6f} N⋅s")
            print(f"      Peak force: {peak_force:.1f} N")
            print(f"      Impact duration: {impact_duration*1000:.1f} ms")
            if not np.isnan(equivalent_velocity):
                print(f"      Equivalent velocity: {equivalent_velocity:.1f} m/s")
                print(f"      Equivalent KE: {equivalent_kinetic_energy:.3f} J")

        return {
            # Primary impulse metrics
            "total_impulse": total_impulse,
            "total_abs_impulse": total_abs_impulse,
            "impact_impulse": impact_impulse,
            "impact_abs_impulse": impact_abs_impulse,
            # Force characteristics
            "peak_force": peak_force,
            "peak_force_positive": peak_force_positive,
            "peak_force_negative": peak_force_negative,
            "rms_force": rms_force,
            # Timing
            "impact_duration": impact_duration,
            "impact_start_time": time[impact_start] if len(time) > impact_start else 0,
            "impact_end_time": time[impact_end] if len(time) > impact_end else 0,
            "total_duration": time[-1] - time[0] if len(time) > 0 else 0,
            # Compatibility with existing analysis
            "equivalent_velocity": equivalent_velocity,
            "equivalent_kinetic_energy": equivalent_kinetic_energy,
            "mass_kg": self.total_mass if self.total_mass else np.nan,
            # Analysis metadata
            "impact_start_idx": impact_start,
            "impact_end_idx": impact_end,
            "sampling_rate_hz": self.sampling_rate,
            "analysis_method": "impulse_integration",
            "material_code": self.material_code,
            "mass_breakdown": self.mass_info,
        }

    def analyze_csv_file(
        self, csv_path: Union[str, Path], show_plot: bool = False
    ) -> Dict:
        """
        Analyze a single CSV file (compatible with existing interface).

        Args:
            csv_path: Path to CSV file
            show_plot: Show boundary validation plot

        Returns:
            Analysis results dictionary
        """
        csv_path = Path(csv_path)

        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Calculate total force
            force_n, force_columns = self._calculate_total_force(df)

            # Get time array
            time = self._get_time_array(df, len(force_n))

            # Quality checks
            if len(force_n) < 50:
                raise ValueError("Insufficient data points for analysis")

            # Remove invalid values
            valid_indices = np.isfinite(force_n)
            force_n = force_n[valid_indices]
            time = time[valid_indices]

            if len(force_n) < 25:
                raise ValueError("Too many invalid data points")

            # Calculate impulse metrics with optional plotting
            results = self.calculate_impulse_metrics(
                force_n, time, show_plot=show_plot, filename=csv_path.name
            )

            # Add metadata (compatible with existing format)
            results.update(
                {
                    "filename": csv_path.name,
                    "material_type": self._extract_material_type(csv_path.name),
                    "sample_number": self._extract_sample_number(csv_path.name),
                    "force_columns_used": ", ".join(force_columns),
                    "baseline_correction": True,
                    "line_mass_included": self.include_line_mass,
                    "line_mass_fraction": self.line_mass_fraction,
                    # For compatibility - map impulse metrics to existing names
                    "kinetic_energy": results["equivalent_kinetic_energy"],
                    "initial_velocity": results["equivalent_velocity"],
                    "max_force": results["peak_force"],
                }
            )

            return results

        except Exception as e:
            return {
                "filename": csv_path.name,
                "material_type": self._extract_material_type(csv_path.name),
                "sample_number": self._extract_sample_number(csv_path.name),
                "error": str(e),
                "analysis_method": "impulse_integration",
            }

    def _extract_material_type(self, filename: str) -> str:
        """Extract material type from filename."""
        try:
            return filename.split("-")[0]
        except:
            return "UNKNOWN"

    def _extract_sample_number(self, filename: str) -> str:
        """Extract sample number from filename."""
        try:
            parts = filename.replace(".csv", "").split("-")
            return "-".join(parts[1:])
        except:
            return "UNKNOWN"


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def analyze_single_file_with_impulse(
    file_path: Union[str, Path],
    material_code: Optional[str] = None,
    include_line_mass: bool = True,
    line_mass_fraction: float = LINE_MASS_FRACTION,
    show_plot: bool = False,
) -> Dict:
    """
    Analyze single file with impulse method (compatible with existing interface).

    Args:
        file_path: Path to CSV file
        material_code: Configuration code
        include_line_mass: Whether to include line mass
        line_mass_fraction: Line mass fraction
        show_plot: Show boundary validation plot

    Returns:
        Analysis results dictionary
    """
    file_path = Path(file_path)

    if material_code is None:
        material_code = file_path.name.split("-")[0]

    analyzer = ImpulseAnalyzer(
        material_code=material_code,
        include_line_mass=include_line_mass,
        line_mass_fraction=line_mass_fraction,
    )

    return analyzer.analyze_csv_file(file_path, show_plot=show_plot)


def run_impulse_analysis(
    data_dir: Union[str, Path] = "data/csv",
    output_dir: Union[str, Path] = "impulse_analysis",
) -> List[Dict]:
    """
    Run impulse analysis on all files (compatible with existing workflow).

    Args:
        data_dir: Directory containing CSV files
        output_dir: Output directory for results

    Returns:
        List of analysis results
    """
    print("🎯 IMPULSE-BASED FISHING LINE ANALYSIS")
    print("=" * 80)
    print("Method: Total momentum transfer via ∫ F(t) dt")
    print("Focus: Direct measurement of impact effectiveness")
    print()

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all CSV files
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {data_path}")
        return []

    print(f"📁 Processing {len(csv_files)} files from {data_path}")
    print(f"📊 Results will be saved to {output_path}")
    print()

    # Show configuration weights for reference
    print(f"⚖️  CONFIGURATION REFERENCE:")
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        line_mass_kg = 0.0388 * LINE_MASS_FRACTION
        total_kg = weight_kg + line_mass_kg
        print(
            f"   {config:4s}: {weight_kg*1000:2.0f}g + {line_mass_kg*1000:.0f}g = {total_kg*1000:.0f}g total"
        )
    print()

    # Process files
    results = []
    material_counts = {}

    print(f"🔄 PROCESSING FILES:")
    print("-" * 80)
    print(
        f"{'Status':<2} {'Filename':<20} {'Mat':<4} {'Total Impulse':<15} {'Peak Force':<12} {'Duration'}"
    )
    print("-" * 80)

    for file_path in csv_files:
        try:
            material = file_path.name.split("-")[0]

            analyzer = ImpulseAnalyzer(
                material_code=material,
                include_line_mass=True,
                line_mass_fraction=LINE_MASS_FRACTION,
            )

            result = analyzer.analyze_csv_file(file_path)

            if "error" not in result:
                impulse = result["total_impulse"]
                peak_force = result["peak_force"]
                duration = result["impact_duration"] * 1000  # ms

                material_counts[material] = material_counts.get(material, 0) + 1

                print(
                    f"✅ {file_path.name:<20} {material:<4} {impulse:>+12.6f} N⋅s "
                    f"{peak_force:>8.0f} N   {duration:>6.1f} ms"
                )
                results.append(result)
            else:
                print(f"❌ {file_path.name:<20} {material:<4} ERROR: {result['error']}")

        except Exception as e:
            material = file_path.name.split("-")[0] if "-" in file_path.name else "UNK"
            print(f"❌ {file_path.name:<20} {material:<4} ERROR: {str(e)}")

    print("-" * 80)

    if results:
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_path / "impulse_results.csv", index=False)

        # Create analysis report
        create_impulse_report(results, output_path)

        # Create visualizations
        create_impulse_visualizations(results, output_path)

        print(f"\n🎉 IMPULSE ANALYSIS COMPLETE!")
        print(f"   Successfully analyzed: {len(results)} files")
        print(f"   Material distribution: {material_counts}")
        print(f"📁 All results saved to: {output_path}")
    else:
        print("❌ No successful analyses to report")

    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_impulse_force_box_si(
    df: pd.DataFrame, output_dir: Union[str, Path], units: str = "SI", dpi: int = 600
) -> None:
    """
    Create horizontal box plots for impulse and force in publication style with background overlay.

    Args:
        df: DataFrame with impulse analysis results
        output_dir: Output directory for plots
        units: "SI" (N⋅s, N), "kSI" (kN⋅s, kN), or "mixed" (N⋅s, kN)
        dpi: Plot resolution
    """
    # Ensure we're using DejaVu Sans
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Material name mapping
    material_names = {
        "BR": "Breakaway",
        "DF": "Dual Fixed",
        "DS": "Dual Sliding",
        "SL": "Sliding",
        "STND": "Standard",
    }

    # Create material_name column
    df_plot = df.copy()
    df_plot["material_name"] = df_plot["material_type"].map(material_names)
    df_plot = df_plot.dropna(subset=["material_name"])

    # Define colors for each material
    material_colors = {
        "Breakaway": "#1f77b4",  # Blue (BR)
        "Dual Fixed": "#ff7f0e",  # Orange (DF)
        "Dual Sliding": "#2ca02c",  # Green (DS)
        "Sliding": "#d62728",  # Red (SL)
        "Standard": "#9467bd",  # Purple (STND)
    }

    # Unit conversions and labels
    if units == "SI":
        impulse_data = df_plot["total_abs_impulse"]
        force_data = df_plot["peak_force"]
        impulse_label = "Impulse Magnitude (N⋅s)"
        force_label = "Peak Force (N)"
        suffix = "SI"
    elif units == "kSI":
        impulse_data = df_plot["total_abs_impulse"] / 1000
        force_data = df_plot["peak_force"] / 1000
        df_plot["total_abs_impulse"] = df_plot["total_abs_impulse"] / 1000
        df_plot["peak_force"] = df_plot["peak_force"] / 1000
        impulse_label = "Impulse Magnitude (kN⋅s)"
        force_label = "Peak Force (kN)"
        suffix = "kSI"
    elif units == "mixed":
        impulse_data = df_plot["total_abs_impulse"]
        force_data = df_plot["peak_force"] / 1000
        df_plot["peak_force"] = df_plot["peak_force"] / 1000
        impulse_label = "Impulse Magnitude (N⋅s)"
        force_label = "Peak Force (kN)"
        suffix = "mixed"
    else:
        raise ValueError("units must be 'SI', 'kSI', or 'mixed'")

    # Define material order and create ordered categorical
    material_code_order = [
        "Standard",
        "Sliding",
        "Dual Fixed",
        "Dual Sliding",
        "Breakaway",
    ]
    existing_materials = [
        mat for mat in material_code_order if mat in df_plot["material_name"].values
    ]
    df_plot["material_name"] = pd.Categorical(
        df_plot["material_name"], categories=existing_materials, ordered=True
    )
    custom_palette = [material_colors.get(mat, "#1f77b4") for mat in existing_materials]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    # Try to load background overlay image
    overlay_path = output_path / "plot-box-J-F-overlay.svg"
    if overlay_path.exists():
        try:
            # If you have cairosvg installed, convert SVG to PNG first
            try:
                import cairosvg

                png_overlay_path = output_path / "plot-box-J-F-overlay.png"
                cairosvg.svg2png(
                    url=str(overlay_path), write_to=str(png_overlay_path), dpi=dpi
                )

                # Load the PNG and set as figure background
                import matplotlib.image as mpimg

                img = mpimg.imread(png_overlay_path)

                # Add background image to figure
                fig.figimage(img, xo=0, yo=0, alpha=1.0, zorder=-1)
                print(f"  📸 Loaded background overlay: {overlay_path.name}")

            except ImportError:
                print(
                    "  ⚠️  cairosvg not available - install with: pip install cairosvg"
                )

        except Exception as e:
            print(f"  ❌ Could not load overlay: {e}")

    plt.subplots_adjust(wspace=1.0)

    flier_props = dict(
        marker="o",
        markerfacecolor="slategray",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=1,
        alpha=0.8,
    )

    # Create box plots with transparency to show background
    sns.boxplot(
        data=df_plot,
        y="material_name",
        x="total_abs_impulse",
        ax=ax1,
        palette=custom_palette,
        linewidth=2,
        order=existing_materials,
        hue="material_name",
        legend=False,
        flierprops=flier_props,
    )

    ax1.set_xlabel(impulse_label, fontsize=24, fontfamily="DejaVu Sans")
    ax1.set_ylabel(" ", fontsize=24, fontfamily="DejaVu Sans")
    ax1.tick_params(axis="both", which="major", labelsize=20)
    ax1.grid(True, alpha=0.3)

    # Make plot background transparent so overlay shows through
    ax1.patch.set_alpha(0.0)

    # Force box plot
    sns.boxplot(
        data=df_plot,
        y="material_name",
        x="peak_force",
        ax=ax2,
        palette=custom_palette,
        linewidth=2,
        order=existing_materials,
        hue="material_name",
        legend=False,
        flierprops=flier_props,
    )

    ax2.set_xlabel(force_label, fontsize=24, fontfamily="DejaVu Sans")
    ax2.set_ylabel(" ", fontsize=24, fontfamily="DejaVu Sans")
    ax2.set_yticklabels([])
    ax2.tick_params(axis="x", which="major", labelsize=20, left=False, labelleft=False)
    ax2.grid(True, alpha=0.3)

    # Make plot background transparent so overlay shows through
    ax2.patch.set_alpha(0.0)

    plt.tight_layout()
    plt.subplots_adjust(wspace=1.0)

    # Save with high quality
    svg_path = output_path / f"plot-box-J-F-{suffix}-with-overlay.svg"
    png_path = output_path / f"plot-box-J-F-{suffix}-with-overlay.png"

    plt.savefig(
        svg_path,
        format="svg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.savefig(
        png_path,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )

    plt.close()

    print(f"  ✅ Saved: {svg_path.name} and {png_path.name}")


def create_impulse_visualizations(results: List[Dict], output_dir: Path):
    """Create individual impulse visualization plots as separate files."""

    # Ensure we're using DejaVu Sans for all plots
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return

    df = pd.DataFrame(valid_results)

    # Define consistent material order
    material_code_order = ["BR", "DS", "DF", "SL", "STND"]

    if "material_type" in df.columns:
        # Only include materials that exist in your data and set categorical order
        existing_materials = [
            m for m in material_code_order if m in df["material_type"].values
        ]
        print(existing_materials)
        if existing_materials:
            df["material_type"] = pd.Categorical(
                df["material_type"], categories=existing_materials, ordered=True
            )
            df = df.sort_values("material_type")

    print(f"📊 Creating individual box plots...")

    # PLOT 1: Total Impulse (Signed) - using proper font
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.boxplot(
        data=df, x="material_type", y="total_impulse", ax=ax, order=existing_materials
    )
    ax.set_title(
        "Total Impulse = ∫ F(t) dt\n(Signed Momentum Transfer)",
        fontsize=16,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax.set_ylabel("Total Impulse (N⋅s)", fontsize=14, fontfamily="DejaVu Sans")
    ax.set_xlabel("Material Configuration", fontsize=14, fontfamily="DejaVu Sans")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "total_impulse_boxplot.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "total_impulse_boxplot.pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: total_impulse_boxplot.png")

    # PLOT 2: Total Absolute Impulse (Magnitude)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.boxplot(
        data=df,
        x="material_type",
        y="total_abs_impulse",
        ax=ax,
        order=existing_materials,
    )
    ax.set_title(
        "Total Absolute Impulse = ∫ |F(t)| dt\n(Magnitude of Momentum Transfer)",
        fontsize=16,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax.set_ylabel("Total Absolute Impulse (N⋅s)", fontsize=14, fontfamily="DejaVu Sans")
    ax.set_xlabel("Material Configuration", fontsize=14, fontfamily="DejaVu Sans")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.savefig(
        output_dir / "absolute_impulse_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "absolute_impulse_boxplot.pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: absolute_impulse_boxplot.png")

    # PLOT 3: Peak Force
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.boxplot(
        data=df, x="material_type", y="peak_force", ax=ax, order=existing_materials
    )
    ax.set_title(
        "Peak Force Comparison",
        fontsize=16,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax.set_ylabel("Peak Force (N)", fontsize=14, fontfamily="DejaVu Sans")
    ax.set_xlabel("Material Configuration", fontsize=14, fontfamily="DejaVu Sans")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Format y-axis for large numbers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}k"))

    plt.tight_layout()
    plt.savefig(output_dir / "peak_force_boxplot.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "peak_force_boxplot.pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: peak_force_boxplot.png")

    # PLOT 4: Impact Duration
    df["impact_duration_ms"] = df["impact_duration"] * 1000
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.boxplot(
        data=df,
        x="material_type",
        y="impact_duration_ms",
        ax=ax,
        order=existing_materials,
    )
    ax.set_title(
        "Impact Duration Comparison",
        fontsize=16,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax.set_ylabel("Impact Duration (ms)", fontsize=14, fontfamily="DejaVu Sans")
    ax.set_xlabel("Material Configuration", fontsize=14, fontfamily="DejaVu Sans")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.savefig(
        output_dir / "impact_duration_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "impact_duration_boxplot.pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: impact_duration_boxplot.png")

    # PLOT 5: Impulse vs Force Scatter Plot
    if len(df) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Color by material type
        materials = df["material_type"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))

        for material, color in zip(materials, colors):
            material_data = df[df["material_type"] == material]
            ax.scatter(
                material_data["peak_force"],
                material_data["total_abs_impulse"],
                label=material,
                alpha=0.7,
                s=100,
                color=color,
            )

        ax.set_xlabel("Peak Force (N)", fontsize=14, fontfamily="DejaVu Sans")
        ax.set_ylabel(
            "Total Impulse Magnitude (N⋅s)", fontsize=14, fontfamily="DejaVu Sans"
        )
        ax.set_title(
            "Impulse vs Peak Force Relationship",
            fontsize=16,
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )
        ax.legend(title="Material", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Format x-axis for large numbers
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}k"))

        plt.tight_layout()
        plt.savefig(
            output_dir / "impulse_vs_force_scatter.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(output_dir / "impulse_vs_force_scatter.pdf", bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved: impulse_vs_force_scatter.png")

    # PLOT 6: Horizontal Bar Chart Summary
    if len(df) > 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Calculate means by material - add observed=False to suppress warning
        material_summary = df.groupby("material_type", observed=False).agg(
            {"total_abs_impulse": ["mean", "std"], "peak_force": ["mean", "std"]}
        )

        # Impulse ranking
        impulse_means = material_summary[("total_abs_impulse", "mean")].sort_values(
            ascending=True
        )
        impulse_stds = material_summary[("total_abs_impulse", "std")].reindex(
            impulse_means.index
        )

        y_pos = np.arange(len(impulse_means))
        bars1 = ax1.barh(
            y_pos,
            impulse_means.values,
            xerr=impulse_stds.values,
            alpha=0.7,
            capsize=5,
            color="skyblue",
            edgecolor="navy",
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(impulse_means.index)
        ax1.set_xlabel(
            "Total Impulse Magnitude (N⋅s)", fontsize=12, fontfamily="DejaVu Sans"
        )
        ax1.set_title(
            "Material Ranking by Impulse",
            fontsize=14,
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )
        ax1.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (mean, std) in enumerate(zip(impulse_means.values, impulse_stds.values)):
            ax1.text(
                mean + std * 1.1,
                i,
                f"{mean:.0f}±{std:.0f}",
                ha="left",
                va="center",
                fontweight="bold",
            )

        # Force ranking
        force_means = material_summary[("peak_force", "mean")].sort_values(
            ascending=True
        )
        force_stds = material_summary[("peak_force", "std")].reindex(force_means.index)

        y_pos = np.arange(len(force_means))
        bars2 = ax2.barh(
            y_pos,
            force_means.values,
            xerr=force_stds.values,
            alpha=0.7,
            capsize=5,
            color="lightcoral",
            edgecolor="darkred",
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(force_means.index)
        ax2.set_xlabel("Peak Force (N)", fontsize=12, fontfamily="DejaVu Sans")
        ax2.set_title(
            "Material Ranking by Peak Force",
            fontsize=14,
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )
        ax2.grid(True, alpha=0.3, axis="x")

        # Format x-axis and add labels
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}k"))
        for i, (mean, std) in enumerate(zip(force_means.values, force_stds.values)):
            ax2.text(
                mean + std * 1.1,
                i,
                f"{mean/1000:.0f}k±{std/1000:.0f}k",
                ha="left",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / "material_ranking_bars.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(output_dir / "material_ranking_bars.pdf", bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved: material_ranking_bars.png")

    # NEW: Add SI box plots in publication style
    print(f"📊 Creating SI box plots in publication style...")
    create_si_boxplots_from_impulse_results(valid_results, output_dir)

    print(f"\n🎉 All individual plots saved to: {output_dir}")
    print(f"📁 Files created:")
    print(f"   • total_impulse_boxplot.png/.pdf")
    print(f"   • absolute_impulse_boxplot.png/.pdf")
    print(f"   • peak_force_boxplot.png/.pdf")
    print(f"   • impact_duration_boxplot.png/.pdf")
    print(f"   • impulse_vs_force_scatter.png/.pdf")
    print(f"   • material_ranking_bars.png/.pdf")
    print(f"   • plot-box-J-F-SI.svg/.png (publication style, SI units)")
    print(f"   • plot-box-J-F-kSI.svg/.png (publication style, kN⋅s/kN units)")
    print(f"   • plot-box-J-F-mixed.svg/.png (publication style, N⋅s/kN mixed)")


def create_si_boxplots_from_impulse_results(results: List[Dict], output_dir: Path):
    """
    Create SI box plots directly from impulse analysis results.

    Args:
        results: List of impulse analysis results
        output_dir: Output directory
    """
    # Filter valid results
    valid_results = [
        r
        for r in results
        if "error" not in r and "total_abs_impulse" in r and "peak_force" in r
    ]

    if not valid_results:
        print(f"  ❌ No valid results for SI box plots")
        return

    df = pd.DataFrame(valid_results)

    print(f"📊 Creating SI box plots in publication style...")

    # Create all three unit versions
    plot_impulse_force_box_si(df, output_dir, units="SI")
    plot_impulse_force_box_si(df, output_dir, units="kSI")
    plot_impulse_force_box_si(df, output_dir, units="mixed")


def create_latex_table(results: List[Dict], output_dir: Path):
    """
    Create LaTeX table from impulse analysis results matching the desired format.

    Args:
        results: List of impulse analysis results
        output_dir: Output directory for LaTeX table
    """
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("❌ No valid results for LaTeX table")
        return

    df = pd.DataFrame(valid_results)

    # Configuration name mapping
    config_names = {
        "STND": "Standard (SD)",
        "DF": "Dual Fixed (DF)",
        "DS": "Dual Sliding",
        "SL": "Sliding (SL)",
        "BR": "Breakaway (BR)",
    }

    # Calculate statistics by material
    material_stats = (
        df.groupby("material_type", observed=False)
        .agg(
            {
                "total_abs_impulse": ["count", "mean", "std"],
                "peak_force": ["mean", "std"],
                "equivalent_kinetic_energy": ["mean", "std"],
            }
        )
        .round(6)
    )

    # Get STND values for percentage calculations
    stnd_impulse = (
        material_stats.loc["STND", ("total_abs_impulse", "mean")]
        if "STND" in material_stats.index
        else 0
    )
    stnd_energy = (
        material_stats.loc["STND", ("equivalent_kinetic_energy", "mean")]
        if "STND" in material_stats.index
        else 0
    )

    # Define desired material order
    material_order = ["STND", "DF", "DS", "SL", "BR"]

    # Create LaTeX table content
    latex_content = """\\begin{table}[htb]
\\centering
\\scriptsize
\\begin{tabular}{l|c|c|c|c|c|c|c|c}
\\hline
Configuration & \\#Runs & Impulse & STD & \\% vs & Max Force & STD & Avg. Impact & \\% vs \\\\
              &        & [kN·s]  &     & STND  & [kN]      &     & Energy [MJ] & STND \\\\
\\hline\n"""

    for material in material_order:
        if material not in material_stats.index:
            continue

        config_name = config_names.get(material, material)
        n_runs = int(material_stats.loc[material, ("total_abs_impulse", "count")])

        # Convert units
        impulse_kNs = (
            material_stats.loc[material, ("total_abs_impulse", "mean")] / 1000
        )  # N⋅s to kN⋅s
        impulse_std_kNs = (
            material_stats.loc[material, ("total_abs_impulse", "std")] / 1000
        )
        force_kN = (
            material_stats.loc[material, ("peak_force", "mean")] / 1000
        )  # N to kN
        force_std_kN = material_stats.loc[material, ("peak_force", "std")] / 1000
        energy_MJ = (
            material_stats.loc[material, ("equivalent_kinetic_energy", "mean")] / 1e6
        )  # J to MJ

        # Calculate percentages vs STND
        if material == "STND":
            impulse_pct = "0\\%"
            energy_pct = "0\\%"
        else:
            impulse_pct_val = (
                ((impulse_kNs * 1000 - stnd_impulse) / stnd_impulse * 100)
                if stnd_impulse > 0
                else 0
            )
            energy_pct_val = (
                ((energy_MJ * 1e6 - stnd_energy) / stnd_energy * 100)
                if stnd_energy > 0
                else 0
            )
            impulse_pct = f"{impulse_pct_val:+.0f}\\%"
            energy_pct = f"{energy_pct_val:+.0f}\\%"

        # Format the row
        latex_content += f"{config_name:<14} & {n_runs:2d} & {impulse_kNs:4.2f} & {impulse_std_kNs:4.2f} & {impulse_pct:>5s} & {force_kN:5.2f} & {force_std_kN:5.2f} & {energy_MJ:4.0f} & {energy_pct:>5s} \\\\\n"

    latex_content += """\\hline
\\end{tabular}
\\caption{Flyback impact performance comparison across gear configurations. Impact Energy calculated assuming inelastic collision.}
\\label{tab:flyback_results}
\\end{table}"""

    # Save LaTeX table
    latex_file = output_dir / "flyback_results_table.tex"
    with open(latex_file, "w") as f:
        f.write(latex_content)

    print(f"📄 LaTeX table saved to: {latex_file}")

    # Also print to console for immediate use
    print(f"\n📋 LATEX TABLE OUTPUT:")
    print("=" * 80)
    print(latex_content)
    print("=" * 80)

    return latex_content


def create_impulse_report(results: List[Dict], output_dir: Path):
    """Create comprehensive impulse analysis report."""

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return

    df = pd.DataFrame(valid_results)

    # Calculate statistics by material
    material_stats = (
        df.groupby("material_type")
        .agg(
            {
                "total_impulse": ["count", "mean", "std", "min", "max"],
                "total_abs_impulse": ["mean", "std"],
                "peak_force": ["mean", "std"],
                "impact_duration": ["mean", "std"],
            }
        )
        .round(6)
    )

    # ANOVA test
    materials = df["material_type"].unique()
    impulse_groups = [
        df[df["material_type"] == mat]["total_impulse"].values for mat in materials
    ]

    try:
        f_stat, p_value = stats.f_oneway(*impulse_groups)
    except:
        f_stat, p_value = np.nan, np.nan

    # Create report
    with open(output_dir / "impulse_analysis_report.txt", "w") as f:
        f.write("FISHING LINE IMPULSE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: Total impulse integration (∫ F(t) dt)\n")
        f.write(f"Primary Metric: Total momentum transfer (N⋅s)\n")
        f.write(f"Configuration weights: Included\n")
        f.write(f"Line mass: {LINE_MASS_FRACTION*100:.0f}% effective\n\n")

        # Method ranking
        method_means = (
            df.groupby("material_type")["total_abs_impulse"]
            .mean()
            .sort_values(ascending=False)
        )

        f.write("METHOD RANKING (by Total Absolute Impulse):\n")
        f.write("-" * 50 + "\n")
        for i, (method, mean_impulse) in enumerate(method_means.items(), 1):
            count = len(df[df["material_type"] == method])
            std = df[df["material_type"] == method]["total_abs_impulse"].std()
            signed_impulse = df[df["material_type"] == method]["total_impulse"].mean()
            f.write(
                f"{i}. {method}: {mean_impulse:.6f} N⋅s magnitude "
                f"({signed_impulse:+.6f} signed) ± {std:.6f} ({count} samples)\n"
            )

        f.write(f"\nSTATISTICAL ANALYSIS:\n")
        f.write(f"ANOVA F-statistic: {f_stat:.3f}\n")
        f.write(f"p-value: {p_value:.6f}\n")

        if p_value < 0.05:
            f.write("Result: Significant difference between methods\n")
        else:
            f.write("Result: No significant difference between methods\n")

        f.write(f"\nINTERPRETATION:\n")
        f.write(f"• Total impulse = total momentum transfer to fish/lure\n")
        f.write(f"• Positive values = forward momentum (typical impact)\n")
        f.write(f"• Negative values = backward momentum (strong rebound)\n")
        f.write(f"• Higher magnitude = more effective impact\n")

    print(
        f"📄 Impulse analysis report saved to: {output_dir / 'impulse_analysis_report.txt'}"
    )

    # Create LaTeX table
    create_latex_table(valid_results, output_dir)


def create_impulse_ranking_plot(df: pd.DataFrame, output_dir: Path):
    """Create method ranking visualization."""

    # Calculate ranking by absolute impulse
    material_summary = df.groupby("material_type").agg(
        {
            "total_impulse": ["mean", "std"],
            "total_abs_impulse": ["mean", "std"],
            "peak_force": "mean",
        }
    )

    material_summary_sorted = material_summary.sort_values(
        ("total_abs_impulse", "mean"), ascending=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Fishing Line Method Performance Ranking (Impulse-Based)",
        fontsize=16,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )

    # Signed impulse
    methods = material_summary_sorted.index
    impulses = material_summary_sorted[("total_impulse", "mean")]
    errors = material_summary_sorted[("total_impulse", "std")]

    bars1 = ax1.barh(
        range(len(methods)),
        impulses,
        xerr=errors,
        color=["red" if x < 0 else "green" for x in impulses],
        alpha=0.7,
        capsize=5,
    )
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_xlabel("Total Impulse (N⋅s)", fontfamily="DejaVu Sans")
    ax1.set_title(
        "Signed Impulse\n(+ = forward, - = backward)", fontfamily="DejaVu Sans"
    )
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (impulse, error) in enumerate(zip(impulses, errors)):
        label_x = impulse + error * 1.1 if impulse >= 0 else impulse - error * 1.1
        ax1.text(
            label_x,
            i,
            f"{impulse:+.4f}",
            ha="left" if impulse >= 0 else "right",
            va="center",
            fontweight="bold",
        )

    # Absolute impulse magnitude
    abs_impulses = material_summary_sorted[("total_abs_impulse", "mean")]
    abs_errors = material_summary_sorted[("total_abs_impulse", "std")]

    bars2 = ax2.barh(
        range(len(methods)),
        abs_impulses,
        xerr=abs_errors,
        color="blue",
        alpha=0.7,
        capsize=5,
    )
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    ax2.set_xlabel("Total Absolute Impulse (N⋅s)", fontfamily="DejaVu Sans")
    ax2.set_title(
        "Impulse Magnitude\n(Total Momentum Transfer)", fontfamily="DejaVu Sans"
    )
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for i, (abs_impulse, error) in enumerate(zip(abs_impulses, abs_errors)):
        ax2.text(
            abs_impulse + error * 1.1,
            i,
            f"{abs_impulse:.4f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "impulse_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_method_comparison_plots(
    ke_results: List[Dict], impulse_results: List[Dict], output_dir: Path
):
    """
    Create comparison plots between kinetic energy and impulse methods.

    Args:
        ke_results: Results from kinetic energy analysis
        impulse_results: Results from impulse analysis
        output_dir: Directory to save comparison plots
    """

    # Filter valid results
    ke_valid = [
        r
        for r in ke_results
        if "error" not in r and not np.isnan(r.get("kinetic_energy", np.nan))
    ]
    impulse_valid = [r for r in impulse_results if "error" not in r]

    if not ke_valid or not impulse_valid:
        print("❌ Insufficient valid results for comparison")
        return

    # Match files between methods
    ke_df = pd.DataFrame(ke_valid)
    impulse_df = pd.DataFrame(impulse_valid)

    # Merge on filename
    merged = pd.merge(ke_df, impulse_df, on="filename", suffixes=("_ke", "_impulse"))

    if len(merged) == 0:
        print("❌ No matching files between methods")
        return

    print(f"📊 Creating comparison plots for {len(merged)} matched files...")

    # Create comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Kinetic Energy vs Impulse Analysis Comparison",
        fontsize=16,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )

    # 1. Method correlation
    axes[0, 0].scatter(
        merged["kinetic_energy"], merged["equivalent_kinetic_energy"], alpha=0.6, s=50
    )
    axes[0, 0].set_xlabel("Kinetic Energy Method (J)", fontfamily="DejaVu Sans")
    axes[0, 0].set_ylabel("Impulse→KE Equivalent (J)", fontfamily="DejaVu Sans")
    axes[0, 0].set_title(
        "Energy Comparison\n(KE vs Impulse-derived)", fontfamily="DejaVu Sans"
    )

    # Add correlation line
    if len(merged) > 2:
        corr = np.corrcoef(
            merged["kinetic_energy"], merged["equivalent_kinetic_energy"]
        )[0, 1]
        z = np.polyfit(merged["kinetic_energy"], merged["equivalent_kinetic_energy"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(
            merged["kinetic_energy"].min(), merged["kinetic_energy"].max(), 100
        )
        axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.8)
        axes[0, 0].text(
            0.05,
            0.95,
            f"R = {corr:.3f}",
            transform=axes[0, 0].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Velocity comparison
    axes[0, 1].scatter(
        merged["initial_velocity"].abs(), merged["equivalent_velocity"], alpha=0.6, s=50
    )
    axes[0, 1].set_xlabel("KE Method Velocity (m/s)", fontfamily="DejaVu Sans")
    axes[0, 1].set_ylabel("Impulse Method Velocity (m/s)", fontfamily="DejaVu Sans")
    axes[0, 1].set_title("Velocity Comparison", fontfamily="DejaVu Sans")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Force comparison
    axes[0, 2].scatter(merged["max_force_ke"], merged["peak_force"], alpha=0.6, s=50)
    axes[0, 2].set_xlabel("KE Method Max Force (N)", fontfamily="DejaVu Sans")
    axes[0, 2].set_ylabel("Impulse Method Peak Force (N)", fontfamily="DejaVu Sans")
    axes[0, 2].set_title("Force Comparison", fontfamily="DejaVu Sans")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Material comparison - KE method
    sns.boxplot(data=merged, x="material_type_ke", y="kinetic_energy", ax=axes[1, 0])
    axes[1, 0].set_title("Kinetic Energy Method\nby Material", fontfamily="DejaVu Sans")
    axes[1, 0].set_ylabel("Kinetic Energy (J)", fontfamily="DejaVu Sans")
    axes[1, 0].set_yscale("log")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Material comparison - Impulse method
    sns.boxplot(
        data=merged, x="material_type_impulse", y="total_abs_impulse", ax=axes[1, 1]
    )
    axes[1, 1].set_title("Impulse Method\nby Material", fontfamily="DejaVu Sans")
    axes[1, 1].set_ylabel("Total Absolute Impulse (N⋅s)", fontfamily="DejaVu Sans")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Method ranking comparison
    ke_ranking = (
        merged.groupby("material_type_ke")["kinetic_energy"]
        .mean()
        .sort_values(ascending=False)
    )
    impulse_ranking = (
        merged.groupby("material_type_impulse")["total_abs_impulse"]
        .mean()
        .sort_values(ascending=False)
    )

    # Create ranking comparison table
    axes[1, 2].axis("off")

    ranking_data = []
    for i, (ke_mat, impulse_mat) in enumerate(
        zip(ke_ranking.index, impulse_ranking.index)
    ):
        ranking_data.append(
            [
                f"#{i+1}",
                ke_mat,
                f"{ke_ranking[ke_mat]:.2e}",
                impulse_mat,
                f"{impulse_ranking[impulse_mat]:.4f}",
            ]
        )

    table = axes[1, 2].table(
        cellText=ranking_data,
        colLabels=[
            "Rank",
            "KE Method",
            "Energy (J)",
            "Impulse Method",
            "Impulse (N⋅s)",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    axes[1, 2].set_title("Method Ranking Comparison", pad=20, fontfamily="DejaVu Sans")

    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Method comparison plots saved to: {output_dir}")


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================


def create_si_plots_standalone(csv_dir: str, output_dir: str):
    """
    Standalone function to create SI box plots from impulse analysis.

    Args:
        csv_dir: Directory containing CSV files
        output_dir: Output directory for plots
    """
    print("🎯 Creating SI Box Plots from Impulse Analysis")
    print("=" * 50)

    # Run impulse analysis
    results = run_impulse_analysis(csv_dir, f"{output_dir}/analysis_results")

    # Create SI plots
    create_si_boxplots_from_impulse_results(results, Path(output_dir))

    print(f"\n✅ SI box plots created in publication style!")
    print(f"📁 Check {output_dir} for:")
    print(f"   • plot-box-J-F-SI.svg (SI units: N⋅s, N)")
    print(f"   • plot-box-J-F-kSI.svg (kilo-SI: kN⋅s, kN)")
    print(f"   • plot-box-J-F-mixed.svg (mixed: N⋅s, kN)")


# Test function to verify Unicode support
def test_unicode_support():
    """Test if the current font supports the dot operator."""
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Test: N⋅s and N·s",
            fontsize=16,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Unicode Dot Test", fontfamily="DejaVu Sans")
        plt.savefig("unicode_test.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("✅ Unicode test passed - saved as unicode_test.png")
        return True
    except Exception as e:
        print(f"❌ Unicode test failed: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎯 Impulse Analysis Module")
    print("=" * 50)
    print("This module provides impulse-based analysis for fishing line")
    print("flyback testing. It measures total momentum transfer via ∫ F(t) dt.")
    print()
    print("Key advantages:")
    print("• Direct measurement of impact effectiveness")
    print("• Simple integration of complete force curve")
    print("• More relevant to fishing line performance")
    print("• Compatible with existing project structure")
    print("• NEW: Boundary validation plotting with --show-plot flag")
    print("• NEW: Publication-style SI box plots")
    print("• FIXED: Font compatibility and seaborn warnings")
    print()
    print("Usage:")
    print("from impulse_analysis import run_impulse_analysis")
    print("results = run_impulse_analysis('data/csv', 'impulse_output')")
    print()
    print("# Single file with plotting:")
    print("from impulse_analysis import analyze_single_file_with_impulse")
    print("result = analyze_single_file_with_impulse('file.csv', show_plot=True)")
    print()
    print("# Create publication-style SI box plots:")
    print("from impulse_analysis import create_si_plots_standalone")
    print("create_si_plots_standalone('data/csv', 'output')")
    print()
    print("# Test Unicode font support:")
    print("from impulse_analysis import test_unicode_support")
    print("test_unicode_support()")
