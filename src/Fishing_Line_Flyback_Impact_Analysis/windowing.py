"""
Interactive Data Windowing Tool for Fishing Line Impact Analysis

This tool allows you to visually inspect force data and select the correct
analysis window to isolate the actual impact event from noise, rebound, etc.
"""

import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
from matplotlib.widgets import SpanSelector

from .analysis import ImpactAnalyzer


class DataWindowingTool:
    """
    Interactive tool for selecting analysis windows in force data.
    """

    def __init__(self):
        self.force_data = None
        self.time_data = None
        self.selected_window = None
        self.analyzer = None
        self.fig = None
        self.ax = None
        self.span_selector = None
        self.current_file = None
        self.force_columns = None

        # Window selection state
        self.window_start_idx = None
        self.window_end_idx = None

    def load_csv_file(self, csv_path: str) -> bool:
        """
        Load CSV file and prepare force data.

        Args:
            csv_path: Path to CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.current_file = Path(csv_path)
            df = pd.read_csv(csv_path)

            # Initialize analyzer to process data
            self.analyzer = ImpactAnalyzer(mass=0.072)  # Use default mass for preview

            # Calculate total force
            self.force_data, self.force_columns = self.analyzer._calculate_total_force(
                df
            )
            self.time_data = self.analyzer._get_time_array(df, len(self.force_data))

            print(f"✅ Loaded: {self.current_file.name}")
            print(f"   Force columns: {', '.join(self.force_columns)}")
            print(f"   Data points: {len(self.force_data):,}")
            print(f"   Duration: {self.time_data[-1]:.4f} s")
            print(
                f"   Force range: {np.min(self.force_data):.1f} to {np.max(self.force_data):.1f} N"
            )

            return True

        except Exception as e:
            print(f"❌ Error loading {csv_path}: {e}")
            return False

    def create_interactive_plot(self):
        """Create interactive plot for window selection."""

        if self.force_data is None:
            print("❌ No data loaded. Use load_csv_file() first.")
            return

        # Create figure
        self.fig, (self.ax, self.ax_zoom) = plt.subplots(2, 1, figsize=(15, 10))
        self.fig.suptitle(
            f"Interactive Window Selection: {self.current_file.name}",
            fontsize=14,
            fontweight="bold",
        )

        # Main plot
        self.ax.plot(self.time_data, self.force_data, "b-", alpha=0.7, linewidth=1)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")
        self.ax.set_title("Full Force vs Time Data\n(Drag to select analysis window)")
        self.ax.grid(True, alpha=0.3)

        # Find and highlight potential impact region automatically
        self._suggest_impact_window()

        # Zoom plot (initially empty)
        self.ax_zoom.set_xlabel("Time (s)")
        self.ax_zoom.set_ylabel("Force (N)")
        self.ax_zoom.set_title("Selected Window (will update when you make selection)")
        self.ax_zoom.grid(True, alpha=0.3)

        # Create span selector for window selection
        self.span_selector = SpanSelector(
            self.ax,
            self.on_window_select,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="green"),  # Changed from rectprops to props
            minspan=0.001,  # Minimum 1ms window
        )

        # Add buttons
        ax_button_analyze = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_button_save = plt.axes([0.81, 0.02, 0.1, 0.04])
        ax_button_reset = plt.axes([0.59, 0.02, 0.1, 0.04])

        self.button_analyze = Button(ax_button_analyze, "Analyze")
        self.button_save = Button(ax_button_save, "Save Window")
        self.button_reset = Button(ax_button_reset, "Reset")

        self.button_analyze.on_clicked(self.analyze_selected_window)
        self.button_save.on_clicked(self.save_window_selection)
        self.button_reset.on_clicked(self.reset_selection)

        # Add instructions
        self.fig.text(
            0.02,
            0.02,
            'Instructions: Drag on upper plot to select window → Click "Analyze" to test → Click "Save Window" to save selection',
            fontsize=10,
            style="italic",
        )

        plt.tight_layout()
        plt.show()

    def _suggest_impact_window(self):
        """Suggest a potential impact window based on force characteristics."""

        # Find peak force
        max_force_idx = np.argmax(np.abs(self.force_data))
        max_force = np.abs(self.force_data[max_force_idx])

        # Look for significant force rise (>20% of peak)
        threshold = max_force * 0.2
        significant_indices = np.where(np.abs(self.force_data) > threshold)[0]

        if len(significant_indices) > 0:
            # Suggest window from first significant force to some time after peak
            suggested_start = significant_indices[0]
            suggested_end = min(
                len(self.force_data) - 1, max_force_idx + len(significant_indices) // 4
            )

            # Ensure reasonable duration (at least 1ms, at most 50ms)
            min_samples = int(0.001 / self.analyzer.dt)  # 1ms
            max_samples = int(0.05 / self.analyzer.dt)  # 50ms

            if suggested_end - suggested_start < min_samples:
                suggested_end = min(
                    len(self.force_data) - 1, suggested_start + min_samples
                )
            elif suggested_end - suggested_start > max_samples:
                suggested_end = suggested_start + max_samples

            # Highlight suggested region
            self.ax.axvspan(
                self.time_data[suggested_start],
                self.time_data[suggested_end],
                alpha=0.2,
                color="yellow",
                label=f"Suggested window ({(suggested_end - suggested_start) * self.analyzer.dt * 1000:.1f} ms)",
            )

            # Add peak marker
            self.ax.axvline(
                self.time_data[max_force_idx],
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Peak force ({max_force:.0f} N)",
            )

            self.ax.legend()

            print(
                f"💡 Suggested window: {self.time_data[suggested_start]:.4f} - {self.time_data[suggested_end]:.4f} s"
            )
            print(
                f"   Duration: {(suggested_end - suggested_start) * self.analyzer.dt * 1000:.1f} ms"
            )
            print(
                f"   Peak force: {max_force:.0f} N at {self.time_data[max_force_idx]:.4f} s"
            )

    def on_window_select(self, start_time: float, end_time: float):
        """Handle window selection from span selector."""

        # Convert time to indices
        start_idx = np.searchsorted(self.time_data, start_time)
        end_idx = np.searchsorted(self.time_data, end_time)

        # Ensure valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(self.force_data) - 1, end_idx)

        if end_idx <= start_idx:
            print("❌ Invalid window selection")
            return

        self.window_start_idx = start_idx
        self.window_end_idx = end_idx

        # Update zoom plot
        window_time = self.time_data[start_idx : end_idx + 1]
        window_force = self.force_data[start_idx : end_idx + 1]

        self.ax_zoom.clear()
        self.ax_zoom.plot(window_time, window_force, "g-", linewidth=2)
        self.ax_zoom.set_xlabel("Time (s)")
        self.ax_zoom.set_ylabel("Force (N)")

        duration_ms = (end_time - start_time) * 1000
        max_force_in_window = np.max(np.abs(window_force))

        self.ax_zoom.set_title(
            f"Selected Window: {duration_ms:.1f} ms, Peak: {max_force_in_window:.0f} N"
        )
        self.ax_zoom.grid(True, alpha=0.3)

        # Add peak marker in zoom view
        peak_idx_in_window = np.argmax(np.abs(window_force))
        self.ax_zoom.axvline(
            window_time[peak_idx_in_window], color="red", linestyle="--", alpha=0.7
        )

        self.fig.canvas.draw()

        print(
            f"📏 Selected window: {start_time:.4f} - {end_time:.4f} s ({duration_ms:.1f} ms)"
        )
        print(f"   Samples: {end_idx - start_idx + 1}")
        print(f"   Peak force in window: {max_force_in_window:.0f} N")

    def analyze_selected_window(self, event):
        """Analyze the currently selected window."""

        if self.window_start_idx is None or self.window_end_idx is None:
            print("❌ No window selected. Please drag on the plot to select a window.")
            return

        # Extract windowed data
        window_force = self.force_data[self.window_start_idx : self.window_end_idx + 1]
        window_time = (
            self.time_data[self.window_start_idx : self.window_end_idx + 1]
            - self.time_data[self.window_start_idx]
        )

        print(f"\n🔬 ANALYZING SELECTED WINDOW:")
        print(
            f"   Time range: {self.time_data[self.window_start_idx]:.4f} - {self.time_data[self.window_end_idx]:.4f} s"
        )
        print(
            f"   Duration: {(self.window_end_idx - self.window_start_idx) * self.analyzer.dt * 1000:.1f} ms"
        )
        print(f"   Samples: {len(window_force)}")

        # Test with different material configurations
        materials_to_test = ["STND", "DF", "DS", "SL", "BR"]

        print(f"\n📊 RESULTS FOR DIFFERENT CONFIGURATIONS:")
        print(f"{'Material':<8} {'Mass':<6} {'Velocity':<9} {'Energy':<8} {'Status'}")
        print("-" * 45)

        for material in materials_to_test:
            try:
                # Create analyzer for this material
                test_analyzer = ImpactAnalyzer(
                    material_code=material, include_line_mass=True
                )

                # Analyze windowed data
                result = test_analyzer.calculate_corrected_energy(
                    window_force, window_time
                )

                velocity = abs(result["initial_velocity"])
                energy = result["kinetic_energy"]
                mass_g = test_analyzer.mass * 1000

                # Status based on velocity
                if 150 <= velocity <= 400:
                    status = "✅"
                elif 100 <= velocity <= 500:
                    status = "🟡"
                else:
                    status = "❌"

                print(
                    f"{material:<8} {mass_g:<4.0f}g  {velocity:<6.0f} m/s {energy:<6.1f} J  {status}"
                )

            except Exception as e:
                print(f"{material:<8} ERROR: {e}")

        print(f"\n💡 Window selection tips:")
        print(f"   ✅ = Target range (150-400 m/s)")
        print(f"   🟡 = Reasonable (100-500 m/s)")
        print(f"   ❌ = Outside range - try different window")
        print(f"   • Include impact rise but exclude rebound")
        print(f"   • Typical good windows: 2-20 ms duration")
        print(f"   • Should see consistent results across materials")

    def save_window_selection(self, event):
        """Save the current window selection."""

        if self.window_start_idx is None or self.window_end_idx is None:
            print("❌ No window selected to save.")
            return

        # Prepare window info
        window_info = {
            "filename": self.current_file.name,
            "start_time": float(self.time_data[self.window_start_idx]),
            "end_time": float(self.time_data[self.window_end_idx]),
            "start_idx": int(self.window_start_idx),
            "end_idx": int(self.window_end_idx),
            "duration_ms": float(
                (self.window_end_idx - self.window_start_idx) * self.analyzer.dt * 1000
            ),
            "samples": int(self.window_end_idx - self.window_start_idx + 1),
            "sampling_rate": float(self.analyzer.sampling_rate),
            "force_columns": self.force_columns,
        }

        # Save to JSON file
        output_file = self.current_file.parent / f"{self.current_file.stem}_window.json"

        with open(output_file, "w") as f:
            json.dump(window_info, f, indent=2)

        print(f"💾 Window selection saved to: {output_file}")

        # Also create windowed CSV file
        self.create_windowed_csv(window_info)

    def create_windowed_csv(self, window_info: Dict):
        """Create a new CSV file with only the selected window data."""

        try:
            # Load original CSV
            df_original = pd.read_csv(self.current_file)

            # Extract windowed data
            start_idx = window_info["start_idx"]
            end_idx = window_info["end_idx"]

            df_windowed = df_original.iloc[start_idx : end_idx + 1].copy()

            # Reset time to start from zero if time column exists
            time_cols = [col for col in df_windowed.columns if "time" in col.lower()]
            if time_cols:
                time_col = time_cols[0]
                df_windowed[time_col] = (
                    df_windowed[time_col] - df_windowed[time_col].iloc[0]
                )

            # Save windowed CSV
            output_csv = (
                self.current_file.parent / f"{self.current_file.stem}_windowed.csv"
            )
            df_windowed.to_csv(output_csv, index=False)

            print(f"📄 Windowed CSV saved to: {output_csv}")
            print(f"   Original samples: {len(df_original)}")
            print(f"   Windowed samples: {len(df_windowed)}")

        except Exception as e:
            print(f"❌ Error creating windowed CSV: {e}")

    def reset_selection(self, event):
        """Reset the window selection."""
        self.window_start_idx = None
        self.window_end_idx = None

        # Clear zoom plot
        self.ax_zoom.clear()
        self.ax_zoom.set_xlabel("Time (s)")
        self.ax_zoom.set_ylabel("Force (N)")
        self.ax_zoom.set_title("Selected Window (make selection above)")
        self.ax_zoom.grid(True, alpha=0.3)

        self.fig.canvas.draw()
        print("🔄 Selection reset")


def batch_window_files(
    data_dir: str, pattern: str = "*.csv", auto_suggest: bool = True
) -> List[Dict]:
    """
    Batch process files to suggest windows automatically.

    Args:
        data_dir: Directory containing CSV files
        pattern: File pattern to match
        auto_suggest: Whether to auto-suggest windows

    Returns:
        List of suggested windows
    """
    data_path = Path(data_dir)
    files = list(data_path.glob(pattern))

    suggestions = []
    tool = DataWindowingTool()

    print(f"🔍 BATCH WINDOW SUGGESTIONS")
    print(f"Processing {len(files)} files...")
    print("-" * 60)

    for file_path in files:
        try:
            if tool.load_csv_file(file_path):
                # Get automatic suggestion
                max_force_idx = np.argmax(np.abs(tool.force_data))
                max_force = np.abs(tool.force_data[max_force_idx])

                # Find significant force region
                threshold = max_force * 0.2
                significant_indices = np.where(np.abs(tool.force_data) > threshold)[0]

                if len(significant_indices) > 0:
                    start_idx = significant_indices[0]
                    end_idx = min(
                        len(tool.force_data) - 1,
                        max_force_idx + len(significant_indices) // 4,
                    )

                    # Limit duration
                    max_samples = int(0.02 / tool.analyzer.dt)  # 20ms max
                    if end_idx - start_idx > max_samples:
                        end_idx = start_idx + max_samples

                    duration_ms = (end_idx - start_idx) * tool.analyzer.dt * 1000

                    suggestion = {
                        "filename": file_path.name,
                        "start_time": tool.time_data[start_idx],
                        "end_time": tool.time_data[end_idx],
                        "duration_ms": duration_ms,
                        "peak_force": max_force,
                        "peak_time": tool.time_data[max_force_idx],
                    }

                    suggestions.append(suggestion)

                    print(
                        f"✅ {file_path.name}: {duration_ms:.1f}ms window, peak {max_force:.0f}N"
                    )
                else:
                    print(f"❌ {file_path.name}: No significant force detected")

        except Exception as e:
            print(f"❌ {file_path.name}: {e}")

    # Save batch suggestions
    if suggestions:
        output_file = data_path / "batch_window_suggestions.json"
        with open(output_file, "w") as f:
            json.dump(suggestions, f, indent=2)
        print(f"\n💾 Batch suggestions saved to: {output_file}")

    return suggestions


def review_window_suggestions(data_dir: str, suggestions_file: str = None):
    """
    Review and validate window suggestions.

    Args:
        data_dir: Directory containing CSV files
        suggestions_file: Path to suggestions JSON file
    """
    data_path = Path(data_dir)

    if suggestions_file is None:
        suggestions_file = data_path / "batch_window_suggestions.json"

    if not Path(suggestions_file).exists():
        print("❌ No suggestions file found. Run suggest-windows first.")
        return

    # Load suggestions
    with open(suggestions_file) as f:
        suggestions = json.load(f)

    print("🔍 REVIEWING WINDOW SUGGESTIONS")
    print("=" * 80)
    print(f"📁 Data directory: {data_dir}")
    print(f"📄 Suggestions file: {suggestions_file}")
    print(f"🔢 Total suggestions: {len(suggestions)}")
    print()

    # Analyze suggestions
    durations = [s["duration_ms"] for s in suggestions]
    peak_forces = [s["peak_force"] for s in suggestions]
    peak_times = [s["peak_time"] for s in suggestions]

    print("📊 SUGGESTION STATISTICS:")
    print(f"   Duration range: {min(durations):.1f} - {max(durations):.1f} ms")
    print(f"   Average duration: {np.mean(durations):.1f} ± {np.std(durations):.1f} ms")
    print(f"   Peak force range: {min(peak_forces):.0f} - {max(peak_forces):.0f} N")
    print(f"   Peak time range: {min(peak_times):.1f} - {max(peak_times):.1f} s")
    print()

    # Identify potential issues
    print("⚠️  POTENTIAL ISSUES:")

    # Issue 1: Very short windows (< 3ms)
    short_windows = [s for s in suggestions if s["duration_ms"] < 3.0]
    if short_windows:
        print(f"   🔸 {len(short_windows)} windows < 3ms (may miss impact data):")
        for s in short_windows[:5]:  # Show first 5
            print(f"      {s['filename']}: {s['duration_ms']:.1f}ms")
        if len(short_windows) > 5:
            print(f"      ... and {len(short_windows) - 5} more")

    # Issue 2: Maximum duration windows (exactly 20ms - likely capped)
    max_windows = [s for s in suggestions if abs(s["duration_ms"] - 20.0) < 0.1]
    if max_windows:
        print(f"   🔸 {len(max_windows)} windows at 20ms limit (may include rebound):")
        for s in max_windows[:5]:
            print(f"      {s['filename']}: {s['peak_force']:.0f}N peak")
        if len(max_windows) > 5:
            print(f"      ... and {len(max_windows) - 5} more")

    # Issue 3: Very low peak forces (< 1000N)
    low_force = [s for s in suggestions if s["peak_force"] < 1000]
    if low_force:
        print(f"   🔸 {len(low_force)} windows with very low peak force (< 1000N):")
        for s in low_force:
            print(f"      {s['filename']}: {s['peak_force']:.0f}N")

    # Issue 4: Very high peak forces (> 100,000N)
    high_force = [s for s in suggestions if s["peak_force"] > 100000]
    if high_force:
        print(f"   🔸 {len(high_force)} windows with very high peak force (> 100kN):")
        for s in high_force:
            print(f"      {s['filename']}: {s['peak_force']:.0f}N")

    # Issue 5: Late peak times (> 100s - likely wrong events)
    late_peaks = [s for s in suggestions if s["peak_time"] > 100]
    if late_peaks:
        print(f"   🔸 {len(late_peaks)} windows with late peak times (> 100s):")
        for s in late_peaks:
            print(f"      {s['filename']}: peak at {s['peak_time']:.1f}s")

    print()

    # Test some suggestions with analysis
    print("🧪 TESTING SUGGESTIONS WITH ANALYSIS:")
    print("-" * 60)
    print(
        f"{'Filename':<15} {'Duration':<8} {'Peak Force':<10} {'Est Velocity':<12} {'Status'}"
    )
    print("-" * 60)

    good_suggestions = []
    needs_review = []

    # Test a sample of suggestions
    test_suggestions = suggestions[:20]  # Test first 20

    for suggestion in test_suggestions:
        filename = suggestion["filename"]
        file_path = data_path / filename

        if not file_path.exists():
            continue

        try:
            # Load data and test suggestion
            material = filename.split("-")[0]
            analyzer = ImpactAnalyzer(material_code=material, include_line_mass=True)

            # Load CSV and extract window
            df = pd.read_csv(file_path)
            force_data, _ = analyzer._calculate_total_force(df)
            time_data = analyzer._get_time_array(df, len(force_data))

            # Find window indices
            start_idx = np.searchsorted(time_data, suggestion["start_time"])
            end_idx = np.searchsorted(time_data, suggestion["end_time"])

            # Extract windowed data
            window_force = force_data[start_idx : end_idx + 1]
            window_time = time_data[start_idx : end_idx + 1] - time_data[start_idx]

            # Analyze
            result = analyzer.calculate_corrected_energy(window_force, window_time)
            velocity = abs(result["initial_velocity"])

            # Determine status
            if 150 <= velocity <= 400:
                status = "✅ Good"
                good_suggestions.append(suggestion)
            elif 100 <= velocity <= 500:
                status = "🟡 OK"
                good_suggestions.append(suggestion)
            else:
                status = "❌ Review"
                needs_review.append(suggestion)

            print(
                f"{filename:<15} {suggestion['duration_ms']:<6.1f}ms {suggestion['peak_force']:<8.0f}N {velocity:<8.0f} m/s {status}"
            )

        except Exception as e:
            print(f"{filename:<15} ERROR: {str(e)[:30]}...")
            needs_review.append(suggestion)

    print("-" * 60)

    # Summary and recommendations
    print(f"\n📈 TESTING SUMMARY:")
    print(f"   ✅ Good suggestions: {len(good_suggestions)}")
    print(f"   ❌ Need review: {len(needs_review)}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    priority_files = []

    # High priority: files that definitely need manual review
    if short_windows:
        priority_files.extend([s["filename"] for s in short_windows])
    if low_force:
        priority_files.extend([s["filename"] for s in low_force])
    if late_peaks:
        priority_files.extend([s["filename"] for s in late_peaks])
    if needs_review:
        priority_files.extend([s["filename"] for s in needs_review])

    # Remove duplicates and show unique files
    priority_files = list(set(priority_files))

    if priority_files:
        print(
            f"   🔴 HIGH PRIORITY - Manual review needed ({len(priority_files)} files):"
        )
        for filename in priority_files[:10]:  # Show first 10
            print(
                f"      python -m Fishing_Line_Flyback_Impact_Analysis window-tool data/csv/{filename}"
            )
        if len(priority_files) > 10:
            print(f"      ... and {len(priority_files) - 10} more files")

    # Medium priority: 20ms windows that might include rebound
    if (
        max_windows
        and len([s for s in max_windows if s["filename"] not in priority_files]) > 0
    ):
        medium_priority = [
            s["filename"] for s in max_windows if s["filename"] not in priority_files
        ]
        print(
            f"   🟡 MEDIUM PRIORITY - Check for rebound inclusion ({len(medium_priority)} files)"
        )

    # Low priority: everything else looks reasonable
    remaining = len(suggestions) - len(priority_files) - len(max_windows)
    if remaining > 0:
        print(f"   🟢 LOW PRIORITY - Likely good suggestions ({remaining} files)")

    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Review high priority files manually:")
    print(
        f"   python -m Fishing_Line_Flyback_Impact_Analysis window-tool data/csv/[filename]"
    )
    print(f"2. Apply windows (will use suggestions + any manual overrides):")
    print(f"   python -m Fishing_Line_Flyback_Impact_Analysis apply-windows data/csv")
    print(f"3. Analyze windowed data:")
    print(
        f"   python -m Fishing_Line_Flyback_Impact_Analysis analyze-windowed data/csv"
    )

    return {
        "total_suggestions": len(suggestions),
        "good_suggestions": len(good_suggestions),
        "needs_review": len(needs_review),
        "priority_files": priority_files,
        "short_windows": len(short_windows),
        "max_windows": len(max_windows),
        "low_force": len(low_force),
        "high_force": len(high_force),
        "late_peaks": len(late_peaks),
    }


def create_review_report(data_dir: str, output_file: str = None):
    """Create a detailed review report."""

    if output_file is None:
        output_file = Path(data_dir) / "window_review_report.txt"

    print(f"📝 Creating detailed review report...")

    # Run review and capture results
    import io
    import sys

    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        review_results = review_window_suggestions(data_dir)
        captured_text = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    # Write to file
    with open(output_file, "w") as f:
        f.write("FISHING LINE IMPACT ANALYSIS - WINDOW SUGGESTIONS REVIEW\n")
        f.write("=" * 80 + "\n\n")
        f.write(captured_text)
        f.write("\n\nREVIEW COMPLETED\n")
        f.write(
            f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    print(f"📄 Review report saved to: {output_file}")

    # Also print the summary
    print(captured_text)

    return review_results


# Convenience functions
def interactive_window_tool(csv_path: str):
    """Launch interactive windowing tool for a single file."""
    tool = DataWindowingTool()

    if tool.load_csv_file(csv_path):
        print(f"\n🎛️  INTERACTIVE WINDOWING TOOL")
        print(f"File: {csv_path}")
        print("Instructions:")
        print("1. Drag on the upper plot to select analysis window")
        print("2. Click 'Analyze' to test window with different configurations")
        print("3. Click 'Save Window' to save selection")
        print("4. Look for consistent velocities (150-400 m/s) across materials")
        print()

        tool.create_interactive_plot()
    else:
        print(f"❌ Failed to load {csv_path}")


if __name__ == "__main__":
    print("🎛️  Data Windowing Tool for Fishing Line Impact Analysis")
    print("=" * 60)
    print("Usage:")
    print("# Interactive tool for single file:")
    print("interactive_window_tool('data/csv/STND-21-5.csv')")
    print()
    print("# Batch suggestions for all files:")
    print("suggestions = batch_window_files('data/csv')")
    print()
    print("# Review suggestions:")
    print("review_window_suggestions('data/csv')")
