"""Impact analysis module for fishing line flyback analysis."""

import logging
import os
import types
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import find_peaks


class ImpactAnalyzer:
    """Analyzer for fishing line flyback impact properties."""

    def __init__(self) -> None:
        """Initialize the ImpactAnalyzer."""
        self.log = logging.getLogger(__name__)

        # Configuration for different test types and their masses
        self.config_masses = {
            "STND": 49,
            "DF": 62,
            "DS": 75,
            "SL": 65,
            "BR": 45,  # Default/break away
        }

    def load_file(self, filepath: str) -> pd.DataFrame:
        """Load impact test data from CSV or H5 file.

        Args:
            filepath: Path to CSV or H5 file

        Returns:
            DataFrame with impact test data and metadata

        Raises:
            ValueError: If file format not supported
        """
        self.log.debug("Loading file: %s", filepath)

        # Parse metadata from filename
        fname = os.path.basename(filepath)

        # Expected format: CONFIG-DIAMETER-FILENUM.csv
        fname_parts = fname.split("-")
        if len(fname_parts) >= 3:
            config = fname_parts[0]
            diam = fname_parts[1]
            fnum = fname_parts[-1].split(".")[0]
        else:
            config = "UNKNOWN"
            diam = "21"  # default
            fnum = "0"

        if filepath.endswith(".csv"):
            df, peak = self._load_csv_file(filepath)
        elif filepath.endswith(".h5"):
            df, peak = self._load_h5_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # Set metadata - this is the key fix
        df.meta = types.SimpleNamespace()
        df.meta.fname = fname
        df.meta.filepath = filepath
        df.meta.run_num = fnum
        df.meta.config = config
        df.meta.diam = diam
        df.meta.peak = peak

        return df

    def _load_csv_file(self, filepath: str) -> pd.DataFrame:
        """Load and process CSV file from Dewesoft."""
        df = pd.read_csv(filepath)

        # Remove video column if present
        if "Video/Camera 0 ()" in df.columns:
            df = df.drop(["Video/Camera 0 ()"], axis=1)

        # Convert to float and clean data
        df = df.astype(float)
        df = df.abs()
        df = df.dropna()

        # Rename columns to standard format
        column_mapping = {
            "Time (s)": "time",
            "AI 1/AI 1 (lbf)": "S1",
            "AI 2/AI 2 (lbf)": "S2",
            "AI 3/AI 3 (lbf)": "S3",
            "AI 4/AI 4 (lbf)": "S4",
        }
        df.rename(columns=column_mapping, inplace=True)

        # Find max sensor reading
        sensor_cols = [
            col for col in df.columns if col.startswith("S") and col != "SUM"
        ]
        if sensor_cols:
            max_row = df.loc[:, sensor_cols].idxmax()
            df["Max"] = df[df.loc[max_row].max().idxmax()]

        # Calculate sum of all sensors (mean * 4 for averaging)
        total = df.loc[:, sensor_cols].mean(axis=1) * 4
        df.insert(5, "SUM", total)

        # Find peaks and trim data around main peak
        peaks, _ = find_peaks(df["SUM"], height=1000, distance=100000)
        peak = None
        if peaks.size > 0:
            peak = max(peaks)
            # Trim data around peak (3000 samples before, 20000 after)
            start_idx = max(0, peak - 3000)
            end_idx = min(len(df), peak + 20000)
            df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Generate H5 file for faster future loading
        h5_dir = Path("data/h5")
        h5_dir.mkdir(parents=True, exist_ok=True)
        h5_path = h5_dir / f"{Path(filepath).stem}.h5"
        try:
            df.to_hdf(h5_path, key="df", mode="w")
        except Exception as e:
            self.log.warning(f"Could not save H5 file: {e}")

        return df, peak

    def _load_h5_file(self, filepath: str) -> Tuple[pd.DataFrame, Optional[int]]:
        """Load preprocessed H5 file."""
        df = pd.read_hdf(filepath)

        # Find peak again for metadata
        peaks, _ = find_peaks(df["SUM"], height=1000, distance=100000)
        peak = None
        if peaks.size > 0:
            peak = max(peaks)

        return df, peak

    def calculate_impact_properties(
        self, df: pd.DataFrame, param_y: str = "SUM", param_x: str = "time"
    ) -> Dict[str, float]:
        """Calculate impact properties from sensor data.

        Args:
            df: DataFrame with sensor data
            param_y: Y parameter for analysis (default: "SUM")
            param_x: X parameter for analysis (default: "time")

        Returns:
            Dictionary with calculated properties

        Raises:
            ValueError: If DataFrame is empty or invalid
        """
        # Handle empty dataframes
        if df.empty or len(df) == 0:
            raise ValueError("DataFrame is empty - cannot calculate properties")

        config = df.meta.config

        # Get mass for this configuration
        mass_kg = self.config_masses.get(config, 45) * 6.85218e-5  # Convert to kg

        # Get data arrays
        x = df[param_x].values

        # Handle "All" parameter by using SUM
        if param_y == "All":
            param_y = "SUM"

        y = df[param_y].values

        # Adjust time to start from zero
        if param_x == "time":
            x = x - x.min()

        # Calculate properties
        max_force = float(np.max(y))

        # Calculate impulse using Simpson's rule
        impulse = simpson(y, x, dx=1 / 20000)

        # Calculate velocity and energy
        velocity = impulse / mass_kg if mass_kg > 0 else 0
        kinetic_energy = (impulse**2) / (2 * mass_kg) if mass_kg > 0 else 0

        properties = {
            "max_force_N": max_force,
            "impulse_Ns": impulse,
            "velocity_ms": velocity,
            "kinetic_energy_J": kinetic_energy,
            "mass_kg": mass_kg,
            "config": config,
            "diameter": df.meta.diam,
            "run_number": df.meta.run_num,
            "filename": df.meta.fname,  # Add filename to properties
        }

        return properties

    def process_single_file(self, filepath: str) -> Dict[str, Any]:
        """Process a single file and return results.

        Args:
            filepath: Path to data file

        Returns:
            Dictionary with analysis results
        """
        df = self.load_file(filepath)
        properties = self.calculate_impact_properties(df)

        # Add file info
        properties["filepath"] = filepath
        properties["filename"] = df.meta.fname

        return properties

    def load_results_file(self, filepath: str) -> pd.DataFrame:
        """Load results from text file for post-processing.

        Args:
            filepath: Path to results text file

        Returns:
            DataFrame with processed results

        Raises:
            ValueError: If file format is invalid
        """
        if not filepath.endswith(".txt"):
            raise ValueError("Results file must be .txt format")

        # Read the results file (format: filename,J=value,F=value)
        df = pd.read_csv(filepath, header=None, sep="=|,", engine="python")

        # Handle the parsing - should have columns: filename, 'J', value, 'F', value
        if len(df.columns) >= 5:
            df = df[[0, 2, 4]]  # filename, J value, F value
            df.columns = ["fname", "J", "F"]
        else:
            # Fallback parsing
            df = df.drop(columns=[1, 3])  # Remove the '=' columns
            df.columns = ["fname", "J", "F"]

        # Convert numeric columns
        df.loc[:, df.columns != "fname"] = df.loc[:, df.columns != "fname"].astype(
            float
        )

        # Extract test type from filename
        test_type = df["fname"].str.split("-").str[0]
        df.insert(1, "test_type", test_type)

        # Filter out bad files (you can customize this list)
        bad_files = [
            "STND-21-1",
            "STND-21-3",
            "DF-21-6",
            "DF-21-7",
            "DS-21-1",
            "DS-21-8",
            "SL-21-8",
        ]

        for bad_file in bad_files:
            df = df[~df["fname"].str.contains(bad_file)]

        # Add metadata
        df.meta = types.SimpleNamespace()
        df.meta.filepath = filepath

        return df

    def calculate_summary_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics by test type.

        Args:
            results_df: DataFrame with test results

        Returns:
            DataFrame with summary statistics
        """
        test_types = ["STND", "BR", "DF", "DS", "SL"]
        results = pd.DataFrame(
            columns=[
                "test_type",
                "mean_J",
                "std_J",
                "var_J",
                "mean_F",
                "std_F",
                "var_F",
                "%diff_J",
                "%diff_F",
            ]
        )

        stnd_mean = None

        for test_type in test_types:
            type_data = results_df[results_df["test_type"] == test_type]
            if type_data.empty:
                continue

            calc_data = type_data[["J", "F"]]
            val_std = calc_data.std()
            val_mean = calc_data.mean()
            val_var = calc_data.var()

            if test_type == "STND":
                stnd_mean = val_mean
                diff_j = 0
                diff_f = 0
            else:
                diff_j = (
                    (val_mean["J"] / stnd_mean["J"]) - 1 if stnd_mean is not None else 0
                )
                diff_f = (
                    (val_mean["F"] / stnd_mean["F"]) - 1 if stnd_mean is not None else 0
                )

            results.loc[len(results)] = [
                test_type,
                val_mean["J"],
                val_std["J"],
                val_var["J"],
                val_mean["F"],
                val_std["F"],
                val_var["F"],
                diff_j,
                diff_f,
            ]

        return results

    def generate_summary_table(
        self, results_df: pd.DataFrame, output_dir: str = None
    ) -> Dict[str, Any]:
        """Generate human-readable summary table and print to terminal.

        Args:
            results_df: DataFrame with test results
            output_dir: Optional output directory for saving files

        Returns:
            Dictionary with configuration statistics
        """
        # Calculate detailed statistics for each configuration
        config_stats = {}

        # Configuration name mapping for display
        config_names = {
            "STND": "Standard (SD)",
            "DF": "Dual Fixed (DF)",
            "DS": "Dual Sliding",
            "SL": "Sliding (SL)",
            "BR": "Breakaway (BR)",
        }

        # First, get STND baseline values for percentage calculations
        stnd_data = results_df[results_df["test_type"] == "STND"]
        stnd_impulse_mean = stnd_data["J"].mean() if not stnd_data.empty else 1.0
        stnd_force_mean = stnd_data["F"].mean() if not stnd_data.empty else 1.0

        # Calculate stats for each configuration
        for config in ["STND", "DF", "DS", "SL", "BR"]:
            config_data = results_df[results_df["test_type"] == config]
            if config_data.empty:
                continue

            # Convert units: lbf to kN for force, lbf·s to kN·s for impulse
            impulse_kns = config_data["J"] * 4.44822 / 1000  # lbf·s to kN·s
            force_kn = config_data["F"] * 4.44822 / 1000  # lbf to kN

            # Calculate impact energy (assuming inelastic collision)
            # E = (impulse^2) / (2 * mass), convert to MJ
            mass_kg = self.config_masses.get(config, 45) * 6.85218e-5  # to kg
            energy_mj = (
                ((config_data["J"] * 4.44822) ** 2) / (2 * mass_kg) / 1e6
            )  # to MJ

            # Calculate percentages relative to STND
            impulse_pct = (
                ((config_data["J"].mean() / stnd_impulse_mean) - 1) * 100
                if stnd_impulse_mean > 0
                else 0
            )
            force_pct = (
                ((config_data["F"].mean() / stnd_force_mean) - 1) * 100
                if stnd_force_mean > 0
                else 0
            )
            energy_pct = impulse_pct  # Energy percentage same as impulse since E ∝ J²

            config_stats[config] = {
                "name": config_names.get(config, config),
                "n_runs": len(config_data),
                "impulse_mean": impulse_kns.mean(),
                "impulse_std": impulse_kns.std(),
                "impulse_pct": impulse_pct,
                "force_mean": force_kn.mean(),
                "force_std": force_kn.std(),
                "force_pct": force_pct,
                "energy_mean": energy_mj.mean(),
                "energy_std": energy_mj.std(),
                "energy_pct": energy_pct,
            }

        # Print the table
        self._print_summary_table(config_stats)

        # Optionally save to file if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save detailed summary file
            summary_file = output_path / "flyback_summary_table.txt"
            with open(summary_file, "w") as f:
                f.write("FLYBACK IMPACT ANALYSIS - SUMMARY TABLE\n")
                f.write("=" * 80 + "\n\n")

                # Write the same table that was printed
                self._write_table_to_file(f, config_stats)

                f.write("\n\nNOTES:\n")
                f.write(
                    "- Impact Energy calculated assuming inelastic collision: "
                    "E = impulse²/(2×mass)\n"
                )
                f.write(
                    "- Percentages calculated relative to Standard (SD) configuration\n"
                )
                f.write("- Force converted from lbf to kN (×4.44822/1000)\n")
                f.write("- Impulse converted from lbf·s to kN·s (×4.44822/1000)\n")
                f.write("- Energy converted to MJ (×1e-6)\n")

            self.log.info(f"Summary table saved to: {summary_file}")

        return config_stats

    def _print_summary_table(self, config_stats: Dict[str, Any]) -> None:
        """Print formatted table to terminal."""
        print("\n" + "=" * 100)
        print("FLYBACK IMPACT ANALYSIS - SUMMARY TABLE")
        print("=" * 100)

        # Header
        print(
            f"{'Configuration':<16} {'Runs':<5} {'Impulse':<12} {'STD':<8} {'% vs':<8} "
            f"{'Max Force':<12} {'STD':<8} {'Impact Energy':<14} {'% vs':<8}"
        )
        print(
            f"{'':16} {'':5} {'[kN·s]':<12} {'':8} {'STND':<8} "
            f"{'[kN]':<12} {'':8} {'[MJ]':<14} {'STND':<8}"
        )
        print("-" * 100)

        # Data rows in specific order
        for config in ["STND", "DF", "DS", "SL", "BR"]:
            if config in config_stats:
                stats = config_stats[config]
                print(
                    f"{stats['name']:<16} {stats['n_runs']:<5} "
                    f"{stats['impulse_mean']:<12.2f} {stats['impulse_std']:<8.2f} "
                    f"{stats['impulse_pct']:<+8.0f}% "
                    f"{stats['force_mean']:<12.2f} {stats['force_std']:<8.2f} "
                    f"{stats['energy_mean']:<14.0f} {stats['energy_pct']:<+8.0f}%"
                )

        print("-" * 100)
        print("KEY FINDINGS:")

        # Calculate and display key insights
        if "STND" in config_stats and "BR" in config_stats:
            stnd_energy = config_stats["STND"]["energy_mean"]
            br_energy = config_stats["BR"]["energy_mean"]
            reduction = ((stnd_energy - br_energy) / stnd_energy) * 100
            print(
                f"• Breakaway reduces impact energy by {reduction:.0f}% " f"vs Standard"
            )

        if "STND" in config_stats:
            best_alternative = None
            best_reduction = 0
            for config in ["DF", "DS", "SL"]:
                if config in config_stats:
                    reduction = abs(config_stats[config]["energy_pct"])
                    if reduction < best_reduction or best_alternative is None:
                        best_alternative = config
                        best_reduction = reduction

            if best_alternative:
                print(
                    f"• {config_stats[best_alternative]['name']} shows best compromise "
                    f"({best_reduction:.0f}% energy reduction)"
                )

        print("\nNOTE: Impact Energy calculated assuming inelastic collision")
        print("=" * 100 + "\n")

    def _write_table_to_file(self, f, config_stats: Dict[str, Any]) -> None:
        """Write formatted table to file."""
        f.write(
            f"{'Configuration':<16} {'Runs':<5} {'Impulse':<12} {'STD':<8} {'% vs':<8} "
            f"{'Max Force':<12} {'STD':<8} {'Impact Energy':<14} {'% vs':<8}\n"
        )
        f.write(
            f"{'':16} {'':5} {'[kN·s]':<12} {'':8} {'STND':<8} "
            f"{'[kN]':<12} {'':8} {'[MJ]':<14} {'STND':<8}\n"
        )
        f.write("-" * 100 + "\n")

        for config in ["STND", "DF", "DS", "SL", "BR"]:
            if config in config_stats:
                stats = config_stats[config]
                f.write(
                    f"{stats['name']:<16} {stats['n_runs']:<5} "
                    f"{stats['impulse_mean']:<12.2f} {stats['impulse_std']:<8.2f} "
                    f"{stats['impulse_pct']:<+8.0f}% "
                    f"{stats['force_mean']:<12.2f} {stats['force_std']:<8.2f} "
                    f"{stats['energy_mean']:<14.0f} {stats['energy_pct']:<+8.0f}%\n"
                )

        f.write("-" * 100 + "\n")

    def generate_latex_table(self, results_df: pd.DataFrame, output_dir: str) -> None:
        """Generate LaTeX table with comprehensive statistics.

        Args:
            results_df: DataFrame with test results
            output_dir: Output directory path

        Returns:
            dict: Configuration statistics
        """
        # Calculate detailed statistics for each configuration
        config_stats = {}

        # Configuration name mapping for display
        config_names = {
            "STND": "Standard (SD)",
            "DF": "Dual Fixed (DF)",
            "DS": "Dual Sliding",
            "SL": "Sliding (SL)",
            "BR": "Breakaway (BR)",
        }

        # Calculate stats for each configuration
        for config in ["STND", "DF", "DS", "SL", "BR"]:
            config_data = results_df[results_df["test_type"] == config]
            if config_data.empty:
                continue

            # Convert units: lbf to kN for force, lbf·s to kN·s for impulse
            impulse_kns = config_data["J"] * 4.44822 / 1000  # lbf·s to kN·s
            force_kn = config_data["F"] * 4.44822 / 1000  # lbf to kN

            # Calculate impact energy (assuming inelastic collision)
            # E = (impulse^2) / (2 * mass), convert to MJ
            mass_kg = self.config_masses.get(config, 45) * 6.85218e-5  # to kg
            energy_mj = (
                ((config_data["J"] * 4.44822) ** 2) / (2 * mass_kg) / 1e6
            )  # to MJ

            config_stats[config] = {
                "name": config_names[config],
                "n_runs": len(config_data),
                "impulse_mean": impulse_kns.mean(),
                "impulse_std": impulse_kns.std(),
                "force_mean": force_kn.mean(),
                "force_std": force_kn.std(),
                "energy_mean": energy_mj.mean(),
                "energy_std": energy_mj.std(),
            }

        # Calculate percentages vs STND
        stnd_stats = config_stats.get("STND", {})
        if stnd_stats:
            for config in config_stats:
                if config != "STND":
                    config_stats[config]["impulse_pct"] = (
                        (
                            config_stats[config]["impulse_mean"]
                            / stnd_stats["impulse_mean"]
                        )
                        - 1
                    ) * 100
                    config_stats[config]["energy_pct"] = (
                        (
                            config_stats[config]["energy_mean"]
                            / stnd_stats["energy_mean"]
                        )
                        - 1
                    ) * 100
                else:
                    config_stats[config]["impulse_pct"] = 0
                    config_stats[config]["energy_pct"] = 0

        # Generate LaTeX table
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        latex_file = output_path / "flyback_results_table.tex"

        with open(latex_file, "w") as f:
            f.write("\\begin{table}[htb]\n")
            f.write("\\centering\n")
            f.write("\\scriptsize\n")
            f.write("\\begin{tabular}{l|c|c|c|c|c|c|c|c}\n")
            f.write("\\hline\n")
            f.write(
                "Configuration & \\#Runs & Impulse & STD & \\% vs & "
                "Max Force & STD & Avg. Impact & \\% vs \\\\\n"
            )
            f.write(
                "              &        & [kN·s]  &     & STND  & "
                "[kN]      &     & Energy [MJ] & STND \\\\\n"
            )
            f.write("\\hline\n")

            # Write data rows in specific order
            for config in ["STND", "DF", "DS", "SL", "BR"]:
                if config in config_stats:
                    stats = config_stats[config]
                    f.write(
                        "{name} & {n_runs} & "
                        "{impulse_mean:.2f} & {impulse_std:.2f} & "
                        "{impulse_pct:+.0f}\\% & "
                        "{force_mean:.2f} & {force_std:.2f} & "
                        "{energy_mean:.0f} & {energy_pct:+.0f}\\% \\\\n".format(
                            name=stats["name"],
                            n_runs=stats["n_runs"],
                            impulse_mean=stats["impulse_mean"],
                            impulse_std=stats["impulse_std"],
                            impulse_pct=stats["impulse_pct"],
                            force_mean=stats["force_mean"],
                            force_std=stats["force_std"],
                            energy_mean=stats["energy_mean"],
                            energy_pct=stats["energy_pct"],
                        )
                    )

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(
                "\\caption{Flyback impact performance comparison across "
                "gear configurations. Impact Energy calculated assuming "
                "inelastic collision.}\n"
            )
            f.write("\\label{tab:flyback_results}\n")
            f.write("\\end{table}\n")

        # Also generate a readable summary file
        summary_file = output_path / "table_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Flyback Impact Analysis - Table Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(
                f"{'Configuration':<15} {'Runs':<5} {'Impulse':<10} "
                f"{'±STD':<8} {'%STND':<8} {'Force':<10} {'±STD':<8} "
                f"{'Energy':<10} {'%STND':<8}\n"
            )
            f.write(
                f"{'':15} {'':5} {'(kN·s)':<10} {'':8} {'':8} "
                f"{'(kN)':<10} {'':8} {'(MJ)':<10} {'':8}\n"
            )
            f.write("-" * 90 + "\n")

            for config in ["STND", "DF", "DS", "SL", "BR"]:
                if config in config_stats:
                    stats = config_stats[config]
                    f.write(
                        f"{stats['name']:<15} {stats['n_runs']:<5} "
                        f"{stats['impulse_mean']:<10.2f} {stats['impulse_std']:<8.2f} "
                        f"{stats['impulse_pct']:<+8.0f}% "
                        f"{stats['force_mean']:<10.2f} {stats['force_std']:<8.2f} "
                        f"{stats['energy_mean']:<10.0f} {stats['energy_pct']:<+8.0f}%\n"
                    )

        self.log.info(f"LaTeX table saved to: {latex_file}")
        self.log.info(f"Summary table saved to: {summary_file}")

        return config_stats

    def save_results_to_csv(self, results: pd.DataFrame, output_dir: str) -> None:
        """Save summary results to CSV file.

        Args:
            results: Summary statistics DataFrame
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        csv_path = output_path / "results.csv"
        results.to_csv(csv_path, index=False)
        self.log.info(f"Results saved to: {csv_path}")

    def generate_summary_report(self, results: pd.DataFrame, output_dir: str) -> None:
        """Generate a summary report.

        Args:
            results: Summary statistics DataFrame
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "summary_report.txt"

        with open(report_path, "w") as f:
            f.write("Fishing Line Flyback Impact Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")

            for _, row in results.iterrows():
                f.write(f"Test Type: {row['test_type']}\n")
                f.write(
                    f"  Mean Impulse: {row['mean_J']:.3f} ± {row['std_J']:.3f} Ns\n"
                )
                f.write(f"  Mean Force: {row['mean_F']:.3f} ± {row['std_F']:.3f} N\n")
                f.write(f"  Impulse % diff from STND: {row['%diff_J'] * 100:.1f}%\n")
                f.write(f"  Force % diff from STND: {row['%diff_F'] * 100:.1f}%\n")
                f.write("\n")

        self.log.info(f"Summary report saved to: {report_path}")
