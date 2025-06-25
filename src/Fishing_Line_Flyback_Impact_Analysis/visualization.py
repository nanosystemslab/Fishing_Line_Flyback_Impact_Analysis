"""Visualization module for fishing line flyback impact analysis."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ImpactVisualizer:
    """Visualizer for fishing line flyback impact properties."""

    def __init__(self, output_dir: str = "out"):
        """Initialize the ImpactVisualizer.

        Args:
            output_dir: Directory for saving plots and outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log = logging.getLogger(__name__)

        # Plot parameters configuration
        self.plot_params = {
            "time": {"label": "Time (s)", "unit": "Time (s)"},
            "S1": {"label": "Sensor 1 (N)", "unit": "Sensor 1 (N)"},
            "S2": {"label": "Sensor 2 (N)", "unit": "Sensor 2 (N)"},
            "S3": {"label": "Sensor 3 (N)", "unit": "Sensor 3 (N)"},
            "S4": {"label": "Sensor 4 (N)", "unit": "Sensor 4 (N)"},
            "SUM": {"label": "Sum (N)", "unit": "Sum (N)"},
            "Max": {"label": "Max (lbf)", "unit": "Max (lbf)"},
            "All": {"label": "Force (N)", "unit": "Force (N)"},
            "J": {"label": "Impulse (kN·s)", "unit": "Impulse (kN·s)"},
            "F": {"label": "Max Force (kN)", "unit": "Max Force (kN)"},
        }

        # Default plot settings
        self.figsize = 4
        self.figdpi = 600
        self.hwratio = 4.0 / 3.0

    def plot_time_series(
        self,
        df: pd.DataFrame,
        param_y: str = "SUM",
        param_x: str = "time",
        show_all_sensors: bool = False,
    ) -> None:
        """Plot time series data from impact test.

        Args:
            df: DataFrame with impact data
            param_y: Y parameter to plot
            param_x: X parameter to plot
            show_all_sensors: If True, plot all individual sensors
        """
        config = df.meta.config
        diam = df.meta.diam
        run_num = df.meta.run_num

        # Prepare data
        x = df[param_x].values
        if param_x == "time":
            x = x - x.min()  # Start time from zero

        # Calculate properties for annotations
        if param_y == "All":
            # Use SUM when "All" is specified
            y = df["SUM"].values
        elif param_y == "SUM":
            y = df["SUM"].values
        else:
            y = df[param_y].values

        max_force = np.max(y)
        try:
            # Use trapezoid if available (newer numpy)
            impulse = np.trapezoid(y, x)
        except AttributeError:
            # Fallback to trapz for older numpy
            impulse = np.trapz(y, x)

        # Create plot
        fig = plt.figure(
            figsize=(self.figsize * self.hwratio, self.figsize), dpi=self.figdpi
        )
        ax = fig.add_subplot(111)

        with sns.axes_style("darkgrid"):
            if show_all_sensors or param_y == "All":
                # Plot all individual sensors
                sensor_cols = ["S1", "S2", "S3", "S4"]
                for sensor in sensor_cols:
                    if sensor in df.columns:
                        sensor_x = df[param_x].values
                        if param_x == "time":
                            sensor_x = sensor_x - sensor_x.min()
                        ax.plot(sensor_x, df[sensor], label=f"Sensor {sensor[-1]}")
                        print(f"{sensor} max: {df[sensor].max():.2f}")
            else:
                # Plot single trace
                ax.plot(x, y, label="Measured Signal")

            # Add property annotations
            ax.plot([], [], " ", label=f"Max Force = {max_force:.2f} N")
            ax.plot([], [], " ", label=f"Impulse = {impulse:.3f} Ns")

            ax.legend()
            ax.set_xlabel(self.plot_params[param_x]["label"])
            ax.set_ylabel(self.plot_params[param_y]["label"])

            # Save plot
            plot_filename = (
                f"plot-{param_x}_vs_{param_y}--{config}-{diam}-{run_num}.png"
            )
            plot_path = self.output_dir / plot_filename

            self.log.info(f"Saving plot to: {plot_path}")
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)

    def plot_box_plots(
        self, df: pd.DataFrame, param: str = "F", convert_units: bool = True
    ) -> None:
        """Create box plots for comparing test configurations.

        Args:
            df: DataFrame with results data
            param: Parameter to plot ("F" for force or "J" for impulse)
            convert_units: Convert to kN/kN·s units
        """
        # Prepare data
        plot_df = df.copy()

        if convert_units:
            if param == "J":
                plot_df["J"] = (plot_df["J"] * 4.44822) / 1e3  # Convert to kN·s
            elif param == "F":
                plot_df["F"] = (plot_df["F"] * 4.44822) / 1e3  # Convert to kN

        # Create plot
        fig = plt.figure(
            figsize=(self.figsize * self.hwratio, self.figsize), dpi=self.figdpi
        )
        ax = fig.add_subplot(111)

        with sns.axes_style("whitegrid"):
            # Replace test type names for display
            display_names = {
                "BR": "Break Away",
                "DF": "Dual Fixed",
                "DS": "Dual Sliding",
                "SL": "Sliding",
                "STND": "Standard",
            }
            plot_data = plot_df.replace({"test_type": display_names})

            sns.boxplot(data=plot_data, y="test_type", x=param, hue="test_type", ax=ax)

            ax.set_xlabel(self.plot_params[param]["label"])
            ax.set_ylabel("Test Configuration Type")

            # Save plot
            plot_filename = f"plot-box-{param}.png"
            plot_path = self.output_dir / plot_filename

            self.log.info(f"Saving box plot to: {plot_path}")
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)

    def plot_violin_plots(self, df: pd.DataFrame, param: str = "F") -> None:
        """Create violin plots for comparing test configurations.

        Args:
            df: DataFrame with results data
            param: Parameter to plot ("F" for force or "J" for impulse)
        """
        fig = plt.figure(
            figsize=(self.figsize * self.hwratio, self.figsize), dpi=self.figdpi
        )
        ax = fig.add_subplot(111)

        with sns.axes_style("whitegrid"):
            sns.violinplot(data=df, y="test_type", x=param, hue="test_type", ax=ax)

            ax.set_xlabel(self.plot_params[param]["label"])
            ax.set_ylabel("Test Configuration Type")

            # Save plot
            plot_filename = f"plot-violin-{param}.png"
            plot_path = self.output_dir / plot_filename

            self.log.info(f"Saving violin plot to: {plot_path}")
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)

    def plot_dual_box_plots(self, df: pd.DataFrame, save_format: str = "png") -> None:
        """Create side-by-side box plots for impulse and force.

        Args:
            df: DataFrame with results data
            save_format: Format to save ("png" or "svg")
        """
        # Prepare data with unit conversion
        plot_df = df.copy()
        plot_df["J"] = (plot_df["J"] * 4.44822) / 1e3  # Convert to kN·s
        plot_df["F"] = (plot_df["F"] * 4.44822) / 1e3  # Convert to kN

        # Create subplots
        figsize = 4
        hwratio = 16.0 / 10
        fig, axes = plt.subplots(
            1, 2, figsize=(figsize * hwratio * 2, figsize), dpi=self.figdpi
        )
        fig.subplots_adjust(wspace=0.95)

        with sns.axes_style("whitegrid"):
            # Plot impulse on left
            sns.boxplot(ax=axes[0], data=plot_df, y="test_type", x="J", hue="test_type")
            axes[0].set_xlabel(self.plot_params["J"]["label"], fontsize=20)
            axes[0].set_ylabel(" ", fontsize=1)
            axes[0].tick_params(axis="both", which="major", labelsize=20)

            # Plot force on right
            sns.boxplot(ax=axes[1], data=plot_df, y="test_type", x="F", hue="test_type")
            axes[1].set_xlabel(self.plot_params["F"]["label"], fontsize=20)
            axes[1].set_ylabel("")
            axes[1].tick_params(labelleft=False)
            axes[1].tick_params(
                axis="both", which="major", labelsize=20, labelleft=False
            )

            # Save plot
            plot_filename = f"plot-box-J-F.{save_format}"
            plot_path = self.output_dir / plot_filename

            self.log.info(f"Saving dual box plot to: {plot_path}")

            if save_format == "svg":
                fig.savefig(
                    plot_path, format="svg", bbox_inches="tight", transparent=True
                )
            else:
                fig.savefig(plot_path, bbox_inches="tight")

            plt.close(fig)

    def create_summary_plots(self, results_df: pd.DataFrame) -> None:
        """Create all summary visualization plots.

        Args:
            results_df: DataFrame with test results
        """
        self.log.info("Creating summary plots...")

        # Box plots for individual parameters
        self.plot_box_plots(results_df, param="F")
        self.plot_box_plots(results_df, param="J")

        # Violin plots
        self.plot_violin_plots(results_df, param="F")
        self.plot_violin_plots(results_df, param="J")

        # Dual box plot
        self.plot_dual_box_plots(results_df, save_format="png")
        self.plot_dual_box_plots(results_df, save_format="svg")

        self.log.info("Summary plots completed")

    def _get_mass_for_config(self, config: str) -> float:
        """Get mass value for configuration type.

        Args:
            config: Configuration type string

        Returns:
            Mass value in original units
        """
        config_masses = {"STND": 49, "DF": 62, "DS": 75, "SL": 65, "BR": 45}
        return config_masses.get(config, 45)

    def plot_output_data(
        self, filepath: str, x_param: str = "D", y_param: str = "KE"
    ) -> None:
        """Plot output data from CSV file.

        Args:
            filepath: Path to output CSV file
            x_param: X-axis parameter
            y_param: Y-axis parameter
        """
        # Load output data (headerless CSV)
        df = pd.read_csv(filepath, header=None)

        if len(df.columns) >= 2:
            x_data = df.iloc[:, 0].values
            y_data = df.iloc[:, 1].values

            fig = plt.figure(
                figsize=(self.figsize * self.hwratio, self.figsize), dpi=self.figdpi
            )
            ax = fig.add_subplot(111)

            ax.scatter(x_data, y_data, alpha=0.7)
            ax.set_xlabel(self.plot_params.get(x_param, {"label": x_param})["label"])
            ax.set_ylabel(self.plot_params.get(y_param, {"label": y_param})["label"])

            plot_filename = f"output-{x_param}_vs_{y_param}.png"
            plot_path = self.output_dir / plot_filename

            self.log.info(f"Saving output plot to: {plot_path}")
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
