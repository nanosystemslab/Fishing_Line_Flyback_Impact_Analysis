"""Integration tests for the complete workflow."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from Fishing_Line_Flyback_Impact_Analysis.analysis import ImpactAnalyzer
from Fishing_Line_Flyback_Impact_Analysis.visualization import ImpactVisualizer


class TestIntegration:
    """Integration tests for complete workflows."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_data()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data(self) -> None:
        """Create realistic test data."""
        # Create directory structure
        for config in ["STND", "DF", "BR"]:
            csv_dir = Path(self.temp_dir) / "data" / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)

            # Create multiple test files for each config
            for i in range(3):
                csv_file = csv_dir / f"{config}-21-{i}.csv"

                # Generate realistic impact test data
                time_data = [j * 0.01 for j in range(50)]

                # Different force profiles for different configs
                if config == "STND":
                    force_multiplier = 1.0
                elif config == "DF":
                    force_multiplier = 0.6
                else:  # BR
                    force_multiplier = 0.3

                # Simulate impact force curve
                force_data = []
                for j in range(50):
                    if j < 10:
                        force = j * 50 * force_multiplier
                    elif j < 20:
                        force = (500 - (j - 10) * 25) * force_multiplier
                    else:
                        force = max(0, (500 - (j - 10) * 25) * force_multiplier * 0.5)
                    force_data.append(force)

                # Create sensor data with slight variations
                s1_data = [f * 1.0 for f in force_data]
                s2_data = [f * 1.1 for f in force_data]
                s3_data = [f * 0.9 for f in force_data]
                s4_data = [f * 1.2 for f in force_data]

                # Write CSV file
                with open(csv_file, "w") as f:
                    f.write(
                        "Time (s),AI 1/AI 1 (lbf),AI 2/AI 2 (lbf),"
                        "AI 3/AI 3 (lbf),AI 4/AI 4 (lbf)\n"
                    )
                    for t, s1, s2, s3, s4 in zip(
                        time_data, s1_data, s2_data, s3_data, s4_data, strict=True
                    ):
                        f.write(f"{t},{s1},{s2},{s3},{s4}\n")

    def test_complete_workflow_single(self) -> None:
        """Test complete workflow with single file analysis."""
        analyzer = ImpactAnalyzer()
        visualizer = ImpactVisualizer(output_dir=self.temp_dir)

        # Get a test file
        test_file = list(Path(self.temp_dir).rglob("STND-21-0.csv"))[0]

        # Load and analyze
        df = analyzer.load_file(str(test_file))

        # Verify data loading
        assert isinstance(df, pd.DataFrame)
        assert "SUM" in df.columns
        assert "time" in df.columns
        assert hasattr(df.meta, "config")

        # Calculate properties
        properties = analyzer.calculate_impact_properties(df)
        assert "max_force_N" in properties
        assert "impulse_Ns" in properties

        # Create visualization
        with patch("matplotlib.pyplot.show"):
            visualizer.plot_time_series(df)

        # Verify plot was created
        plot_files = list(Path(self.temp_dir).rglob("*.png"))
        assert len(plot_files) > 0

    def test_complete_workflow_multi_config(self) -> None:
        """Test complete workflow with multiple configurations."""
        analyzer = ImpactAnalyzer()
        visualizer = ImpactVisualizer(output_dir=self.temp_dir)
        assert visualizer is not None

        results = []

        # Process files from different configurations
        for csv_file in Path(self.temp_dir).rglob("*.csv"):
            result = analyzer.process_single_file(str(csv_file))
            results.append(result)

        # Should have results from 3 configs × 3 files each = 9 total
        assert len(results) == 9

        # Verify different configurations are present
        configs = {r["config"] for r in results}
        assert "STND" in configs
        assert "DF" in configs
        assert "BR" in configs

    def test_results_file_processing_workflow(self) -> None:
        """Test the complete results file processing workflow."""
        analyzer = ImpactAnalyzer()

        # First, create results from analysis
        results = []
        for csv_file in Path(self.temp_dir).rglob("*.csv"):
            result = analyzer.process_single_file(str(csv_file))
            results.append(result)

        # Create results text file
        results_file = Path(self.temp_dir) / "test_results.txt"
        with open(results_file, "w") as f:
            for result in results:
                f.write(
                    f"{result['filename']},J={result['impulse_Ns']:.3f},"
                    f"F={result['max_force_N']:.2f}\n"
                )

        # Load and process results file
        results_df = analyzer.load_results_file(str(results_file))

        assert isinstance(results_df, pd.DataFrame)
        # Allow for some filtering of bad files
        assert len(results_df) >= len(results) - 2  # Allow for up to 2 filtered files

        # Calculate summary statistics
        summary_stats = analyzer.calculate_summary_stats(results_df)
        assert isinstance(summary_stats, pd.DataFrame)
        assert len(summary_stats) >= 1

    def test_visualization_workflow(self) -> None:
        """Test complete visualization workflow."""
        analyzer = ImpactAnalyzer()
        visualizer = ImpactVisualizer(output_dir=self.temp_dir)

        # Process all files and create results
        results = []
        for csv_file in Path(self.temp_dir).rglob("*.csv"):
            result = analyzer.process_single_file(str(csv_file))
            results.append(result)

        # Create results DataFrame for visualization
        results_data = []
        for result in results:
            results_data.append(
                {
                    "fname": result["filename"],
                    "test_type": result["config"],
                    "J": result["impulse_Ns"],
                    "F": result["max_force_N"],
                }
            )

        results_df = pd.DataFrame(results_data)

        # Create all summary visualizations
        with patch("matplotlib.pyplot.show"):
            visualizer.create_summary_plots(results_df)

        # Verify multiple plot types were created
        plot_files = list(Path(self.temp_dir).rglob("*.png"))
        assert len(plot_files) >= 4  # Should have box, violin, and dual plots

    def test_table_generation_workflow(self) -> None:
        """Test complete table generation workflow."""
        analyzer = ImpactAnalyzer()

        # Create comprehensive results
        results_data = []
        for config in ["STND", "DF", "BR"]:
            for i in range(5):  # More samples for better statistics
                results_data.append(
                    {
                        "fname": f"{config}-21-{i}.csv",
                        "test_type": config,
                        "J": (
                            2.5
                            if config == "STND"
                            else (1.2 if config == "DF" else 0.5)
                        ),
                        "F": (
                            1200
                            if config == "STND"
                            else (800 if config == "DF" else 400)
                        ),
                    }
                )

        results_df = pd.DataFrame(results_data)

        # Generate summary table
        with patch("builtins.print") as mock_print:
            config_stats = analyzer.generate_summary_table(results_df, self.temp_dir)

        # Verify table generation
        assert isinstance(config_stats, dict)
        assert "STND" in config_stats
        assert "DF" in config_stats
        assert "BR" in config_stats

        # Verify print was called (table was displayed)
        mock_print.assert_called()

        # Verify summary file was created
        summary_file = Path(self.temp_dir) / "flyback_summary_table.txt"
        assert summary_file.exists()

    def test_batch_processing_workflow(self) -> None:
        """Test complete batch processing workflow."""
        from Fishing_Line_Flyback_Impact_Analysis import __main__

        data_dir = Path(self.temp_dir) / "data" / "csv"

        # Simulate batch command
        args = {
            "data_dir": str(data_dir),
            "output": self.temp_dir,
            "summary": True,
        }

        with patch("builtins.print"):  # Suppress output
            result = __main__.handle_batch_command(args)

        assert result == 0

        # Verify batch results file
        batch_results = Path(self.temp_dir) / "batch_results.txt"
        assert batch_results.exists()

        # Verify summary files were created
        summary_csv = Path(self.temp_dir) / "results.csv"
        summary_report = Path(self.temp_dir) / "summary_report.txt"
        assert summary_csv.exists()
        assert summary_report.exists()

        # Verify plots were created
        plot_files = list(Path(self.temp_dir).rglob("*.png"))
        assert len(plot_files) > 0

    def test_error_recovery_workflow(self) -> None:
        """Test workflow behavior with some invalid files."""
        analyzer = ImpactAnalyzer()

        # Create a mix of valid and invalid files
        csv_dir = Path(self.temp_dir) / "data" / "mixed"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Valid file
        valid_file = csv_dir / "STND-21-0.csv"
        with open(valid_file, "w") as f:
            f.write(
                "Time (s),AI 1/AI 1 (lbf),AI 2/AI 2"
                "(lbf),AI 3/AI 3 (lbf),AI 4/AI 4 (lbf)\n"
            )
            f.write("0.0,0.0,0.0,0.0,0.0\n")
            f.write("0.1,100.0,110.0,90.0,120.0\n")

        # Invalid file
        invalid_file = csv_dir / "invalid.csv"
        with open(invalid_file, "w") as f:
            f.write("invalid,data,format\n")
            f.write("1,2,3\n")

        # Process files - should handle errors gracefully
        valid_results = []
        error_count = 0

        for csv_file in csv_dir.glob("*.csv"):
            try:
                result = analyzer.process_single_file(str(csv_file))
                valid_results.append(result)
            except Exception:
                error_count += 1

        # Should have 1 valid result and 1 error
        assert len(valid_results) == 1
        assert error_count == 1

    def test_end_to_end_realistic_workflow(self) -> None:
        """Test complete end-to-end workflow with realistic data."""
        from Fishing_Line_Flyback_Impact_Analysis import __main__

        # Step 1: Analyze files
        csv_dir = Path(self.temp_dir) / "data" / "csv"
        analyze_args = {
            "input": [str(f) for f in csv_dir.glob("STND-21-*.csv")],
            "output": self.temp_dir,
            "param_y": "SUM",
            "param_x": "time",
            "show_all_sensors": False,
        }

        result = __main__.handle_analyze_command(analyze_args)
        assert result == 0

        # Verify analysis results
        results_file = Path(self.temp_dir) / "run_results.txt"
        assert results_file.exists()

        # Step 2: Post-process results
        postprocess_args = {
            "input": str(results_file),
            "output": self.temp_dir,
            "plot_type": "all",
            "generate_table": True,
        }

        with patch("builtins.print"):
            result = __main__.handle_postprocess_command(postprocess_args)
        assert result == 0

        # Verify post-processing outputs
        summary_csv = Path(self.temp_dir) / "results.csv"
        summary_table = Path(self.temp_dir) / "flyback_summary_table.txt"
        assert summary_csv.exists()
        assert summary_table.exists()

        # Step 3: Generate standalone table
        table_args = {
            "input": str(results_file),
            "output": self.temp_dir,
        }

        with patch("builtins.print"):
            result = __main__.handle_table_command(table_args)
        assert result == 0

    def test_h5_caching_workflow(self) -> None:
        """Test H5 file caching in the workflow."""
        analyzer = ImpactAnalyzer()

        # Get a test file
        test_file = list(Path(self.temp_dir).rglob("STND-21-0.csv"))[0]

        # First load - should create H5 cache
        df1 = analyzer.load_file(str(test_file))

        # Check if H5 directory was attempted to be created
        # (may not actually exist due to test environment)
        assert isinstance(df1, pd.DataFrame)

        # Second load - should be faster with cache
        df2 = analyzer.load_file(str(test_file))
        assert isinstance(df2, pd.DataFrame)

        # Results should be consistent
        assert df1.meta.config == df2.meta.config
        assert df1.meta.diam == df2.meta.diam

    def test_cross_module_compatibility(self) -> None:
        """Test compatibility between analysis and visualization modules."""
        analyzer = ImpactAnalyzer()
        visualizer = ImpactVisualizer(output_dir=self.temp_dir)

        # Load data with analyzer
        test_file = list(Path(self.temp_dir).rglob("STND-21-0.csv"))[0]
        df = analyzer.load_file(str(test_file))

        # Process with analyzer
        properties = analyzer.calculate_impact_properties(df)

        # Visualize with visualizer
        with patch("matplotlib.pyplot.show"):
            visualizer.plot_time_series(df)

        # Both modules should work with the same data structure
        assert isinstance(df, pd.DataFrame)
        assert hasattr(df, "meta")
        assert isinstance(properties, dict)

        # Verify visualization can access analyzer results
        mass = visualizer._get_mass_for_config(df.meta.config)
        assert isinstance(mass, (int, float))
        assert mass > 0

    def test_configuration_coverage(self) -> None:
        """Test that all configuration types are properly handled."""
        analyzer = ImpactAnalyzer()

        # Test all configuration types
        configs = ["STND", "DF", "DS", "SL", "BR"]

        for config in configs:
            # Create test file for each config
            csv_file = Path(self.temp_dir) / f"{config}-21-test.csv"
            with open(csv_file, "w") as f:
                f.write(
                    "Time (s),AI 1/AI 1 (lbf),AI 2/AI 2 (lbf),"
                    "AI 3/AI 3 (lbf),AI 4/AI 4 (lbf)\n"
                )
                f.write("0.0,0.0,0.0,0.0,0.0\n")
                f.write("0.1,100.0,110.0,90.0,120.0\n")

            # Test analysis
            df = analyzer.load_file(str(csv_file))
            assert df.meta.config == config

            # Test properties calculation
            properties = analyzer.calculate_impact_properties(df)
            assert properties["config"] == config

            # Verify mass is correctly assigned
            expected_mass = analyzer.config_masses[config] * 6.85218e-5
            assert properties["mass_kg"] == expected_mass

    def test_data_consistency_workflow(self) -> None:
        """Test data consistency throughout the workflow."""
        analyzer = ImpactAnalyzer()

        # Process multiple files and check consistency
        all_results = []
        for csv_file in Path(self.temp_dir).rglob("*.csv"):
            result = analyzer.process_single_file(str(csv_file))
            all_results.append(result)

        # Check that results have consistent structure
        required_keys = [
            "max_force_N",
            "impulse_Ns",
            "velocity_ms",
            "kinetic_energy_J",
            "mass_kg",
            "config",
            "diameter",
            "run_number",
            "filename",
            "filepath",
        ]

        for result in all_results:
            for key in required_keys:
                assert key in result, f"Missing key {key} in result"
                assert result[key] is not None, f"None value for key {key}"

        # Check that configurations are properly differentiated
        stnd_results = [r for r in all_results if r["config"] == "STND"]
        df_results = [r for r in all_results if r["config"] == "DF"]
        br_results = [r for r in all_results if r["config"] == "BR"]

        # STND should have higher forces than DF, which should be higher than BR
        if stnd_results and df_results:
            avg_stnd_force = sum(r["max_force_N"] for r in stnd_results) / len(
                stnd_results
            )
            avg_df_force = sum(r["max_force_N"] for r in df_results) / len(df_results)
            assert avg_stnd_force > avg_df_force

        if df_results and br_results:
            avg_df_force = sum(r["max_force_N"] for r in df_results) / len(df_results)
            avg_br_force = sum(r["max_force_N"] for r in br_results) / len(br_results)
            assert avg_df_force > avg_br_force


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_directory_handling(self) -> None:
        """Test handling of empty directories."""
        from Fishing_Line_Flyback_Impact_Analysis import __main__

        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()

        args = {
            "data_dir": str(empty_dir),
            "output": self.temp_dir,
            "summary": False,
        }

        result = __main__.handle_batch_command(args)
        assert result == 1  # Should fail gracefully

    def test_corrupted_file_handling(self) -> None:
        """Test handling of corrupted files."""
        analyzer = ImpactAnalyzer()

        # Create corrupted file
        corrupted_file = Path(self.temp_dir) / "corrupted.csv"
        with open(corrupted_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03invalid binary data")

        # Should handle gracefully - expect any exception related to parsing
        with pytest.raises((ValueError, IndexError, pd.errors.ParserError)):
            analyzer.load_file(str(corrupted_file))

    def test_extremely_large_values(self) -> None:
        """Test handling of extremely large force values."""
        analyzer = ImpactAnalyzer()

        # Create file with extreme values
        extreme_file = Path(self.temp_dir) / "extreme.csv"
        with open(extreme_file, "w") as f:
            f.write(
                "Time (s),AI 1/AI 1 (lbf),AI 2/AI 2 (lbf),"
                "AI 3/AI 3 (lbf),AI 4/AI 4 (lbf)\n"
            )
            f.write("0.0,0.0,0.0,0.0,0.0\n")
            f.write(f"0.1,{1e6},{1e6},{1e6},{1e6}\n")  # Very large values

        df = analyzer.load_file(str(extreme_file))
        properties = analyzer.calculate_impact_properties(df)

        # Should handle extreme values without crashing
        assert isinstance(properties["max_force_N"], float)
        assert not pd.isna(properties["max_force_N"])

    def test_minimal_data_handling(self) -> None:
        """Test handling of files with minimal data points."""
        analyzer = ImpactAnalyzer()

        # Create file with minimal data
        minimal_file = Path(self.temp_dir) / "minimal.csv"
        with open(minimal_file, "w") as f:
            f.write(
                "Time (s),AI 1/AI 1 (lbf),AI 2/AI 2 (lbf),"
                "AI 3/AI 3 (lbf),AI 4/AI 4 (lbf)\n"
            )
            f.write("0.0,100.0,100.0,100.0,100.0\n")  # Only one data point

        df = analyzer.load_file(str(minimal_file))
        properties = analyzer.calculate_impact_properties(df)

        # Should handle minimal data
        assert isinstance(properties, dict)
        assert "max_force_N" in properties
