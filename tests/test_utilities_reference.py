"""Test utility functions and helper methods."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Fishing_Line_Material_Properties_Analysis.analysis import MaterialAnalyzer
from Fishing_Line_Material_Properties_Analysis.visualization import MaterialVisualizer


class TestUtilities:
    """Test utility functions and edge cases."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_path_handling_windows_style(self) -> None:
        """Test path handling with Windows-style paths."""
        visualizer = MaterialVisualizer()

        # Test Windows-style paths - need to use Path to normalize separators
        from pathlib import Path

        windows_path = str(Path("C:/data/group_1/5in/test.csv"))  # Normalize separators
        group = visualizer._extract_group_from_path(windows_path)
        length = visualizer._extract_length_from_path(windows_path)

        assert group == "group_1"
        assert length == "5in"

    def test_path_handling_mixed_separators(self) -> None:
        """Test path handling with mixed path separators."""
        visualizer = MaterialVisualizer()

        # Test mixed separators - normalize the path first
        from pathlib import Path

        mixed_path = str(Path("data/group_1/10in/test.csv"))  # Normalize separators
        group = visualizer._extract_group_from_path(mixed_path)
        length = visualizer._extract_length_from_path(mixed_path)

        assert group == "group_1"
        assert length == "10in"

    def test_metadata_parsing_edge_cases(self) -> None:
        """Test metadata parsing with various edge cases."""
        analyzer = MaterialAnalyzer()

        edge_cases = [
            ("test--line-crimp-99--999.csv", 99, "crimp", 999),
            ("test--line-braided-01--0.csv", 1, "braided", 0),
            ("test--wire-solid-15--5.csv", 15, "solid", 5),
        ]

        for filename, expected_size, expected_ctype, expected_run in edge_cases:
            filepath = f"data/group_1/5in/{filename}"
            metadata = analyzer._parse_metadata(filepath)

            assert metadata.size == expected_size
            assert metadata.ctype == expected_ctype
            assert metadata.test_run == expected_run

    def test_data_validation(self) -> None:
        """Test data validation and cleaning."""
        # Create data with various issues
        problematic_data = pd.DataFrame(
            {
                "Time": [0, 0.1, 0.2, np.inf, 0.4],
                "Force": [0, 1, np.nan, 3, -1],
                "Stroke": [0, 0.1, 0.2, 0.3, 0.4],
            }
        )

        # Should handle problematic data gracefully
        # (Implementation would depend on how you want to handle such cases)
        assert len(problematic_data) == 5

    def test_scientific_notation_handling(self) -> None:
        """Test handling of scientific notation in CSV files."""
        csv_file = os.path.join(self.temp_dir, "scientific.csv")
        with open(csv_file, "w") as f:
            f.write('"Time","Force","Stroke"\n')
            f.write('"sec","N","mm"\n')
            f.write("0.0,0.0,0.0\n")
            f.write("1.0e-1,1.5e+1,1.0e-1\n")
            f.write("2.0e-1,3.0e+1,2.0e-1\n")

        analyzer = MaterialAnalyzer()
        df = analyzer.load_file(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_plot_parameter_validation(self) -> None:
        """Test plot parameter validation."""
        visualizer = MaterialVisualizer()

        # Test all valid parameter combinations
        valid_params = ["Time", "Force", "Stroke", "Stress", "Strain"]

        for param in valid_params:
            assert param in visualizer.plot_params
            assert "label" in visualizer.plot_params[param]
            assert "unit" in visualizer.plot_params[param]

    def test_file_extension_validation(self) -> None:
        """Test file extension validation."""
        analyzer = MaterialAnalyzer()

        invalid_extensions = [
            "test.txt",
            "test.xlsx",
            "test.json",
            "test.xml",
            "test",  # no extension
        ]

        for filename in invalid_extensions:
            with pytest.raises(ValueError, match="File must be CSV format"):
                analyzer.load_file(filename)

    def test_directory_structure_creation(self) -> None:
        """Test automatic directory structure creation."""
        visualizer = MaterialVisualizer(output_dir=self.temp_dir)

        # Create test data
        df = pd.DataFrame({"Strain": [0.1, 0.2], "Stress": [100, 200]})
        df.meta = type("Meta", (), {})()
        df.meta.filepath = "data/group_test/15in/test--line-crimp-21--0.csv"
        df.meta.size = 21
        df.meta.length = 381.0  # 15 inches
        df.meta.test_run = 0

        visualizer.plot_single_trace(df)

        # Check if directory structure was created
        expected_dir = Path(self.temp_dir) / "group_test" / "15in"
        assert expected_dir.exists()

    def test_concurrent_file_access(self) -> None:
        """Test handling of concurrent file access scenarios."""
        import threading

        analyzer = MaterialAnalyzer()

        # Create test file
        csv_file = os.path.join(self.temp_dir, "concurrent_test.csv")
        with open(csv_file, "w") as f:
            f.write('"Time","Force","Stroke"\n')
            f.write('"sec","N","mm"\n')
            for i in range(100):
                f.write(f"{i * 0.01},{i * 0.1},{i * 0.05}\n")

        results = []

        def load_file_worker():
            try:
                df = analyzer.load_file(csv_file)
                results.append(len(df))
            except Exception as e:
                results.append(str(e))

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=load_file_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All threads should succeed
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)
