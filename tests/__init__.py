"""Tests for main package __init__.py functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import Fishing_Line_Flyback_Impact_Analysis as fli  # noqa: N813


class TestPackageImports:
    """Test package-level imports work correctly."""

    def test_core_imports(self):
        """Test that core components are importable."""
        # Core analysis classes
        assert hasattr(fli, "ImpulseAnalyzer")
        assert hasattr(fli, "run_impulse_analysis")
        assert hasattr(fli, "analyze_single_file_with_impulse")

        # Visualization functions
        assert hasattr(fli, "plot_single_file_analysis")
        assert hasattr(fli, "show_force_preview")
        assert hasattr(fli, "create_impulse_boxplots")

        # Shared utilities
        assert hasattr(fli, "load_csv_file")
        assert hasattr(fli, "calculate_total_force")
        assert hasattr(fli, "get_time_array")

    def test_convenience_functions(self):
        """Test convenience functions are available."""
        assert hasattr(fli, "quick_analysis")
        assert hasattr(fli, "batch_analysis")
        assert hasattr(fli, "get_configuration_info")

    def test_gui_functions(self):
        """Test GUI functions are available."""
        assert hasattr(fli, "launch_boundary_viewer")
        assert hasattr(fli, "quick_boundary_check")
        assert hasattr(fli, "gui_info")

    def test_constants_available(self):
        """Test that constants are accessible."""
        assert hasattr(fli, "CONFIG_WEIGHTS")
        assert hasattr(fli, "MATERIAL_NAMES")
        assert hasattr(fli, "LINE_MASS_FRACTION")

    def test_package_metadata(self):
        """Test package metadata."""
        assert hasattr(fli, "__version__")
        assert hasattr(fli, "__author__")
        assert hasattr(fli, "__description__")

        assert fli.__version__ == "1.0.0"
        assert "Nanosystems Lab" in fli.__author__
        assert "Impulse-Focused" in fli.__description__


class TestPackageFunctionality:
    """Test package-level functionality."""

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def test_quick_analysis_integration(self):
        """Test quick_analysis function works end-to-end."""
        # Create test data
        test_data = {
            "AI_Channel_1_lbf": np.random.normal(0, 10, 1000),
            "AI_Channel_2_lbf": np.random.normal(0, 5, 1000),
            "Time": np.linspace(0, 0.1, 1000),
        }
        # Add impact signal
        test_data["AI_Channel_1_lbf"][400:600] = 1000

        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            result = fli.quick_analysis(file_path, show_plot=False)

            assert "error" not in result
            assert "total_impulse" in result
            assert "material_type" in result
            assert result["material_type"] == "STND"

        finally:
            file_path.unlink()

    def test_get_configuration_info(self):
        """Test configuration info function."""
        info = fli.get_configuration_info()

        assert isinstance(info, dict)
        assert "configurations" in info
        assert "line_mass_fraction" in info
        assert "version" in info

        # Check all expected materials
        configs = info["configurations"]
        expected_materials = {"STND", "DF", "DS", "SL", "BR"}
        assert set(configs.keys()) == expected_materials

    def test_gui_info_function(self):
        """Test GUI info function."""
        info = fli.gui_info()

        assert isinstance(info, dict)
        assert "gui_available" in info
        assert "components" in info
        assert "requirements" in info

    def test_package_all_exports(self):
        """Test that __all__ includes all expected exports."""
        assert hasattr(fli, "__all__")
        assert isinstance(fli.__all__, list)

        # Check key functions are in __all__
        expected_in_all = [
            "ImpulseAnalyzer",
            "run_impulse_analysis",
            "quick_analysis",
            "batch_analysis",
            "get_configuration_info",
        ]

        for item in expected_in_all:
            assert item in fli.__all__


class TestPackageConstants:
    """Test package-level constants."""

    def test_config_weights(self):
        """Test CONFIG_WEIGHTS constant."""
        weights = fli.CONFIG_WEIGHTS

        assert isinstance(weights, dict)
        assert len(weights) == 5

        expected_configs = {"STND", "DF", "DS", "SL", "BR"}
        assert set(weights.keys()) == expected_configs

        # Check reasonable weight values
        for _config, weight in weights.items():
            assert isinstance(weight, float)
            assert 0.040 <= weight <= 0.080

    def test_material_names(self):
        """Test MATERIAL_NAMES constant."""
        names = fli.MATERIAL_NAMES

        assert isinstance(names, dict)
        assert len(names) == 5

        # Check mapping consistency
        for code, name in names.items():
            assert code in fli.CONFIG_WEIGHTS
            assert isinstance(name, str)
            assert len(name) > 0

    def test_line_mass_fraction(self):
        """Test LINE_MASS_FRACTION constant."""
        fraction = fli.LINE_MASS_FRACTION

        assert isinstance(fraction, float)
        assert 0.0 < fraction <= 1.0
        assert fraction == 0.70  # Expected value


class TestErrorHandling:
    """Test error handling at package level."""

    def test_import_robustness(self):
        """Test that package imports are robust."""
        # Re-import to test robustness
        import importlib

        importlib.reload(fli)

        # Should still have key components
        assert hasattr(fli, "ImpulseAnalyzer")
        assert hasattr(fli, "CONFIG_WEIGHTS")

    def test_gui_availability_handling(self):
        """Test GUI availability is handled gracefully."""
        # Should not raise exception even if GUI not available
        info = fli.gui_info()
        assert "gui_available" in info

    def test_matplotlib_configuration(self):
        """Test matplotlib configuration doesn't break imports."""
        # Should not raise exception during import
        import matplotlib.pyplot as plt

        # Should have basic configuration
        assert plt.rcParams is not None


class TestVersionCompatibility:
    """Test version compatibility and dependencies."""

    def test_python_version_compatibility(self):
        """Test package works with supported Python versions."""
        import sys

        # Should work with Python 3.11+
        assert sys.version_info >= (3, 11)

    def test_core_dependencies_available(self):
        """Test that core dependencies are available."""
        import matplotlib
        import numpy
        import pandas

        # Should not raise ImportError
        assert numpy.__version__
        assert pandas.__version__
        assert matplotlib.__version__

    def test_optional_dependencies_handling(self):
        """Test optional dependencies are handled gracefully."""
        # GUI dependencies might not be available
        try:
            gui_available = True
        except ImportError:
            gui_available = False

        # Package should still work
        info = fli.gui_info()
        assert info["gui_available"] == gui_available


if __name__ == "__main__":
    pytest.main([__file__])
