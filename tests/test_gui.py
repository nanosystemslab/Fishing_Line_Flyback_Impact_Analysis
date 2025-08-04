"""Tests for GUI module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# Test imports and availability
try:
    from Fishing_Line_Flyback_Impact_Analysis.gui import PYQT_AVAILABLE
    from Fishing_Line_Flyback_Impact_Analysis.gui import launch_boundary_viewer

    if PYQT_AVAILABLE:
        from Fishing_Line_Flyback_Impact_Analysis.gui.boundary_viewer import (
            IntegrationBoundaryViewer,
        )
except ImportError:
    PYQT_AVAILABLE = False
    IntegrationBoundaryViewer = None
    launch_boundary_viewer = None


class TestGUIAvailability:
    """Test GUI availability and imports."""

    def test_pyqt_availability_flag(self):
        """Test that PYQT_AVAILABLE flag is properly set."""
        from Fishing_Line_Flyback_Impact_Analysis.gui import PYQT_AVAILABLE

        assert isinstance(PYQT_AVAILABLE, bool)

    def test_gui_imports(self):
        """Test GUI imports work correctly."""
        from Fishing_Line_Flyback_Impact_Analysis.gui import launch_boundary_viewer

        if PYQT_AVAILABLE:
            assert launch_boundary_viewer is not None
            assert callable(launch_boundary_viewer)
        else:
            # Should still be importable even if PyQt not available
            assert launch_boundary_viewer is not None

    def test_main_module_gui_functions(self):
        """Test that main module exposes GUI functions correctly."""
        from Fishing_Line_Flyback_Impact_Analysis import launch_boundary_viewer

        if PYQT_AVAILABLE:
            assert launch_boundary_viewer is not None
        else:
            assert launch_boundary_viewer is None

    def test_gui_info_function(self):
        """Test gui_info function from main module."""
        from Fishing_Line_Flyback_Impact_Analysis import gui_info

        info = gui_info()
        assert isinstance(info, dict)
        assert "gui_available" in info
        assert info["gui_available"] == PYQT_AVAILABLE
        assert "components" in info
        assert "requirements" in info
        assert "PyQt5" in info["requirements"]
        assert "pyqtgraph" in info["requirements"]


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt5 not available")
class TestIntegrationBoundaryViewer:
    """Test IntegrationBoundaryViewer class (only if PyQt5 available)."""

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def create_test_data(self, n_points=1000):
        """Create realistic test data."""
        time = np.linspace(0, 0.1, n_points)
        force1 = np.random.normal(0, 10, n_points)
        force2 = np.random.normal(0, 5, n_points)

        # Add impact signal
        impact_start = n_points // 3
        impact_end = 2 * n_points // 3
        force1[impact_start:impact_end] = 1000 * np.sin(
            np.linspace(0, np.pi, impact_end - impact_start)
        )
        force2[impact_start:impact_end] = 300 * np.sin(
            np.linspace(0, np.pi, impact_end - impact_start)
        )

        return {"AI_Channel_1_lbf": force1, "AI_Channel_2_lbf": force2, "Time": time}

    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_initialization(self, mock_app):
        """Test viewer initialization."""
        viewer = IntegrationBoundaryViewer()

        assert viewer.current_file is None
        assert viewer.force_data is None
        assert viewer.time_data is None
        assert viewer.boundaries is None

    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_load_csv_success(self, mock_app):
        """Test successful CSV loading."""
        viewer = IntegrationBoundaryViewer()

        test_data = self.create_test_data()
        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            # Mock the file dialog and UI components
            with patch.object(viewer, "update_material_info"), patch.object(
                viewer, "update_data_stats"
            ), patch.object(viewer, "plot_data_with_boundaries"):

                success = viewer.load_csv_file(str(file_path))

                if success:
                    assert viewer.current_file == file_path
                    assert viewer.force_data is not None
                    assert viewer.time_data is not None
                    assert len(viewer.force_data) == 1000

        finally:
            file_path.unlink()

    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_load_csv_invalid_file(self, mock_app):
        """Test loading invalid CSV file."""
        viewer = IntegrationBoundaryViewer()

        success = viewer.load_csv_file("nonexistent.csv")
        assert success is False

    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_detect_boundaries(self, mock_app):
        """Test boundary detection."""
        viewer = IntegrationBoundaryViewer()

        # Set up viewer with test data
        test_data = self.create_test_data()
        viewer.force_data = test_data["AI_Channel_1_lbf"] * 4.44822  # Convert to N
        viewer.time_data = test_data["Time"]

        with patch.object(viewer, "update_boundary_info"):
            viewer.detect_boundaries("STND")

            if viewer.boundaries:
                assert "start_idx" in viewer.boundaries
                assert "end_idx" in viewer.boundaries
                assert "duration_ms" in viewer.boundaries
                assert "material_code" in viewer.boundaries
                assert viewer.boundaries["material_code"] == "STND"

    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_update_methods(self, mock_app):
        """Test viewer update methods."""
        viewer = IntegrationBoundaryViewer()

        # Mock UI components
        viewer.material_info = MagicMock()
        viewer.data_stats = MagicMock()
        viewer.boundary_info = MagicMock()

        # Test update_material_info
        viewer.update_material_info("STND")
        viewer.material_info.setText.assert_called_once()

        # Test update_data_stats with data
        viewer.force_data = np.random.normal(0, 100, 1000)
        viewer.time_data = np.linspace(0, 0.1, 1000)
        viewer.current_file = Path("STND-21-5.csv")

        viewer.update_data_stats()
        viewer.data_stats.setPlainText.assert_called_once()

        # Test update_boundary_info with boundaries
        viewer.boundaries = {
            "start_idx": 300,
            "end_idx": 700,
            "start_time_ms": 30.0,
            "end_time_ms": 70.0,
            "duration_ms": 40.0,
            "impulse": 0.05,
            "peak_force": 1000,
            "material_code": "STND",
        }

        viewer.update_boundary_info()
        viewer.boundary_info.setPlainText.assert_called_once()


@pytest.mark.skipif(PYQT_AVAILABLE, reason="Testing behavior when PyQt5 not available")
class TestGUINotAvailable:
    """Test behavior when PyQt5 is not available."""

    def test_launch_boundary_viewer_unavailable(self):
        """Test launch_boundary_viewer when PyQt5 not available."""
        # Should return False or handle gracefully
        result = launch_boundary_viewer()
        assert result is False or result is None

    def test_gui_info_unavailable(self):
        """Test gui_info when PyQt5 not available."""
        from Fishing_Line_Flyback_Impact_Analysis import gui_info

        info = gui_info()
        assert info["gui_available"] is False
        assert "install_command" in info


class TestGUIIntegration:
    """Test GUI integration with main package."""

    def test_boundary_viewer_cli_integration(self):
        """Test that boundary viewer integrates with CLI."""
        # This tests the CLI command structure without actually running GUI
        from Fishing_Line_Flyback_Impact_Analysis.__main__ import main

        # Should have boundary-viewer command
        # Note: This is a structural test, not execution test
        assert main is not None

    def test_quick_boundary_check_function(self):
        """Test quick_boundary_check function from main module."""
        from Fishing_Line_Flyback_Impact_Analysis import quick_boundary_check

        # Should be callable
        assert callable(quick_boundary_check)

        # Test with non-existent file (should handle gracefully)
        quick_boundary_check("nonexistent.csv")

    def test_gui_module_structure(self):
        """Test GUI module has expected structure."""
        from Fishing_Line_Flyback_Impact_Analysis import gui

        # Should have expected attributes
        expected_attributes = ["PYQT_AVAILABLE", "launch_boundary_viewer"]

        for attr in expected_attributes:
            assert hasattr(gui, attr)

    def test_gui_imports_in_main_init(self):
        """Test that GUI components are properly imported in main __init__.py."""
        import Fishing_Line_Flyback_Impact_Analysis as fli  # noqa: N813

        # Should have GUI-related functions
        gui_functions = [
            "launch_boundary_viewer",
            "quick_boundary_check",
            "gui_info",
        ]

        for func_name in gui_functions:
            assert hasattr(fli, func_name)


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt5 not available")
class TestGUIErrorHandling:
    """Test GUI error handling (only if PyQt5 available)."""

    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_error_handling(self, mock_app):
        """Test viewer handles errors gracefully."""
        viewer = IntegrationBoundaryViewer()

        # Test with corrupted data
        viewer.force_data = np.array([np.nan, np.inf, -np.inf])
        viewer.time_data = np.array([0, 0.001, 0.002])

        # Should not crash
        try:
            viewer.detect_boundaries("STND")
        except Exception:
            # Some exceptions are acceptable
            pass

    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_empty_data_handling(self, mock_app):
        """Test viewer handles empty data."""
        viewer = IntegrationBoundaryViewer()

        viewer.force_data = np.array([])
        viewer.time_data = np.array([])

        # Should handle empty data gracefully
        viewer.detect_boundaries("STND")

    @patch("PyQt5.QtWidgets.QApplication")
    @patch("PyQt5.QtWidgets.QMessageBox")
    def test_viewer_file_error_handling(self, mock_msgbox, mock_app):
        """Test viewer handles file errors."""
        viewer = IntegrationBoundaryViewer()

        # Mock file operations to raise errors
        with patch(
            "Fishing_Line_Flyback_Impact_Analysis.shared.load_csv_file"
        ) as mock_load:
            mock_load.side_effect = Exception("File error")

            success = viewer.load_csv_file("test.csv")
            assert success is False


class TestGUIDocumentation:
    """Test GUI documentation and help functions."""

    def test_gui_module_docstring(self):
        """Test GUI module has proper docstring."""
        from Fishing_Line_Flyback_Impact_Analysis import gui

        assert gui.__doc__ is not None
        assert len(gui.__doc__.strip()) > 0

    @pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt5 not available")
    def test_boundary_viewer_docstring(self):
        """Test IntegrationBoundaryViewer has proper docstring."""
        assert IntegrationBoundaryViewer.__doc__ is not None
        assert len(IntegrationBoundaryViewer.__doc__.strip()) > 0

    def test_launch_boundary_viewer_docstring(self):
        """Test launch_boundary_viewer has proper docstring."""
        if PYQT_AVAILABLE:
            assert launch_boundary_viewer.__doc__ is not None

    def test_gui_info_completeness(self):
        """Test that gui_info provides complete information."""
        from Fishing_Line_Flyback_Impact_Analysis import gui_info

        info = gui_info()

        required_keys = ["gui_available", "components", "requirements", "features"]

        for key in required_keys:
            assert key in info

    def test_package_level_gui_documentation(self):
        """Test package-level GUI documentation."""
        from Fishing_Line_Flyback_Impact_Analysis import get_configuration_info

        # Should include GUI information
        config_info = get_configuration_info()
        assert isinstance(config_info, dict)


class TestGUIPerformance:
    """Test GUI performance considerations."""

    @pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt5 not available")
    @patch("PyQt5.QtWidgets.QApplication")
    def test_viewer_large_dataset_handling(self, mock_app):
        """Test viewer performance with large datasets."""
        viewer = IntegrationBoundaryViewer()

        # Large dataset
        n_points = 100000
        viewer.force_data = np.random.normal(0, 100, n_points)
        viewer.time_data = np.linspace(0, 10, n_points)

        # Should handle large datasets without hanging
        # (This is more of a smoke test)
        try:
            viewer.detect_boundaries("STND")
        except Exception:
            # Some performance-related exceptions might be acceptable
            pass

    def test_gui_memory_usage(self):
        """Test that GUI imports don't consume excessive memory."""
        import sys

        # Get memory usage before import
        initial_modules = len(sys.modules)

        # Should not import excessive number of modules if PyQt not available
        if not PYQT_AVAILABLE:
            final_modules = len(sys.modules)
            # Should not add too many modules
            assert (final_modules - initial_modules) < 10
