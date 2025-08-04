"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_force_data():
    """Generate sample force and time data for testing."""
    n_points = 1000
    time = np.linspace(0, 0.1, n_points)
    force = np.random.normal(0, 10, n_points)

    # Add clear impact signal
    impact_start = 300
    impact_end = 700
    force[impact_start:impact_end] = 1000 * np.sin(
        np.linspace(0, np.pi, impact_end - impact_start)
    )

    return force, time


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data structure."""
    n_points = 1000
    time = np.linspace(0, 0.1, n_points)
    force1 = np.random.normal(0, 10, n_points)
    force2 = np.random.normal(0, 5, n_points)

    # Add impact signal
    force1[400:600] = 1000 * np.sin(np.linspace(0, np.pi, 200))
    force2[400:600] = 300 * np.sin(np.linspace(0, np.pi, 200))

    return {"AI_Channel_1_lbf": force1, "AI_Channel_2_lbf": force2, "Time": time}


@pytest.fixture
def create_test_csv():
    """Factory fixture to create test CSV files."""
    created_files = []

    def _create_csv(data_dict, filename="test.csv", temp_dir=None):
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir())
        else:
            temp_dir = Path(temp_dir)

        file_path = temp_dir / filename
        df = pd.DataFrame(data_dict)
        df.to_csv(file_path, index=False)
        created_files.append(file_path)
        return file_path

    yield _create_csv

    # Cleanup
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def sample_analysis_result():
    """Generate sample analysis result for testing."""
    return {
        "filename": "STND-21-5.csv",
        "material_type": "STND",
        "sample_number": "21-5",
        "total_impulse": 0.05234,
        "total_abs_impulse": 0.05234,
        "impact_impulse": 0.04987,
        "impact_abs_impulse": 0.04987,
        "peak_force": 1847.3,
        "peak_force_positive": 1847.3,
        "peak_force_negative": -234.1,
        "rms_force": 387.2,
        "impact_duration": 0.0234,
        "impact_start_time": 0.0123,
        "impact_end_time": 0.0357,
        "total_duration": 0.1,
        "equivalent_velocity": 234.7,
        "equivalent_kinetic_energy": 1.847,
        "mass_kg": 0.0717,
        "impact_start_idx": 1230,
        "impact_end_idx": 3570,
        "sampling_rate_hz": 100000,
        "analysis_method": "impulse_integration",
        "material_code": "STND",
        "mass_breakdown": {
            "hardware_mass_kg": 0.045,
            "line_mass_effective_kg": 0.02716,
            "total_mass_kg": 0.0717,
            "material_code": "STND",
        },
    }


@pytest.fixture
def multiple_analysis_results():
    """Generate multiple analysis results for different materials."""
    materials = ["STND", "DF", "DS", "SL", "BR"]
    results = []

    for i, material in enumerate(materials):
        for j in range(2):  # 2 samples per material
            result = {
                "filename": f"{material}-21-{j + 1}.csv",
                "material_type": material,
                "sample_number": f"21-{j + 1}",
                "total_impulse": 0.05 * (1 + i * 0.1) * (1 + j * 0.05),
                "total_abs_impulse": 0.05 * (1 + i * 0.1) * (1 + j * 0.05),
                "impact_impulse": 0.048 * (1 + i * 0.1) * (1 + j * 0.05),
                "impact_abs_impulse": 0.048 * (1 + i * 0.1) * (1 + j * 0.05),
                "peak_force": 1500 * (1 + i * 0.2) * (1 + j * 0.1),
                "peak_force_positive": 1500 * (1 + i * 0.2) * (1 + j * 0.1),
                "peak_force_negative": -200 * (1 + i * 0.1),
                "rms_force": 300 * (1 + i * 0.15),
                "impact_duration": 0.02 * (1 + i * 0.1),
                "impact_start_time": 0.01,
                "impact_end_time": 0.03 * (1 + i * 0.1),
                "total_duration": 0.1,
                "equivalent_velocity": 200 * (1 + i * 0.2),
                "equivalent_kinetic_energy": 1.5 * (1 + i * 0.3),
                "mass_kg": 0.07 + i * 0.005,
                "impact_start_idx": 1000,
                "impact_end_idx": 3000,
                "sampling_rate_hz": 100000,
                "analysis_method": "impulse_integration",
                "material_code": material,
                "mass_breakdown": {
                    "hardware_mass_kg": 0.045 + i * 0.005,
                    "line_mass_effective_kg": 0.025,
                    "total_mass_kg": 0.07 + i * 0.005,
                    "material_code": material,
                },
            }
            results.append(result)

    return results


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to prevent actual plotting during tests."""
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch(
        "matplotlib.pyplot.close"
    ):
        yield


@pytest.fixture
def mock_gui_unavailable():
    """Mock GUI as unavailable for testing fallback behavior."""
    with patch("Fishing_Line_Flyback_Impact_Analysis.gui.PYQT_AVAILABLE", False):
        yield


@pytest.fixture
def mock_gui_available():
    """Mock GUI as available for testing GUI functionality."""
    with patch("Fishing_Line_Flyback_Impact_Analysis.gui.PYQT_AVAILABLE", True):
        yield


@pytest.fixture(scope="session")
def realistic_test_dataset():
    """Create a realistic test dataset that persists for the session."""
    return {
        "materials": ["STND", "DF", "DS", "SL", "BR"],
        "samples_per_material": 3,
        "data_points": 10000,
        "sampling_rate": 100000,
        "duration": 0.1,
        "impact_characteristics": {
            "STND": {"magnitude": 1500, "duration_factor": 1.0},
            "DF": {"magnitude": 1800, "duration_factor": 1.2},
            "DS": {"magnitude": 2000, "duration_factor": 1.1},
            "SL": {"magnitude": 1700, "duration_factor": 1.3},
            "BR": {"magnitude": 1400, "duration_factor": 0.9},
        },
    }


@pytest.fixture
def create_realistic_force_data(realistic_test_dataset):
    """Factory to create realistic force data based on material type."""

    def _create_data(material="STND", sample_id=1, noise_level=1.0):
        dataset = realistic_test_dataset
        n_points = dataset["data_points"]

        # Base time array
        time = np.linspace(0, dataset["duration"], n_points)

        # Base noise
        force1 = np.random.normal(0, 10 * noise_level, n_points)
        force2 = np.random.normal(0, 5 * noise_level, n_points)

        # Material-specific impact characteristics
        char = dataset["impact_characteristics"].get(
            material, {"magnitude": 1500, "duration_factor": 1.0}
        )

        # Impact timing
        impact_start = int(0.3 * n_points)
        impact_duration = int(0.2 * n_points * char["duration_factor"])
        impact_end = impact_start + impact_duration

        # Impact profile (asymmetric)
        impact_indices = np.arange(impact_start, impact_end)
        peak_idx = impact_start + impact_duration // 3  # Peak early in impact

        # Rising phase
        rise_indices = impact_indices[impact_indices <= peak_idx]
        if len(rise_indices) > 0:
            rise_profile = char["magnitude"] * np.power(
                (rise_indices - impact_start) / (peak_idx - impact_start), 2
            )
            force1[rise_indices] += rise_profile

        # Falling phase
        fall_indices = impact_indices[impact_indices > peak_idx]
        if len(fall_indices) > 0:
            fall_profile = char["magnitude"] * np.exp(
                -3 * (fall_indices - peak_idx) / (impact_end - peak_idx)
            )
            force1[fall_indices] += fall_profile

        # Secondary sensor (reduced magnitude)
        force2[impact_start:impact_end] += 0.3 * force1[impact_start:impact_end]

        # Add sample-specific variation
        variation_factor = 0.8 + (sample_id % 5) * 0.1
        force1 *= variation_factor
        force2 *= variation_factor

        return {"AI_Channel_1_lbf": force1, "AI_Channel_2_lbf": force2, "Time": time}

    return _create_data


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gui: marks tests that require GUI components")
    config.addinivalue_line(
        "markers", "visualization: marks tests that create plots/visualizations"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/name."""
    for item in items:
        # Mark GUI tests
        if "test_gui" in item.nodeid or "gui" in item.name.lower():
            item.add_marker(pytest.mark.gui)

        # Mark visualization tests
        if (
            "visualization" in item.nodeid
            or "plot" in item.name.lower()
            or "show" in item.name.lower()
        ):
            item.add_marker(pytest.mark.visualization)

        # Mark integration tests
        if (
            "integration" in item.name.lower()
            or "workflow" in item.name.lower()
            or "complete" in item.name.lower()
        ):
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if (
            "large" in item.name.lower()
            or "batch" in item.name.lower()
            or "realistic" in item.name.lower()
        ):
            item.add_marker(pytest.mark.slow)


# Custom assertions for fishing line analysis
class FishingLineAssertions:
    """Custom assertion helpers for fishing line analysis testing."""

    @staticmethod
    def assert_valid_impulse_result(result):
        """Assert that an impulse analysis result is valid."""
        assert isinstance(result, dict)
        assert "error" not in result

        # Required fields
        required_fields = [
            "total_impulse",
            "total_abs_impulse",
            "peak_force",
            "impact_duration",
            "filename",
            "material_type",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Value validation
        assert isinstance(result["total_impulse"], (int, float))
        assert isinstance(result["total_abs_impulse"], (int, float))
        assert result["total_abs_impulse"] >= 0
        assert result["peak_force"] > 0
        assert result["impact_duration"] > 0

        # Material validation
        valid_materials = {"STND", "DF", "DS", "SL", "BR", "UNKNOWN"}
        assert result["material_type"] in valid_materials

    @staticmethod
    def assert_valid_force_data(force, time):
        """Assert that force and time data are valid."""
        assert isinstance(force, np.ndarray)
        assert isinstance(time, np.ndarray)
        assert len(force) == len(time)
        assert len(force) > 0
        assert np.all(np.isfinite(force))
        assert np.all(np.isfinite(time))
        assert np.all(np.diff(time) > 0)  # Time should be monotonic

    @staticmethod
    def assert_reasonable_impulse_values(result):
        """Assert that impulse values are in reasonable ranges."""
        # Typical fishing line impulse values
        abs_impulse = abs(result["total_impulse"])
        assert (
            1e-6 <= abs_impulse <= 1.0
        ), f"Impulse {abs_impulse} outside reasonable range"

        # Peak force should be reasonable
        peak_force = result["peak_force"]
        assert (
            10 <= peak_force <= 100000
        ), f"Peak force {peak_force} outside reasonable range"

        # Duration should be reasonable (microseconds to seconds)
        duration = result["impact_duration"]
        assert 1e-6 <= duration <= 1.0, f"Duration {duration} outside reasonable range"

    @staticmethod
    def assert_material_consistency(results):
        """Assert that results are consistent within material types."""
        if not results:
            return

        # Group by material
        by_material = {}
        for result in results:
            if "error" not in result:
                material = result["material_type"]
                if material not in by_material:
                    by_material[material] = []
                by_material[material].append(result)

        # Check consistency within each material
        for material, material_results in by_material.items():
            if len(material_results) < 2:
                continue

            impulses = [abs(r["total_impulse"]) for r in material_results]
            forces = [r["peak_force"] for r in material_results]

            # Should have some consistency (coefficient of variation < 2.0)
            cv_impulse = (
                np.std(impulses) / np.mean(impulses) if np.mean(impulses) > 0 else 0
            )
            cv_force = np.std(forces) / np.mean(forces) if np.mean(forces) > 0 else 0

            assert (
                cv_impulse < 2.0
            ), f"Material {material} impulse too inconsistent (CV={cv_impulse})"
            assert (
                cv_force < 2.0
            ), f"Material {material} force too inconsistent (CV={cv_force})"


@pytest.fixture
def fishing_line_assertions():
    """Provide custom assertion helpers."""
    return FishingLineAssertions()


# Error handling helpers
@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        yield


# Performance testing helpers
@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()
            return self.elapsed

        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

    return Timer()


# Memory testing helpers
@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import os

    import psutil

    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None
            self.peak_memory = None

        def start(self):
            self.initial_memory = self.process.memory_info().rss
            self.peak_memory = self.initial_memory

        def update(self):
            current = self.process.memory_info().rss
            if current > self.peak_memory:
                self.peak_memory = current

        def stop(self):
            self.update()
            return {
                "initial_mb": self.initial_memory / 1024 / 1024,
                "peak_mb": self.peak_memory / 1024 / 1024,
                "increase_mb": (self.peak_memory - self.initial_memory) / 1024 / 1024,
            }

    return MemoryMonitor()


# Skip conditions
skip_without_gui = pytest.mark.skipif(
    "not PYQT_AVAILABLE", reason="PyQt5 not available"
)

skip_without_scipy = pytest.mark.skipif(
    "not SCIPY_AVAILABLE", reason="SciPy not available"
)

skip_slow = pytest.mark.skipif(
    "config.getoption('--fast')", reason="Skipping slow tests in fast mode"
)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast", action="store_true", default=False, help="Run fast tests only"
    )
    parser.addoption(
        "--run-gui",
        action="store_true",
        default=False,
        help="Run GUI tests (requires PyQt5)",
    )
    parser.addoption(
        "--run-visualization",
        action="store_true",
        default=False,
        help="Run visualization tests (may show plots)",
    )
