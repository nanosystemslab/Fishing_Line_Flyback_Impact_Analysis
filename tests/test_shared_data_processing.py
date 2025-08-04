"""Tests for shared data processing module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import (
    apply_baseline_correction,
)
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import (
    calculate_total_force,
)
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import convert_lbf_to_n
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import (
    detect_force_columns,
)
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import (
    extract_material_code,
)
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import (
    extract_sample_number,
)
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import get_system_mass
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import get_time_array
from Fishing_Line_Flyback_Impact_Analysis.shared.data_processing import load_csv_file


class TestDataProcessing:
    """Test data processing functions."""

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def test_load_csv_file_success(self):
        """Test successful CSV loading."""
        test_data = {
            "AI_Channel_1_lbf": [1.0, 2.0, 3.0],
            "AI_Channel_2_lbf": [0.5, 1.0, 1.5],
            "Time": [0.0, 0.1, 0.2],
        }
        file_path = self.create_test_csv(test_data)

        try:
            df = load_csv_file(file_path)
            assert len(df) == 3
            assert "AI_Channel_1_lbf" in df.columns
            pd.testing.assert_frame_equal(df, pd.DataFrame(test_data))
        finally:
            file_path.unlink()

    def test_load_csv_file_not_found(self):
        """Test loading non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            load_csv_file("nonexistent.csv")

    def test_load_csv_file_empty(self):
        """Test loading empty CSV file."""
        empty_file = Path(tempfile.gettempdir()) / "empty.csv"
        empty_file.touch()

        try:
            with pytest.raises(RuntimeError, match="Error reading CSV file"):
                load_csv_file(empty_file)
        finally:
            empty_file.unlink()

    def test_detect_force_columns_ai_lbf(self):
        """Test detection of AI+lbf force columns."""
        df = pd.DataFrame(
            {
                "AI_Channel_1_lbf": [1, 2, 3],
                "AI_Channel_2_lbf": [4, 5, 6],
                "Time": [0.1, 0.2, 0.3],
                "Other_Column": [7, 8, 9],
            }
        )

        columns = detect_force_columns(df)
        assert len(columns) == 2
        assert "AI_Channel_1_lbf" in columns
        assert "AI_Channel_2_lbf" in columns
        assert "Time" not in columns
        assert "Other_Column" not in columns

    def test_detect_force_columns_fallback(self):
        """Test fallback force column detection."""
        df = pd.DataFrame(
            {
                "Force1": [1, 2, 3],
                "Force2": [4, 5, 6],
                "time_ms": [0.1, 0.2, 0.3],
                "text_column": ["a", "b", "c"],
            }
        )

        columns = detect_force_columns(df)
        assert len(columns) == 2
        assert "Force1" in columns
        assert "Force2" in columns
        assert "time_ms" not in columns
        assert "text_column" not in columns

    def test_convert_lbf_to_n(self):
        """Test lbf to Newton conversion."""
        force_lbf = np.array([1.0, 2.0, 0.0, -1.0])
        force_n = convert_lbf_to_n(force_lbf)

        expected = np.array([4.44822, 8.89644, 0.0, -4.44822])
        np.testing.assert_allclose(force_n, expected, rtol=1e-5)

    def test_apply_baseline_correction_short_data(self):
        """Test baseline correction with short data."""
        force = np.array([1, 2, 3])
        corrected = apply_baseline_correction(force)
        np.testing.assert_array_equal(force, corrected)

    def test_apply_baseline_correction_normal(self):
        """Test baseline correction with normal data."""
        # Create data with DC offset
        baseline_offset = 5.0
        force = np.array([baseline_offset + x for x in range(1000)])

        corrected = apply_baseline_correction(force)

        # Should remove the baseline (first 100 points median)
        expected_baseline = np.median(force[:100])
        expected = force - expected_baseline
        np.testing.assert_allclose(corrected, expected)

    def test_calculate_total_force_success(self):
        """Test successful total force calculation."""
        df = pd.DataFrame(
            {
                "AI_Channel_1_lbf": [1.0, 2.0, 3.0],
                "AI_Channel_2_lbf": [0.5, 1.0, 1.5],
                "Time": [0.0, 0.1, 0.2],
            }
        )

        total_force, columns = calculate_total_force(df)

        assert len(columns) == 2
        assert "AI_Channel_1_lbf" in columns
        assert "AI_Channel_2_lbf" in columns

        # Should sum the two channels and convert to N
        expected_lbf = np.array([1.5, 3.0, 4.5])
        expected_n = expected_lbf * 4.44822
        np.testing.assert_allclose(total_force, expected_n, rtol=1e-5)

    def test_calculate_total_force_no_columns(self):
        """Test total force calculation with no force columns."""
        df = pd.DataFrame({"Time": [0.0, 0.1, 0.2], "Text": ["a", "b", "c"]})

        with pytest.raises(ValueError, match="No force columns detected"):
            calculate_total_force(df)

    def test_calculate_total_force_with_nan(self):
        """Test total force calculation with NaN values."""
        df = pd.DataFrame(
            {
                "AI_Channel_1_lbf": [1.0, np.nan, 3.0],
                "AI_Channel_2_lbf": [0.5, 1.0, np.nan],
            }
        )

        total_force, columns = calculate_total_force(df)

        # NaN should be converted to 0
        assert not np.any(np.isnan(total_force))
        expected_lbf = np.array([1.5, 1.0, 3.0])  # NaN becomes 0 in sum
        expected_n = expected_lbf * 4.44822
        np.testing.assert_allclose(total_force, expected_n, rtol=1e-5)

    def test_get_time_array_with_time_column(self):
        """Test time array extraction with time column."""
        df = pd.DataFrame({"Time": [1.0, 1.1, 1.2, 1.3], "Force": [10, 20, 30, 40]})

        time_array, sampling_rate = get_time_array(df, 4)

        # Should normalize to start at zero
        expected_time = np.array([0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(time_array, expected_time, rtol=1e-5)

        # Should calculate sampling rate
        assert abs(sampling_rate - 10.0) < 0.1  # 10 Hz

    def test_get_time_array_no_time_column(self):
        """Test time array generation without time column."""
        df = pd.DataFrame({"Force": [10, 20, 30, 40]})

        sampling_rate = 1000.0
        time_array, actual_rate = get_time_array(df, 4, sampling_rate)

        expected_time = np.array([0.0, 0.001, 0.002, 0.003])
        np.testing.assert_allclose(time_array, expected_time, rtol=1e-5)
        assert actual_rate == sampling_rate

    def test_get_time_array_empty_time(self):
        """Test time array with empty time column."""
        df = pd.DataFrame({"Time": [], "Force": []})

        time_array, sampling_rate = get_time_array(df, 0, 100.0)
        assert len(time_array) == 0

    def test_extract_material_code_valid(self):
        """Test material code extraction from valid filenames."""
        test_cases = [
            ("STND-21-5.csv", "STND"),
            ("DF-21-10.csv", "DF"),
            ("DS-21-1.csv", "DS"),
            ("SL-21-3.csv", "SL"),
            ("BR-21-7.csv", "BR"),
            ("TEST-99-1.csv", "TEST"),
        ]

        for filename, expected in test_cases:
            assert extract_material_code(filename) == expected

    def test_extract_material_code_invalid(self):
        """Test material code extraction from invalid filenames."""
        test_cases = [
            "no_dashes.csv",
            "",
            "single.dash",
            None,
        ]

        for filename in test_cases:
            result = extract_material_code(filename)
            assert result == "UNKNOWN"

    def test_extract_sample_number_valid(self):
        """Test sample number extraction from valid filenames."""
        test_cases = [
            ("STND-21-5.csv", "21-5"),
            ("DF-21-10.csv", "21-10"),
            ("DS-21-1.csv", "21-1"),
            ("TEST-99-1-extra.csv", "99-1-extra"),
        ]

        for filename, expected in test_cases:
            assert extract_sample_number(filename) == expected

    def test_extract_sample_number_invalid(self):
        """Test sample number extraction from invalid filenames."""
        test_cases = [
            "no_dashes.csv",
            "",
            "single",
            None,
        ]

        for filename in test_cases:
            result = extract_sample_number(filename)
            assert result == "UNKNOWN"

    def test_get_system_mass_with_line_mass(self):
        """Test system mass calculation with line mass."""
        result = get_system_mass("STND", include_line_mass=True, line_mass_fraction=0.7)

        assert result["material_code"] == "STND"
        assert result["hardware_mass_kg"] == 0.045
        assert abs(result["line_mass_effective_kg"] - (0.0388 * 0.7)) < 1e-6
        assert result["line_mass_fraction"] == 0.7
        assert abs(result["total_mass_kg"] - (0.045 + 0.0388 * 0.7)) < 1e-6

    def test_get_system_mass_without_line_mass(self):
        """Test system mass calculation without line mass."""
        result = get_system_mass("DS", include_line_mass=False)

        assert result["material_code"] == "DS"
        assert result["hardware_mass_kg"] == 0.072
        assert result["line_mass_effective_kg"] == 0.0
        assert result["line_mass_fraction"] == 0.0
        assert result["total_mass_kg"] == 0.072

    def test_get_system_mass_unknown_material(self):
        """Test system mass calculation with unknown material."""
        result = get_system_mass("UNKNOWN")

        # Should default to STND configuration
        assert result["material_code"] == "UNKNOWN"
        assert result["hardware_mass_kg"] == 0.045  # STND default

    def test_get_system_mass_all_materials(self):
        """Test system mass calculation for all known materials."""
        materials = ["STND", "DF", "DS", "SL", "BR"]

        for material in materials:
            result = get_system_mass(material, include_line_mass=True)

            assert result["material_code"] == material
            assert result["hardware_mass_kg"] > 0
            assert result["line_mass_effective_kg"] > 0
            assert result["total_mass_kg"] > result["hardware_mass_kg"]
            assert 0 < result["line_mass_fraction"] <= 1
