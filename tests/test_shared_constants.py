"""Tests for shared constants module."""

from Fishing_Line_Flyback_Impact_Analysis.shared.constants import CONFIG_WEIGHTS
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import DEFAULT_SAMPLING_RATE
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import GRAMS_TO_KG
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import (
    IMPACT_THRESHOLD_FACTOR,
)
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import INCHES_TO_M
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import LBF_TO_N
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import LINE_MASS_FRACTION
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import MATERIAL_NAMES
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import (
    MEASURED_LINE_LENGTH_INCHES,
)
from Fishing_Line_Flyback_Impact_Analysis.shared.constants import (
    MEASURED_LINE_MASS_GRAMS,
)


class TestConstants:
    """Test constant values and types."""

    def test_config_weights_exist(self):
        """Test that all expected configuration weights exist."""
        expected_configs = {"STND", "DF", "DS", "SL", "BR"}
        assert set(CONFIG_WEIGHTS.keys()) == expected_configs

    def test_config_weights_values(self):
        """Test that configuration weights are reasonable."""
        for _config, weight in CONFIG_WEIGHTS.items():
            assert isinstance(weight, float)
            assert 0.040 <= weight <= 0.080  # 40-80g range

    def test_config_weights_specific_values(self):
        """Test specific configuration weight values."""
        assert CONFIG_WEIGHTS["STND"] == 0.045
        assert CONFIG_WEIGHTS["DF"] == 0.060
        assert CONFIG_WEIGHTS["DS"] == 0.072
        assert CONFIG_WEIGHTS["SL"] == 0.069
        assert CONFIG_WEIGHTS["BR"] == 0.045

    def test_line_mass_fraction(self):
        """Test line mass fraction is reasonable."""
        assert isinstance(LINE_MASS_FRACTION, float)
        assert 0.0 < LINE_MASS_FRACTION <= 1.0
        assert LINE_MASS_FRACTION == 0.70

    def test_measured_line_parameters(self):
        """Test measured line parameters."""
        assert isinstance(MEASURED_LINE_LENGTH_INCHES, float)
        assert isinstance(MEASURED_LINE_MASS_GRAMS, float)
        assert MEASURED_LINE_LENGTH_INCHES > 0
        assert MEASURED_LINE_MASS_GRAMS > 0
        assert MEASURED_LINE_LENGTH_INCHES == 5.5
        assert MEASURED_LINE_MASS_GRAMS == 0.542

    def test_material_names(self):
        """Test material name mappings."""
        expected_materials = {"STND", "DF", "DS", "SL", "BR"}
        assert set(MATERIAL_NAMES.keys()) == expected_materials

        # Test specific mappings
        assert MATERIAL_NAMES["STND"] == "Standard"
        assert MATERIAL_NAMES["DF"] == "Dual Fixed"
        assert MATERIAL_NAMES["DS"] == "Dual Sliding"
        assert MATERIAL_NAMES["SL"] == "Sliding"
        assert MATERIAL_NAMES["BR"] == "Breakaway"

        # All values should be strings
        for name in MATERIAL_NAMES.values():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_analysis_parameters(self):
        """Test analysis parameter constants."""
        assert isinstance(DEFAULT_SAMPLING_RATE, float)
        assert DEFAULT_SAMPLING_RATE > 0
        assert DEFAULT_SAMPLING_RATE == 100000.0

        assert isinstance(IMPACT_THRESHOLD_FACTOR, float)
        assert 0.0 < IMPACT_THRESHOLD_FACTOR < 1.0
        assert IMPACT_THRESHOLD_FACTOR == 0.02

    def test_conversion_factors(self):
        """Test unit conversion factors."""
        assert isinstance(LBF_TO_N, float)
        assert LBF_TO_N > 0
        assert abs(LBF_TO_N - 4.44822) < 1e-5

        assert isinstance(INCHES_TO_M, float)
        assert INCHES_TO_M > 0
        assert abs(INCHES_TO_M - 0.0254) < 1e-6

        assert isinstance(GRAMS_TO_KG, float)
        assert GRAMS_TO_KG > 0
        assert GRAMS_TO_KG == 0.001

    def test_all_constants_immutable(self):
        """Test that we can access all constants without error."""
        # This test ensures all constants are properly defined
        constants = [
            CONFIG_WEIGHTS,
            LINE_MASS_FRACTION,
            MEASURED_LINE_LENGTH_INCHES,
            MEASURED_LINE_MASS_GRAMS,
            MATERIAL_NAMES,
            DEFAULT_SAMPLING_RATE,
            IMPACT_THRESHOLD_FACTOR,
            LBF_TO_N,
            INCHES_TO_M,
            GRAMS_TO_KG,
        ]

        for constant in constants:
            assert constant is not None
