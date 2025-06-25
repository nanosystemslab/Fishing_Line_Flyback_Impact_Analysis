"""Test cases for the package initialization."""


def test_package_imports():
    """Test that package imports work correctly."""
    from Fishing_Line_Flyback_Impact_Analysis import ImpactAnalyzer
    from Fishing_Line_Flyback_Impact_Analysis import ImpactVisualizer
    from Fishing_Line_Flyback_Impact_Analysis import __version__

    assert ImpactAnalyzer is not None
    assert ImpactVisualizer is not None
    assert __version__ is not None


def test_version_format():
    """Test that version follows proper format."""
    from Fishing_Line_Flyback_Impact_Analysis import __version__

    # Should be in format x.y.z
    version_parts = __version__.split(".")
    assert len(version_parts) >= 2  # At least major.minor

    # First two parts should be numeric
    assert version_parts[0].isdigit()
    assert version_parts[1].isdigit()


def test_all_exports():
    """Test that __all__ contains expected exports."""
    from Fishing_Line_Flyback_Impact_Analysis import __all__

    expected_exports = ["ImpactAnalyzer", "ImpactVisualizer", "__version__"]

    for export in expected_exports:
        assert export in __all__


def test_class_instantiation():
    """Test that classes can be instantiated."""
    from Fishing_Line_Flyback_Impact_Analysis import ImpactAnalyzer
    from Fishing_Line_Flyback_Impact_Analysis import ImpactVisualizer

    analyzer = ImpactAnalyzer()
    visualizer = ImpactVisualizer()

    assert analyzer is not None
    assert visualizer is not None


def test_module_docstring():
    """Test that package has proper docstring."""
    import Fishing_Line_Flyback_Impact_Analysis

    assert Fishing_Line_Flyback_Impact_Analysis.__doc__ is not None
    assert (
        "Fishing Line Flyback Impact Analysis"
        in Fishing_Line_Flyback_Impact_Analysis.__doc__
    )
