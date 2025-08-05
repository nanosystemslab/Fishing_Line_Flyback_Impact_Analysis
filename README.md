# Fishing Line Flyback Impact Analysis

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)][license]

**A comprehensive Python package for analyzing fishing line flyback impact data using impulse-based analysis (∫F(t)dt) to measure momentum transfer effectiveness.**

This package provides streamlined tools for processing multi-sensor force data and quantifying fishing line impact properties through direct momentum transfer measurement, which is more relevant to fishing line performance than energy-based methods.

## Overview

When fishing lines snap under tension, the rapid flyback creates significant impact forces that can cause injury. This package analyzes sensor data from controlled flyback tests to quantify momentum transfer and compare the effectiveness of different weight configurations in reducing impact forces.

### Key Scientific Approach

**Primary Method: Impulse Analysis (v1.0)**

- **Direct momentum transfer measurement** via ∫F(t)dt integration
- **What the fish/lure actually experiences** during impact
- **Visual verification** of integration boundaries for quality assurance

## Features

### 🎯 **Primary Impulse Analysis**

- **Direct momentum transfer measurement** using ∫F(t)dt
- **Configuration-specific masses** (STND=45g, DF=60g, DS=72g, SL=69g, BR=45g)

### 🔍 **Data Processing & Verification**

- **Shared data processing components** for consistent analysis
- **Multi-sensor CSV support** from Dewesoft measurements
- **Automatic force summation** with baseline correction
- **Smart peak detection** for efficient large dataset handling
- **Lightweight boundary viewer** for visual verification of integration regions

### 📊 **Visualization & Analysis**

- **Interactive force curve exploration** with zoom and pan
- **Boundary validation plots** showing integration regions
- **Material comparison box plots** with statistical significance
- **Real-time integration boundary verification** for quality control
- **Cumulative impulse visualization** for understanding momentum buildup

### ⚡ **Performance & Usability**

- **Fast boundary viewer GUI** for rapid visual verification
- **CLI with progressive complexity** from simple to research-grade analysis
- **Batch processing** for large datasets
- **Export capabilities** for plots, data, and analysis results

## Requirements

- **Python**: ≥3.11, <4.0
- **Core Dependencies**:
  - `pandas` ≥2.3.0 - Data manipulation and analysis
  - `numpy` ≥2.3.1 - Numerical computing
  - `matplotlib` ≥3.10.3 - Plotting and visualization
  - `seaborn` ≥0.13.2 - Statistical visualization
- **Optional GUI Dependencies**:
  - `PyQt5` - Boundary viewer and interactive dashboard (install with: `poetry add PyQt5 pyqtgraph`)

## Installation

### From Source (Recommended)

Clone the repository and install with Poetry:

```console
$ git clone https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis.git
$ cd Fishing_Line_Flyback_Impact_Analysis
$ poetry install
```

### With GUI Support

For the interactive boundary viewer and dashboard:

```console
$ poetry install
$ poetry add PyQt5 pyqtgraph
```

### Development Installation

For development with all dev dependencies:

```console
$ poetry install --with dev
```

## Quick Start

### 1. Primary Impulse Analysis

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv
```

**Output:**

```
🎯 IMPULSE-BASED FISHING LINE ANALYSIS v3.0
============================================================
📁 Data directory: data/csv
📊 Output directory: impulse_analysis
🧮 Method: Total momentum transfer via ∫ F(t) dt

⚖️  CONFIGURATION REFERENCE:
   STND: Standard      ( 45g + 27g =  72g total)
   DF: Dual Fixed      ( 60g + 27g =  87g total)
   DS: Dual Sliding    ( 72g + 27g =  99g total)
   SL: Sliding         ( 69g + 27g =  96g total)
   BR: Breakaway       ( 45g + 27g =  72g total)

✅ Successfully analyzed: 50/50 files
📁 All results saved to: impulse_analysis
```

### 2. Visual Boundary Verification

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis boundary-viewer
```

**Features:**

- 🔍 **Fast loading** for quick visual verification
- 📊 **Interactive plotting** with zoom and pan controls
- 🎯 **Integration boundaries** highlighted with start/end markers
- ⚙️ **Material auto-detection** from filename
- 📈 **Real-time statistics** for integration region

### 3. Single File Analysis with Validation

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-single data/csv/STND-21-5.csv --show-plot
```

### 4. Quick Dataset Overview

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis quick-check data/csv -n 5
```

**Output:**

```
⚡ QUICK IMPULSE CHECK (5 files)
-----------------------------------------------------------------
File               Mat  Total Impulse    Direction  Peak Force
-----------------------------------------------------------------
STND-21-1.csv      STND +0.002874 N*s   Forward →   1247 N
DF-21-1.csv        DF   +0.001256 N*s   Forward →    892 N
BR-21-1.csv        BR   +0.000123 N*s   Forward →    234 N
```

## Data Format

The package expects CSV files from Dewesoft with force sensor columns:

- **Force columns**: `AI 1/AI 1 (lbf)` through `AI 4/AI 4 (lbf)` (automatically detected)
- **Time column**: `Time (s)` or automatically generated from sampling rate
- **File naming**: `CONFIG-DIAMETER-FILENUM.csv` (e.g., `STND-21-5.csv`)

**Supported Configurations:**

- `STND`: Standard (45g hardware)
- `DF`: Dual Fixed (60g hardware)
- `DS`: Dual Sliding (72g hardware)
- `SL`: Sliding (69g hardware)
- `BR`: Breakaway (45g hardware)

## Command Reference

### Primary Impulse Analysis Commands

#### `analyze-impulse` - Main Analysis Command

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse [DATA_DIR]

Arguments:
  DATA_DIR              Directory containing CSV files

Options:
  --output-dir, -o      Output directory [default: impulse_analysis]
  --create-plots        Create summary box plots [default: True]
```

#### `boundary-viewer` - Visual Verification GUI

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis boundary-viewer

Options:
  --file, -f           CSV file to load on startup

Features:
  🔍 Fast loading and plotting
  📊 Interactive zoom and pan
  🎯 Integration boundaries highlighted
  ⚙️ Material auto-detection
  📈 Real-time statistics display
```

**Purpose:** Visual verification that integration regions capture the main impact event appropriately for human inspection before running batch analysis.

#### `analyze-single` - Single File Analysis

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-single [FILE_PATH]

Options:
  --material, -m       Material configuration code
  --show-plot          Show boundary validation plot
  --debug              Show detailed debug information
```

### Utility Commands

#### `quick-check` - Fast Dataset Overview

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis quick-check [DATA_DIR]

Options:
  --count, -n          Number of files to check [default: 5]
```

#### `plot-file` - Force Curve Visualization

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis plot-file [FILE_PATH]

Options:
  --interactive        Use interactive plotting [default: True]
  --style              Interaction style: explorer, simple [default: explorer]
```

#### `interactive-plot` - Advanced Force Exploration

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis interactive-plot [FILE_PATH]

Options:
  --show-analysis      Show analysis boundaries overlay
  --style              Interaction style [default: explorer]
```

### Legacy Analysis Access

#### `legacy ke-analysis` - Kinetic Energy Method

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis legacy ke-analysis [DATA_DIR]

Options:
  --output-dir, -o     Output directory [default: ke_analysis]
```

#### `legacy window-tool` - Interactive Windowing

```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis legacy window-tool [FILE_PATH]
```

## Output Files

### Impulse Analysis Results

#### Generated Files

- **`impulse_analysis_results.csv`** - Detailed analysis results for all files
- **`impulse_analysis_full_results.json`** - Complete results including errors
- **`impulse_boxplots_SI.png/.svg`** - Publication-quality box plots (SI units)
- **`impulse_boxplots_mixed.png/.svg`** - Mixed units (N\*s, kN force)

#### Result Columns

- `total_impulse` - Total momentum transfer (N\*s)
- `total_abs_impulse` - Absolute impulse magnitude (N\*s)
- `impact_impulse` - Impact region impulse (N\*s)
- `peak_force` - Maximum force (N)
- `impact_duration` - Impact duration (s)
- `equivalent_velocity` - Equivalent velocity (m/s)
- `equivalent_kinetic_energy` - Equivalent KE (J)

### Interactive Exports

- **Plot exports**: PNG, SVG formats for publication
- **Data exports**: Processed CSV with time, force, cumulative impulse
- **Analysis exports**: JSON and text formats for results

## Scientific Background

### Impulse vs. Kinetic Energy Analysis

**Why Impulse Analysis is Primary:**

1. **Direct measurement** of what the fish/lure experiences
2. **No assumptions** about energy conversion efficiency
3. **Complete force curve integration** captures all momentum transfer
4. **More relevant** to fishing line performance evaluation
5. **Simpler and more robust** than peak-focused methods

**When to use Kinetic Energy Analysis (Legacy):**

- Projectile energy characterization
- Comparison with ballistic calculations
- Research validation and method comparison
- Initial velocity estimation

### Mass Calculations

**Hardware Masses (Measured):**

- Standard: 45g, Dual Fixed: 60g, Dual Sliding: 72g
- Sliding: 69g, Breakaway: 45g

**Line Mass:**

- Measured: 5.5" section = 0.542g
- Total line: 38.8g (scaled from measurement)
- Effective: 27g (70% of total, literature validated)

**Total System Mass:** Hardware + Effective Line Mass

## Example Workflows

### Quality Control with Visual Verification

```bash
# 1. Quick dataset overview
poetry run python -m Fishing_Line_Flyback_Impact_Analysis quick-check data/csv

# 2. Visual verification of integration boundaries
poetry run python -m Fishing_Line_Flyback_Impact_Analysis boundary-viewer

# 3. Single file verification with validation plot
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-single problematic_file.csv --show-plot

# 4. Comprehensive analysis after verification
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv
```

### Complete Research Analysis

```bash
# 1. Visual boundary verification for representative files
poetry run python -m Fishing_Line_Flyback_Impact_Analysis boundary-viewer --file data/csv/STND-21-5.csv

# 2. Comprehensive impulse analysis
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv

# 3. Method comparison (impulse vs legacy KE)
poetry run python -m Fishing_Line_Flyback_Impact_Analysis legacy ke-analysis data/csv

# 4. Statistical validation
poetry run python -m Fishing_Line_Flyback_Impact_Analysis quick-check data/csv -n 20
```

### Publication Plot Analysis

```bash
# Generate publication plots and statistics
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv --create-plots

# Create boundary validation plots for methods paper
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-single data/csv/STND-21-5.csv --show-plot

# Interactive force exploration for supplementary materials
poetry run python -m Fishing_Line_Flyback_Impact_Analysis interactive-plot data/csv/STND-21-5.csv --show-analysis
```

## Architecture Overview

**Impulse-Focused Design with Visual Verification:**

```
Fishing_Line_Flyback_Impact_Analysis/
├── impulse_analysis.py          # Primary analysis engine
├── shared/                      # Common components
│   ├── constants.py            # Configuration masses, parameters
│   └── data_processing.py      # CSV loading, force calculation
├── visualization.py            # Impulse-focused plotting
├── gui/                        # Lightweight boundary viewer
│   └── boundary_viewer.py     # Visual verification interface
├── legacy/                     # Previous implementations
│   ├── kinetic_energy_analysis.py
│   ├── windowing.py
│   └── legacy_visualization.py
└── __main__.py                 # Comprehensive CLI
```

## Research Applications

This package supports research in:

- **Fishing safety**: Quantifying flyback momentum transfer risks
- **Equipment design**: Comparing weight configuration effectiveness
- **Impact biomechanics**: Understanding momentum transfer to human body
- **Materials testing**: Line failure and momentum dissipation analysis
- **Method validation**: Comparing impulse vs. energy-based approaches
- **Quality assurance**: Visual verification of analysis boundaries

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_Fishing Line Flyback Impact Analysis_ is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{flyback_impact_analysis_2025,
    title = {Fishing Line Flyback Impact Analysis: Impulse-Based Momentum Transfer Analysis},
    author = {Nanosystems Lab},
    year = {2025},
    url = {https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis},
    version = {3.0.0},
    note = {Impulse-focused analysis package for fishing line flyback impact measurement}
}
```

## Version History

- **v1.0.0** - Impulse-focused architecture, visual boundary verification, lightweight GUI
- **v0.2x** - Kinetic energy analysis (moved to legacy)
- **v0.1x** - Initial implementation (moved to legacy)

## Credits

This project was generated from [@nanosystemslab]'s [Nanosystems Lab Python Cookiecutter] template.

---

[tests]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[@nanosystemslab]: https://github.com/nanosystemslab
[nanosystems lab python cookiecutter]: https://github.com/nanosystemslab/cookiecutter-nanosystemslab
[file an issue]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/issues
[license]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/blob/main/LICENSE
[contributor guide]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/blob/main/CONTRIBUTING.md
