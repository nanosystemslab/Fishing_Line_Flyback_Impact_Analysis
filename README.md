# Fishing Line Flyback Impact Analysis

[![PyPI](https://img.shields.io/pypi/v/Fishing_Line_Flyback_Impact_Analysis.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/Fishing_Line_Flyback_Impact_Analysis.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/Fishing_Line_Flyback_Impact_Analysis)][pypi status]
[![License](https://img.shields.io/pypi/l/Fishing_Line_Flyback_Impact_Analysis)][license]
[![Read the documentation at https://Fishing_Line_Flyback_Impact_Analysis.readthedocs.io/](https://img.shields.io/readthedocs/Fishing_Line_Flyback_Impact_Analysis/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

**A comprehensive Python package for analyzing fishing line flyback impact data from Dewesoft sensor measurements.**

This package provides tools for processing multi-sensor force data, calculating impact properties (impulse, energy, velocity), and generating publication-ready statistical comparisons across different fishing gear configurations.

## Overview

When fishing lines snap under tension, the rapid flyback creates significant impact forces that can cause injury. This package analyzes sensor data from controlled flyback tests to quantify impact properties and compare the effectiveness of different weight configurations in reducing impact energy.

### Key Capabilities

- **Multi-sensor data processing** from Dewesoft CSV files
- **Automated peak detection** and signal trimming around impact events  
- **Impact property calculations** including impulse, force, velocity, and kinetic energy
- **Statistical analysis** across different gear configurations
- **Publication-ready visualizations** including box plots, violin plots, and time series
- **Summary tables** with automatic unit conversions and percentage comparisons

## Features

### 🔍 **Data Analysis**
- Load and process Dewesoft CSV files with multiple force sensors
- Automatic peak detection and data trimming around impact events
- Calculate impact properties: maximum force, impulse, velocity, kinetic energy
- Handle different test configurations (Standard, Dual Fixed, Dual Sliding, Sliding, Breakaway)

### 📊 **Visualization**
- Time series plots of sensor data with property annotations
- Statistical comparison plots (box plots, violin plots)
- Multi-sensor overlay plots
- Dual-parameter comparison plots with publication-quality formatting

### 📈 **Statistical Analysis**
- Automated summary statistics by configuration type
- Percentage comparisons relative to standard configuration
- Impact energy calculations assuming inelastic collision
- Human-readable summary tables printed to terminal

### ⚡ **Performance**
- Automatic H5 file caching for faster reprocessing
- Batch processing capabilities for large datasets
- Configurable output directories and file formats

### 🖥️ **Command-Line Interface**
- `analyze` - Process individual or multiple sensor files
- `postprocess` - Generate statistical plots from results
- `batch` - Process entire directories of data files
- `table` - Generate formatted summary tables
- `visualize` - Create plots from output data

## Requirements

- **Python**: ≥3.11, <4.0
- **Core Dependencies**:
  - `pandas` ≥2.3.0 - Data manipulation and analysis
  - `numpy` ≥2.3.1 - Numerical computing
  - `scipy` ≥1.16.0 - Signal processing and integration
  - `matplotlib` ≥3.10.3 - Plotting and visualization
  - `seaborn` ≥0.13.2 - Statistical visualization

## Installation

You can install *Fishing Line Flyback Impact Analysis* via [pip] from [PyPI]:

```console
$ pip install Fishing_Line_Flyback_Impact_Analysis
```

Or for development installation:

```console
$ git clone https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis.git
$ cd Fishing_Line_Flyback_Impact_Analysis
$ poetry install
```

## Quick Start

### 1. Analyze Single File
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze -i data/STND-21-1.csv
```
**Output:** `STND-21-1.csv,J=2.450,F=1250.32`

### 2. Compare Multiple Configurations
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze -i data/STND-21-*.csv data/DF-21-*.csv data/BR-21-*.csv
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis table -i out/run_results.txt
```

**Terminal Output:**
```
====================================================================================================
FLYBACK IMPACT ANALYSIS - SUMMARY TABLE
====================================================================================================
Configuration    Runs  Impulse      STD      % vs     Max Force    STD      Impact Energy  % vs    
                       [kN·s]               STND     [kN]                  [MJ]           STND    
----------------------------------------------------------------------------------------------------
Standard (SD)    10    8.03         2.71     +0%      85.83        13.17    717            +0%     
Dual Fixed (DF)  9     2.86         1.13     -64%     68.30        12.37    66             -91%    
Breakaway (BR)   10    0.69         0.30     -91%     48.94        12.33    5              -99%    
----------------------------------------------------------------------------------------------------

KEY FINDINGS:
• Breakaway reduces impact energy by 99% vs Standard
• Dual Fixed (DF) shows best compromise (91% energy reduction)
====================================================================================================
```

### 3. Generate Statistical Plots
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess -i out/run_results.txt
```

### 4. Batch Process Directory
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis batch -d data/csv --summary
```

## Data Format

The package expects CSV files from Dewesoft with the following columns:
- `Time (s)` - Time stamps
- `AI 1/AI 1 (lbf)` to `AI 4/AI 4 (lbf)` - Force sensor readings
- Optional: `Video/Camera 0 ()` (automatically removed)

**File Naming Convention**: `CONFIG-DIAMETER-FILENUM.csv`
- `CONFIG`: Test configuration (STND, DF, DS, SL, BR)
- `DIAMETER`: Line diameter (e.g., 21)  
- `FILENUM`: Test run number

**Example**: `STND-21-1.csv`, `DF-21-5.csv`, `BR-21-10.csv`

## Output Files

### Generated Plots
- **Time series**: `plot-time_vs_SUM--CONFIG-DIAM-RUN.png`
- **Box plots**: `plot-box-F.png`, `plot-box-J.png`
- **Violin plots**: `plot-violin-F.png`, `plot-violin-J.png`
- **Dual plots**: `plot-box-J-F.png`, `plot-box-J-F.svg`

### Data Files
- **Results**: `run_results.txt` - Raw analysis results
- **Statistics**: `results.csv` - Summary statistics by configuration
- **Cache**: `data/h5/*.h5` - Processed data for faster reloading
- **Reports**: `summary_report.txt`, `flyback_summary_table.txt`

## Command Reference

### `analyze` - Process Sensor Data
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze [OPTIONS]

Options:
  -i, --input PATH           Input CSV file(s) [required]
  -o, --output PATH          Output directory [default: out]
  --param-y CHOICE           Y-axis parameter [default: SUM]
  --param-x CHOICE           X-axis parameter [default: time]  
  --show-all-sensors         Plot all individual sensors
```

### `postprocess` - Generate Statistical Analysis
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess [OPTIONS]

Options:
  -i, --input PATH           Results text file [required]
  -o, --output PATH          Output directory [default: out]
  --plot-type CHOICE         Plot type: box|violin|dual|all [default: all]
  --generate-table           Also generate summary table
```

### `batch` - Process Multiple Files  
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis batch [OPTIONS]

Options:
  -d, --data-dir PATH        Data directory [required]
  -o, --output PATH          Output directory [default: out]
  --summary                  Generate summary statistics
```

### `table` - Generate Summary Table
```console
$ poetry run python -m Fishing_Line_Flyback_Impact_Analysis table [OPTIONS]

Options:
  -i, --input PATH           Results text file [required]
  -o, --output PATH          Output directory [default: out]
```

## Example Workflows

### Complete Analysis Pipeline
```bash
# 1. Analyze all test files
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze -i data/*.csv

# 2. Generate comprehensive plots and statistics  
poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess -i out/run_results.txt --generate-table

# 3. Create publication-ready summary
poetry run python -m Fishing_Line_Flyback_Impact_Analysis table -i out/run_results.txt
```

### Configuration Comparison Study
```bash
# Compare specific configurations
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze -i data/STND-*.csv data/BR-*.csv
poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess -i out/run_results.txt --plot-type dual
```

## Research Applications

This package is designed for researchers studying:
- **Fishing safety**: Quantifying flyback impact risks
- **Equipment design**: Comparing weight configuration effectiveness  
- **Biomechanics**: Understanding impact forces and injury potential
- **Materials testing**: Analyzing line failure and energy dissipation

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
*Fishing Line Flyback Impact Analysis* is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{flyback_impact_analysis_2025,
    title = {Fishing Line Flyback Impact Analysis},
    author = {Nanosystems Lab},
    year = {2025},
    url = {https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis},
    version = {0.0.1}
}
```

## Credits

This project was generated from [@nanosystemslab]'s [Nanosystems Lab Python Cookiecutter] template.

---

[pypi status]: https://pypi.org/project/Fishing_Line_Flyback_Impact_Analysis/
[read the docs]: https://Fishing_Line_Flyback_Impact_Analysis.readthedocs.io/
[tests]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[@nanosystemslab]: https://github.com/nanosystemslab
[pypi]: https://pypi.org/
[Nanosystems Lab Python Cookiecutter]: https://github.com/nanosystemslab/cookiecutter-nanosystemslab
[file an issue]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/issues
[pip]: https://pip.pypa.io/
[license]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/blob/main/LICENSE
[contributor guide]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/blob/main/CONTRIBUTING.md
[command-line reference]: https://Fishing_Line_Flyback_Impact_Analysis.readthedocs.io/en/latest/usage.html
