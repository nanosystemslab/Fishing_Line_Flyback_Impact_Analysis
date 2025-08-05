# Fishing Line Flyback Impact Analysis

**Fishing Line Flyback Impact Analysis** provides tools for processing multi-sensor force data, calculating impact properties (impulse, energy, velocity), and generating publication-ready statistical comparisons across different fishing gear configurations.

## Overview

When fishing lines snap under tension, the rapid flyback creates significant impact forces that can cause injury. This package analyzes sensor data from controlled flyback tests to quantify impact properties and compare the effectiveness of different weight configurations in reducing impact energy.

## Critical Usage

### Primary Analysis Command

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv
```

This command processes all CSV files in the `data/csv` directory using impulse-based analysis to measure total momentum transfer (∫ F(t) dt).

### Critical Publication Outputs

The analysis generates two key outputs for publication:

- **`plot-box-J-F-*`** - Publication-ready box plots showing impulse (J) vs force (F) comparisons across configurations
- **`flyback_results_table.tex`** - LaTeX-formatted results table with statistical comparisons

## Key Features

- **Multi-sensor data processing** from Dewesoft CSV files
- **Impulse-based analysis** measuring total momentum transfer
- **Automated statistical comparisons** across gear configurations
- **Publication-ready visualizations** and LaTeX tables
- **Configuration-specific hardware masses** (45g-72g)

## Requirements

- **Python**: ≥3.11, <4.0
- **Core Dependencies**: pandas, numpy, scipy, matplotlib, seaborn

## Installation

```console
git clone https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis.git
cd Fishing_Line_Flyback_Impact_Analysis
poetry install
```

## Data Format

CSV files from Dewesoft with naming convention: `CONFIG-DIAMETER-FILENUM.csv`

- **CONFIG**: Test configuration (STND, DF, DS, SL, BR)
- **DIAMETER**: Line diameter (e.g., 21)
- **FILENUM**: Test run number

**Example**: `STND-21-1.csv`, `DF-21-5.csv`, `BR-21-10.csv`

## Supported Configurations

- **STND**: Standard (72g)
- **DF**: Dual Fixed (65g)
- **DS**: Dual Sliding (62g)
- **SL**: Sliding (58g)
- **BR**: Breakaway (45g)

---

## Supplemental Information

<details>
<summary><strong>Alternative Usage Examples</strong></summary>

### Single File Analysis

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse-single data/csv/STND-21-5.csv
```

### Method Comparison

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis compare-methods data/csv
```

### Statistical Plot Generation

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess -i impulse_analysis/impulse_results.csv
```

### Batch Processing

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis batch -d data/csv --summary
```

### Table Generation

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis table -i impulse_analysis/impulse_results.csv
```

</details>

<details>
<summary><strong>Additional Output Files</strong></summary>

### Generated Plots

- **Time series**: Individual file analysis plots with boundary validation
- **Box plots**: `plot-box-J-F-*.png`, `plot-box-J-F-*.svg` (publication-ready)
- **Statistical plots**: Individual impulse, force, and duration comparisons
- **Scatter plots**: Impulse vs force correlations
- **Ranking plots**: Material performance comparisons

### Data Files

- **Results**: `impulse_results.csv` - Complete analysis results
- **LaTeX**: `flyback_results_table.tex` - Publication-ready table
- **Reports**: `impulse_analysis_report.txt` - Detailed statistical analysis
- **Cache**: `data/h5/*.h5` - Processed data for faster reloading

</details>

<details>
<summary><strong>Complete Command Reference</strong></summary>

### `analyze-impulse` - Primary Analysis Command

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse [DATA_DIR]

Arguments:
  DATA_DIR                   Directory containing CSV files [required]

Options:
  -o, --output PATH          Output directory [default: impulse_analysis]
```

### `analyze-impulse-single` - Single File Analysis

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse-single [FILE_PATH]

Arguments:
  FILE_PATH                  CSV file to analyze [required]

Options:
  --debug                    Show detailed analysis plots
  --show-plot                Display boundary validation plots
```

### `compare-methods` - Compare Analysis Methods

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis compare-methods [DATA_DIR]

Arguments:
  DATA_DIR                   Directory containing CSV files [required]

Options:
  -o, --output PATH          Output directory [default: method_comparison]
```

### `postprocess` - Generate Statistical Analysis

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess [OPTIONS]

Options:
  -i, --input PATH           Results text file [required]
  -o, --output PATH          Output directory [default: out]
  --plot-type CHOICE         Plot type: box|violin|dual|all [default: all]
  --generate-table           Also generate summary table
```

### `batch` - Process Multiple Files

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis batch [OPTIONS]

Options:
  -d, --data-dir PATH        Data directory [required]
  -o, --output PATH          Output directory [default: out]
  --summary                  Generate summary statistics
```

### `table` - Generate Summary Table

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis table [OPTIONS]

Options:
  -i, --input PATH           Results text file [required]
  -o, --output PATH          Output directory [default: out]
```

</details>

<details>
<summary><strong>Debug Information</strong></summary>

### Boundary Validation and Signal Processing

For debugging purposes, the analysis includes boundary validation and signal processing visualization:

```console
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse-single data/csv/STND-21-5.csv --debug --show-plot
```

This displays:

- Signal boundary detection
- Peak identification validation
- Integration region verification
- Force curve preprocessing steps

### Analysis Method

The package uses impulse-based analysis measuring total momentum transfer via ∫ F(t) dt:

- **Direct measurement** of impact effectiveness
- **Complete force curve integration** captures full impact event
- **Configuration-specific hardware masses** (45g-72g automatically detected)
- **Measured line mass integration** (38.8g total, 70% effective)
- **Peak-focused impact detection** with realistic velocity targeting

</details>

<details>
<summary><strong>Example Analysis Pipeline</strong></summary>

### Complete Workflow

```bash
# 1. Analyze all test files with impulse method
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv

# 2. Generate comprehensive plots and statistics
poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess -i impulse_analysis/impulse_results.csv --generate-table

# 3. Create publication-ready summary
poetry run python -m Fishing_Line_Flyback_Impact_Analysis table -i impulse_analysis/impulse_results.csv
```

### Configuration Comparison Study

```bash
# Compare specific configurations using impulse analysis
poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv
poetry run python -m Fishing_Line_Flyback_Impact_Analysis postprocess -i impulse_analysis/impulse_results.csv --plot-type dual
```

</details>

## Research Applications

This package is designed for researchers studying:

- **Fishing safety**: Quantifying flyback impact risks
- **Equipment design**: Comparing weight configuration effectiveness
- **Biomechanics**: Understanding impact forces and injury potential
- **Materials testing**: Analyzing line failure and energy dissipation

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

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license], _Fishing Line Flyback Impact Analysis_ is free and open source software.

---

[license]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/blob/main/LICENSE
[contributor guide]: https://github.com/nanosystemslab/Fishing_Line_Flyback_Impact_Analysis/blob/main/CONTRIBUTING.md
