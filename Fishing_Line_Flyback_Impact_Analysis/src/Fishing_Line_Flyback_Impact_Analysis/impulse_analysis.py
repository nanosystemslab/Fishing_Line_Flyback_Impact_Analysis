"""
Impulse Analysis Module for Fishing Line Flyback Impact Analysis

This module provides impulse-based analysis (∫ F(t) dt) as the primary
analysis method for fishing line flyback testing. It focuses on total 
momentum transfer, which is more direct and relevant for fishing line 
performance evaluation than kinetic energy estimation.

Key Features:
- Direct momentum transfer measurement via ∫ F(t) dt
- Configuration-specific weights and line mass
- Simple and robust analysis method
- More relevant to fishing line impact effectiveness
- Boundary validation plotting
- Publication-style visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from scipy import integrate, stats
from scipy.signal import savgol_filter
import warnings
import platform

# Configure matplotlib for cross-platform compatibility  
if platform.system() == "Darwin":  # macOS
    plt.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 10

# Import shared components
from .shared import (
    CONFIG_WEIGHTS, 
    LINE_MASS_FRACTION,
    DEFAULT_SAMPLING_RATE,
    IMPACT_THRESHOLD_FACTOR,
    load_csv_file,
    calculate_total_force,
    get_time_array,
    extract_material_code,
    extract_sample_number,
    get_system_mass
)


class ImpulseAnalyzer:
    """
    Impulse-based analyzer for fishing line flyback impact testing.
    
    Focuses on total momentum transfer: Impulse = ∫ F(t) dt
    This provides a cleaner, more direct measurement of fishing line
    impact effectiveness compared to kinetic energy estimation.
    """
    
    def __init__(self, material_code: Optional[str] = None,
                 include_line_mass: bool = True,
                 line_mass_fraction: float = LINE_MASS_FRACTION,
                 sampling_rate: float = DEFAULT_SAMPLING_RATE,
                 impact_threshold_factor: float = IMPACT_THRESHOLD_FACTOR):
        """
        Initialize impulse analyzer.
        
        Args:
            material_code: Configuration code (e.g., 'STND', 'DF')
            include_line_mass: Whether to include line mass in calculations
            line_mass_fraction: Effective line mass fraction
            sampling_rate: Data sampling rate in Hz
            impact_threshold_factor: Threshold factor for impact detection
        """
        self.material_code = material_code
        self.include_line_mass = include_line_mass
        self.line_mass_fraction = line_mass_fraction
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.impact_threshold_factor = impact_threshold_factor
        
        # Get mass information
        if material_code:
            self.mass_info = get_system_mass(material_code, include_line_mass, line_mass_fraction)
            self.total_mass = self.mass_info['total_mass_kg']
        else:
            self.mass_info = None
            self.total_mass = None
    
    def find_impact_boundaries(self, force: np.ndarray) -> Tuple[int, int]:
        """
        Find start and end indices of impact event.
        
        Args:
            force: Force data array
            
        Returns:
            Tuple of (start_index, end_index)
        """
        if len(force) == 0:
            return 0, 0
        
        # Find peak force
        peak_idx = np.argmax(np.abs(force))
        peak_force = np.abs(force[peak_idx])
        
        # Dynamic threshold based on peak force
        threshold = peak_force * self.impact_threshold_factor
        
        # Find all points above threshold
        above_threshold = np.abs(force) > threshold
        
        if not np.any(above_threshold):
            # Fallback: use smaller threshold
            threshold = peak_force * 0.01
            above_threshold = np.abs(force) > threshold
            
            if not np.any(above_threshold):
                # Last resort: just use peak region
                return max(0, peak_idx - 50), min(len(force) - 1, peak_idx + 50)
        
        # Find continuous regions above threshold
        threshold_indices = np.where(above_threshold)[0]
        
        # Find the region containing the peak
        peak_region_mask = (threshold_indices >= peak_idx - 100) & (threshold_indices <= peak_idx + 100)
        peak_region_indices = threshold_indices[peak_region_mask]
        
        if len(peak_region_indices) > 0:
            start_idx = peak_region_indices[0]
            end_idx = peak_region_indices[-1]
        else:
            # Use all threshold indices if no peak region found
            start_idx = threshold_indices[0]
            end_idx = threshold_indices[-1]
        
        # Ensure reasonable bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(force) - 1, end_idx)
        
        # Ensure minimum duration (at least 10 samples)
        if end_idx - start_idx < 10:
            center = (start_idx + end_idx) // 2
            start_idx = max(0, center - 5)
            end_idx = min(len(force) - 1, center + 5)
        
        return start_idx, end_idx
    
    def calculate_impulse_metrics(self, force: np.ndarray, time: np.ndarray) -> Dict:
        """
        Calculate impulse-based metrics from force and time data.
        
        Args:
            force: Force data in Newtons
            time: Time data in seconds
            
        Returns:
            Dictionary with impulse analysis results
        """
        if len(force) == 0 or len(time) == 0:
            return {'error': 'Empty force or time data'}
        
        try:
            # Find impact boundaries
            impact_start, impact_end = self.find_impact_boundaries(force)
            
            # Extract impact region
            impact_force = force[impact_start:impact_end+1]
            impact_time = time[impact_start:impact_end+1]
            
            if len(impact_force) == 0:
                return {'error': 'No impact region identified'}
            
            # Calculate impulse metrics
            dt = np.diff(time).mean() if len(time) > 1 else self.dt
            
            # Total impulse (signed)
            total_impulse = np.trapz(force, time)
            
            # Total absolute impulse
            total_abs_impulse = np.trapz(np.abs(force), time)
            
            # Impact region impulse (signed)
            impact_impulse = np.trapz(impact_force, impact_time)
            
            # Impact region absolute impulse
            impact_abs_impulse = np.trapz(np.abs(impact_force), impact_time)
            
            # Force characteristics
            peak_force = np.max(np.abs(force))
            peak_force_positive = np.max(force)
            peak_force_negative = np.min(force)
            rms_force = np.sqrt(np.mean(force**2))
            
            # Timing
            impact_duration = time[impact_end] - time[impact_start] if impact_end > impact_start else 0
            
            # Calculate equivalent velocity and kinetic energy (for comparison)
            equivalent_velocity = np.nan
            equivalent_kinetic_energy = np.nan
            
            if self.total_mass and self.total_mass > 0:
                equivalent_velocity = np.abs(total_impulse) / self.total_mass
                equivalent_kinetic_energy = 0.5 * self.total_mass * equivalent_velocity**2
            
            return {
                # Primary impulse metrics
                'total_impulse': total_impulse,
                'total_abs_impulse': total_abs_impulse,
                'impact_impulse': impact_impulse,
                'impact_abs_impulse': impact_abs_impulse,
                
                # Force characteristics
                'peak_force': peak_force,
                'peak_force_positive': peak_force_positive,
                'peak_force_negative': peak_force_negative,
                'rms_force': rms_force,
                
                # Timing
                'impact_duration': impact_duration,
                'impact_start_time': time[impact_start] if len(time) > impact_start else 0,
                'impact_end_time': time[impact_end] if len(time) > impact_end else 0,
                'total_duration': time[-1] - time[0] if len(time) > 0 else 0,
                
                # Compatibility metrics
                'equivalent_velocity': equivalent_velocity,
                'equivalent_kinetic_energy': equivalent_kinetic_energy,
                'mass_kg': self.total_mass if self.total_mass else np.nan,
                
                # Analysis metadata
                'impact_start_idx': impact_start,
                'impact_end_idx': impact_end,
                'sampling_rate_hz': self.sampling_rate,
                'analysis_method': 'impulse_integration',
                'material_code': self.material_code,
                'mass_breakdown': self.mass_info
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def plot_impulse_boundaries(self, force: np.ndarray, time: np.ndarray, 
                               impact_start: int, impact_end: int, filename: str = "unknown"):
        """
        Plot force data with integration boundaries for validation.
        
        Args:
            force: Force data
            time: Time data  
            impact_start: Start index of impact region
            impact_end: End index of impact region
            filename: Filename for plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top plot: Full force curve with boundaries
        ax1.plot(time * 1000, force, 'b-', linewidth=1, alpha=0.7, label='Force')
        ax1.axvline(time[impact_start] * 1000, color='red', linestyle='--', 
                   linewidth=2, label=f'Impact Start ({time[impact_start]*1000:.1f} ms)')
        ax1.axvline(time[impact_end] * 1000, color='red', linestyle='--', 
                   linewidth=2, label=f'Impact End ({time[impact_end]*1000:.1f} ms)')
        
        # Highlight impact region
        impact_mask = np.zeros_like(force, dtype=bool)
        impact_mask[impact_start:impact_end+1] = True
        ax1.fill_between(time * 1000, force, where=impact_mask, alpha=0.3, 
                        color='green', label='Integration Region')
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title(f'Impulse Boundary Validation: {filename}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom plot: Zoomed impact region
        if impact_end > impact_start:
            zoom_start = max(0, impact_start - 50)
            zoom_end = min(len(force), impact_end + 50)
            
            zoom_time = time[zoom_start:zoom_end] * 1000
            zoom_force = force[zoom_start:zoom_end]
            
            ax2.plot(zoom_time, zoom_force, 'b-', linewidth=2, label='Force')
            ax2.axvline(time[impact_start] * 1000, color='red', linestyle='--', 
                       linewidth=2, label='Start')
            ax2.axvline(time[impact_end] * 1000, color='red', linestyle='--', 
                       linewidth=2, label='End')
            
            # Fill integration area
            impact_time_zoom = time[impact_start:impact_end+1] * 1000
            impact_force_zoom = force[impact_start:impact_end+1]
            ax2.fill_between(impact_time_zoom, impact_force_zoom, alpha=0.3, 
                           color='green', label='Integration Area')
            
            # Mark peak
            peak_idx = impact_start + np.argmax(np.abs(force[impact_start:impact_end+1]))
            ax2.plot(time[peak_idx] * 1000, force[peak_idx], 'ro', markersize=8, 
                    label=f'Peak ({force[peak_idx]:.0f} N)')
            
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Force (N)')
            ax2.set_title('Impact Region (Zoomed)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_csv_file(self, csv_path: Union[str, Path], show_plot: bool = False) -> Dict:
        """
        Analyze a single CSV file using impulse method.
        
        Args:
            csv_path: Path to CSV file
            show_plot: Whether to show boundary validation plot
            
        Returns:
            Analysis results dictionary
        """
        csv_path = Path(csv_path)
        
        try:
            # Load and process data
            df = load_csv_file(csv_path)
            force_data, force_columns = calculate_total_force(df)
            time_data, actual_sampling_rate = get_time_array(df, len(force_data), self.sampling_rate)
            
            # Update sampling rate if detected from data
            self.sampling_rate = actual_sampling_rate
            self.dt = 1.0 / actual_sampling_rate
            
            # Extract material code if not set
            if not self.material_code:
                self.material_code = extract_material_code(csv_path.name)
                self.mass_info = get_system_mass(self.material_code, self.include_line_mass, self.line_mass_fraction)
                self.total_mass = self.mass_info['total_mass_kg']
            
            # Perform impulse analysis
            result = self.calculate_impulse_metrics(force_data, time_data)
            
            if 'error' not in result:
                # Add file metadata
                result.update({
                    'filename': csv_path.name,
                    'file_path': str(csv_path),
                    'material_type': self.material_code,
                    'sample_number': extract_sample_number(csv_path.name),
                    'force_columns': force_columns,
                    'data_points': len(force_data)
                })
                
                # Show validation plot if requested
                if show_plot:
                    self.plot_impulse_boundaries(
                        force_data, time_data,
                        result['impact_start_idx'], result['impact_end_idx'],
                        csv_path.name
                    )
            
            return result
            
        except Exception as e:
            return {
                'error': f'Failed to analyze {csv_path.name}: {str(e)}',
                'filename': csv_path.name,
                'file_path': str(csv_path)
            }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_file_with_impulse(file_path: Union[str, Path],
                                   material_code: Optional[str] = None,
                                   include_line_mass: bool = True,
                                   line_mass_fraction: float = LINE_MASS_FRACTION,
                                   show_plot: bool = False) -> Dict:
    """
    Analyze single file with impulse method (main interface function).
    
    Args:
        file_path: Path to CSV file
        material_code: Configuration code
        include_line_mass: Whether to include line mass
        line_mass_fraction: Line mass fraction
        show_plot: Show boundary validation plot
        
    Returns:
        Analysis results dictionary
    """
    file_path = Path(file_path)
    
    if material_code is None:
        material_code = extract_material_code(file_path.name)
    
    analyzer = ImpulseAnalyzer(
        material_code=material_code,
        include_line_mass=include_line_mass,
        line_mass_fraction=line_mass_fraction
    )
    
    return analyzer.analyze_csv_file(file_path, show_plot=show_plot)


def run_impulse_analysis(data_dir: Union[str, Path] = "data/csv",
                        output_dir: Union[str, Path] = "impulse_analysis") -> List[Dict]:
    """
    Run impulse analysis on all CSV files in directory (main batch function).
    
    Args:
        data_dir: Directory containing CSV files
        output_dir: Output directory for results
        
    Returns:
        List of analysis results
    """
    print("🎯 IMPULSE-BASED FISHING LINE ANALYSIS")
    print("=" * 80)
    print("Method: Total momentum transfer via ∫ F(t) dt")
    print("Focus: Direct measurement of impact effectiveness")
    print()
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {data_path}")
        return []
    
    print(f"📁 Processing {len(csv_files)} files from {data_path}")
    print(f"📊 Results will be saved to {output_path}")
    print()
    
    # Show configuration reference
    print(f"⚖️  CONFIGURATION REFERENCE:")
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        line_mass_kg = 0.0388 * LINE_MASS_FRACTION
        total_mass_kg = weight_kg + line_mass_kg
        print(f"   {config:4s}: {weight_kg*1000:2.0f}g + {line_mass_kg*1000:.0f}g = {total_mass_kg*1000:.0f}g total")
    print()
    
    # Analyze all files
    results = []
    successful_analyses = 0
    
    print("🔄 ANALYZING FILES:")
    print("-" * 60)
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i:3d}/{len(csv_files)}] {csv_file.name:<20} ", end="", flush=True)
        
        try:
            # Extract material code
            material_code = extract_material_code(csv_file.name)
            
            # Create analyzer
            analyzer = ImpulseAnalyzer(
                material_code=material_code,
                include_line_mass=True,
                line_mass_fraction=LINE_MASS_FRACTION
            )
            
            # Analyze file
            result = analyzer.analyze_csv_file(csv_file)
            results.append(result)
            
            if 'error' not in result:
                successful_analyses += 1
                impulse = result['total_impulse']
                peak_force = result['peak_force']
                duration_ms = result['impact_duration'] * 1000
                
                # Show quick metrics
                direction = "→" if impulse > 0 else "←"
                print(f"✅ {material_code} | {impulse:+.6f} N*s {direction} | {peak_force:4.0f}N | {duration_ms:4.1f}ms")
            else:
                print(f"❌ ERROR: {result['error']}")
                
        except Exception as e:
            error_result = {
                'error': f'Processing failed: {str(e)}',
                'filename': csv_file.name,
                'file_path': str(csv_file)
            }
            results.append(error_result)
            print(f"❌ EXCEPTION: {str(e)}")
    
    print("-" * 60)
    print(f"✅ ANALYSIS COMPLETE: {successful_analyses}/{len(csv_files)} files successful")
    print()
    
    # Save results
    if results:
        # Create results DataFrame
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            results_df = pd.DataFrame(valid_results)
            
            # Save detailed results
            results_csv = output_path / 'impulse_analysis_results.csv'
            results_df.to_csv(results_csv, index=False)
            print(f"📄 Detailed results saved to: {results_csv}")
            
            # Create summary statistics
            impulses = results_df['total_impulse'].values
            abs_impulses = results_df['total_abs_impulse'].values
            peak_forces = results_df['peak_force'].values
            durations = results_df['impact_duration'].values * 1000  # Convert to ms
            
            print(f"📊 SUMMARY STATISTICS:")
            print(f"   Total impulse range: {np.min(impulses):+.6f} to {np.max(impulses):+.6f} N*s")
            print(f"   Absolute impulse range: {np.min(abs_impulses):.6f} to {np.max(abs_impulses):.6f} N*s")
            print(f"   Peak force range: {np.min(peak_forces):.0f} to {np.max(peak_forces):.0f} N")
            print(f"   Duration range: {np.min(durations):.1f} to {np.max(durations):.1f} ms")
            print()
            
            # Material breakdown
            if 'material_type' in results_df.columns:
                print(f"📋 MATERIAL BREAKDOWN:")
                material_stats = results_df.groupby('material_type').agg({
                    'total_impulse': ['count', 'mean', 'std'],
                    'total_abs_impulse': 'mean',
                    'peak_force': 'mean'
                }).round(6)
                
                for material in sorted(results_df['material_type'].unique()):
                    material_data = results_df[results_df['material_type'] == material]
                    count = len(material_data)
                    avg_impulse = material_data['total_impulse'].mean()
                    avg_abs_impulse = material_data['total_abs_impulse'].mean()
                    avg_force = material_data['peak_force'].mean()
                    
                    print(f"   {material:4s}: {count:2d} samples | "
                          f"Impulse: {avg_impulse:+.6f} N*s | "
                          f"Abs: {avg_abs_impulse:.6f} N*s | "
                          f"Force: {avg_force:.0f} N")
                print()
        
        # Save all results (including errors) as JSON for debugging
        import json
        results_json = output_path / 'impulse_analysis_full_results.json'
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"🔧 Full results (including errors) saved to: {results_json}")
    
    print(f"📁 All outputs saved to: {output_path}")
    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_impulse_boxplots(results: List[Dict], output_dir: Path, 
                           units: str = "SI") -> None:
    """
    Create publication-style box plots for impulse analysis results.
    
    Args:
        results: List of analysis results
        output_dir: Output directory for plots
        units: Unit system ("SI", "kSI", or "mixed")
    """
    # Filter valid results
    valid_results = [r for r in results if 'error' not in r and 'total_abs_impulse' in r]
    
    if not valid_results:
        print("❌ No valid results for plotting")
        return
    
    # Create DataFrame
    df = pd.DataFrame(valid_results)
    
    # Set up unit conversions
    unit_configs = {
        "SI": {
            "impulse_label": "Impulse (N*s)",
            "force_label": "Peak Force (N)",
            "impulse_scale": 1.0,
            "force_scale": 1.0,
            "suffix": "SI"
        },
        "kSI": {
            "impulse_label": "Impulse (kN*s)", 
            "force_label": "Peak Force (kN)",
            "impulse_scale": 1e-3,
            "force_scale": 1e-3,
            "suffix": "kSI"
        },
        "mixed": {
            "impulse_label": "Impulse (N*s)",
            "force_label": "Peak Force (kN)", 
            "impulse_scale": 1.0,
            "force_scale": 1e-3,
            "suffix": "mixed"
        }
    }
    
    config = unit_configs.get(units, unit_configs["SI"])
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams.update({"font.size": 16})
    
    # Impulse box plot
    impulse_data = df['total_abs_impulse'] * config['impulse_scale']
    ax1.boxplot(impulse_data, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax1.set_xlabel(config['impulse_label'])
    ax1.set_title('Total Absolute Impulse')
    ax1.grid(True, alpha=0.3)
    
    # Force box plot  
    force_data = df['peak_force'] * config['force_scale']
    ax2.boxplot(force_data, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_xlabel(config['force_label'])
    ax2.set_title('Peak Force')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    for ext in ['.svg', '.png']:
        output_file = output_dir / f"impulse_boxplots_{config['suffix']}{ext}"
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    print(f"📊 Box plots saved to: {output_dir}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎯 Impulse Analysis Module v3.0")
    print("=" * 50)
    print("This module provides impulse-based analysis for fishing line")
    print("flyback testing. It measures total momentum transfer via ∫ F(t) dt.")
    print()
    print("Key advantages:")
    print("• Direct measurement of impact effectiveness")
    print("• Simple integration of complete force curve")
    print("• More relevant to fishing line performance")
    print("• Uses shared data processing components")
    print("• Boundary validation plotting")
    print("• Publication-style visualization")
    print()
    print("Usage:")
    print("from impulse_analysis import run_impulse_analysis")
    print("results = run_impulse_analysis('data/csv', 'impulse_output')")
    print()
    print("# Single file with plotting:")
    print("from impulse_analysis import analyze_single_file_with_impulse")
    print("result = analyze_single_file_with_impulse('file.csv', show_plot=True)")
    print()
    print("# Create box plots:")
    print("from impulse_analysis import create_impulse_boxplots")
    print("create_impulse_boxplots(results, Path('output'))")
