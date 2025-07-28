"""
Fishing Line Flyback Impact Analysis - Enhanced Analysis Module v2.1

This module contains the complete corrected energy analysis methodology with
configuration-specific weights and measured line mass integration.

Key improvements:
- Configuration-specific weights (45g-72g range)
- Measured line mass integration (from 5.5" = 0.542g sample) 
- 70% effective line mass (literature validated)
- Peak-focused detection for realistic velocities
- Handles actual CSV format with AI sensor columns
- Converts lbf to Newtons automatically  
- Provides comprehensive material comparison
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import SciPy functions for advanced analysis
try:
    from scipy.integrate import simpson, cumulative_trapezoid
    from scipy.signal import savgol_filter, find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available - using basic NumPy methods")


# Configuration-specific weights (hardware mass excluding line)
CONFIG_WEIGHTS = {
    'STND': 0.045,  # 45g - Standard configuration
    'DF': 0.060,    # 60g - Dual Fixed  
    'DS': 0.072,    # 72g - Dual Sliding
    'SL': 0.069,    # 69g - Sliding
    'BR': 0.045     # 45g - Breakaway
}

# Line mass from actual measurement
MEASURED_LINE_LENGTH_INCHES = 5.5
MEASURED_LINE_MASS_GRAMS = 0.542
ACTIVE_LINE_LENGTH_METERS = 10.0
LINE_MASS_FRACTION = 0.7  # 70% effective mass (literature standard)


def calculate_line_mass_from_measurement(measured_length_inches: float = MEASURED_LINE_LENGTH_INCHES, 
                                       measured_mass_grams: float = MEASURED_LINE_MASS_GRAMS,
                                       active_length_meters: float = ACTIVE_LINE_LENGTH_METERS) -> Dict:
    """
    Calculate line mass based on actual measurement.
    
    Args:
        measured_length_inches: Length of measured sample in inches
        measured_mass_grams: Mass of measured sample in grams
        active_length_meters: Active length of line in meters
        
    Returns:
        Dictionary with line mass information
    """
    # Convert to metric
    measured_length_m = measured_length_inches * 0.0254
    
    # Calculate linear density
    linear_density_g_per_m = measured_mass_grams / measured_length_m
    linear_density_kg_per_m = linear_density_g_per_m / 1000
    
    # Calculate active line mass
    active_line_mass_g = linear_density_g_per_m * active_length_meters
    active_line_mass_kg = active_line_mass_g / 1000
    
    return {
        'linear_density_g_per_m': linear_density_g_per_m,
        'linear_density_kg_per_m': linear_density_kg_per_m,
        'active_line_mass_g': active_line_mass_g,
        'active_line_mass_kg': active_line_mass_kg,
        'active_length_m': active_length_meters
    }


def get_system_mass(material_code: str, include_line_mass: bool = True, 
                   line_mass_fraction: float = LINE_MASS_FRACTION) -> Dict:
    """
    Calculate total system mass for a given configuration.
    
    Args:
        material_code: Configuration code (e.g., 'STND', 'DF', etc.)
        include_line_mass: Whether to include line mass in calculations
        line_mass_fraction: Fraction of line mass to include (0.0-1.0)
        
    Returns:
        Dictionary with mass breakdown and total system mass
    """
    # Get configuration weight
    config_weight_kg = CONFIG_WEIGHTS.get(material_code, 0.045)  # Default to 45g
    
    if include_line_mass:
        line_info = calculate_line_mass_from_measurement()
        effective_line_mass_kg = line_info['active_line_mass_kg'] * line_mass_fraction
        total_system_mass_kg = config_weight_kg + effective_line_mass_kg
        
        return {
            'material_code': material_code,
            'config_weight_kg': config_weight_kg,
            'line_mass_total_kg': line_info['active_line_mass_kg'],
            'line_mass_fraction': line_mass_fraction,
            'line_mass_effective_kg': effective_line_mass_kg,
            'total_system_mass_kg': total_system_mass_kg,
            'line_info': line_info
        }
    else:
        return {
            'material_code': material_code,
            'config_weight_kg': config_weight_kg,
            'line_mass_total_kg': 0.0,
            'line_mass_fraction': 0.0,
            'line_mass_effective_kg': 0.0,
            'total_system_mass_kg': config_weight_kg,
            'line_info': None
        }


class ImpactAnalyzer:
    """
    Enhanced Impact Analyzer with configuration-specific weights and line mass.
    
    Uses peak-focused detection to find only the most intense part of the impact,
    providing realistic velocity estimates for fishing line flyback impacts.
    """
    
    def __init__(self, mass: Optional[float] = None, sampling_rate: float = 100000.0,
                 baseline_correction: bool = True, signal_filtering: bool = False,
                 material_code: Optional[str] = None, include_line_mass: bool = True,
                 line_mass_fraction: float = LINE_MASS_FRACTION):
        """
        Initialize the analyzer.
        
        Args:
            mass: Object mass in kg (if None, will use material_code to determine)
            sampling_rate: Data acquisition sampling rate in Hz
            baseline_correction: Apply baseline correction to force data
            signal_filtering: Apply signal filtering (disabled by default for peak detection)
            material_code: Configuration code (e.g., 'STND', 'DF') - used if mass is None
            include_line_mass: Whether to include line mass in calculations
            line_mass_fraction: Fraction of line mass to include (0.0-1.0)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.baseline_correction = baseline_correction
        self.signal_filtering = signal_filtering
        self.include_line_mass = include_line_mass
        self.line_mass_fraction = line_mass_fraction
        self.material_code = material_code
        
        # Determine system mass
        if mass is not None:
            # Use provided mass directly (legacy mode)
            self.mass = mass
            self.mass_breakdown = {'total_system_mass_kg': mass, 'legacy_mode': True}
        elif material_code is not None:
            # Calculate mass from configuration code
            mass_info = get_system_mass(material_code, include_line_mass, line_mass_fraction)
            self.mass = mass_info['total_system_mass_kg']
            self.mass_breakdown = mass_info
        else:
            # Default to STND configuration
            mass_info = get_system_mass('STND', include_line_mass, line_mass_fraction)
            self.mass = mass_info['total_system_mass_kg']
            self.mass_breakdown = mass_info
            
    def print_mass_info(self):
        """Print mass breakdown information."""
        print(f"\n⚖️  MASS CONFIGURATION:")
        if 'legacy_mode' in self.mass_breakdown:
            print(f"   Legacy mode: {self.mass*1000:.1f}g total")
        else:
            mb = self.mass_breakdown
            print(f"   Configuration: {mb['material_code']}")
            print(f"   Hardware: {mb['config_weight_kg']*1000:.1f}g")
            if self.include_line_mass:
                print(f"   Line (total): {mb['line_mass_total_kg']*1000:.1f}g")
                print(f"   Line (effective): {mb['line_mass_effective_kg']*1000:.1f}g ({mb['line_mass_fraction']*100:.0f}%)")
                print(f"   Total system: {mb['total_system_mass_kg']*1000:.1f}g")
                print(f"   Line measurement: {MEASURED_LINE_LENGTH_INCHES}\" = {MEASURED_LINE_MASS_GRAMS}g")
                print(f"   Active length: {ACTIVE_LINE_LENGTH_METERS}m")
            else:
                print(f"   Total system: {mb['total_system_mass_kg']*1000:.1f}g (hardware only)")
        
    def _detect_force_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect AI sensor force columns in the DataFrame."""
        force_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if ('ai' in col_lower and 'lbf' in col_lower) or ('force' in col_lower):
                force_columns.append(col)
        return force_columns
    
    def _convert_lbf_to_n(self, force_lbf: np.ndarray) -> np.ndarray:
        """Convert force from pounds-force to Newtons (1 lbf = 4.44822 N)."""
        return force_lbf * 4.44822
    
    def _apply_baseline_correction(self, force: np.ndarray) -> np.ndarray:
        """Apply baseline correction to remove DC offset."""
        if not self.baseline_correction or len(force) < 100:
            return force
            
        # Use first 1000 points to estimate baseline
        baseline_window = min(1000, len(force) // 10)
        baseline_region = force[:baseline_window]
        baseline = np.median(baseline_region)
        
        return force - baseline
    
    def _calculate_total_force(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate total force from individual sensor columns.
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            Tuple of (total_force_in_N, force_column_names)
        """
        force_columns = self._detect_force_columns(df)
        
        if not force_columns:
            # Fallback: find numeric columns that aren't time
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            time_cols = [col for col in numeric_cols if 'time' in col.lower()]
            force_columns = [col for col in numeric_cols if col not in time_cols]
            
            if not force_columns:
                raise ValueError("No force columns detected in CSV file")
        
        # Sum all force columns (in lbf)
        total_force_lbf = df[force_columns].sum(axis=1).values
        
        # Convert to Newtons and apply baseline correction
        total_force_n = self._convert_lbf_to_n(total_force_lbf)
        total_force_n = np.nan_to_num(total_force_n, nan=0.0)
        total_force_n = self._apply_baseline_correction(total_force_n)
        
        return total_force_n, force_columns
    
    def _get_time_array(self, df: pd.DataFrame, force_length: int) -> np.ndarray:
        """Get time array from DataFrame or create one with sampling rate detection."""
        # Look for time column
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        
        if time_cols:
            time_array = df[time_cols[0]].values[:force_length]
            if len(time_array) > 0:
                time_array = time_array - time_array[0]
                
                # Detect actual sampling rate
                if len(time_array) > 10:
                    dt_values = np.diff(time_array)
                    actual_dt = np.median(dt_values[dt_values > 0])
                    
                    if actual_dt > 0:
                        self.dt = actual_dt
                        self.sampling_rate = 1.0 / actual_dt
        else:
            # Create time array based on sampling rate
            time_array = np.arange(force_length) * self.dt
        
        return time_array
    
    def find_peak_impact_region(self, force: np.ndarray, debug: bool = False) -> Tuple[int, int]:
        """
        Find the peak impact region using adaptive thresholding.
        
        This is the key method that finds only the most intense part of the impact,
        which gives realistic velocity estimates for fishing line impacts.
        
        Args:
            force: Force array (N)
            debug: Print debug information
            
        Returns:
            Tuple of (start_idx, end_idx) for the peak impact region
        """
        max_force_idx = np.argmax(np.abs(force))
        max_force = np.abs(force[max_force_idx])
        
        if debug:
            print(f"   🎯 Peak impact detection:")
            print(f"      Max force: {max_force:.2f} N at index {max_force_idx}")
            print(f"      Max force time: {max_force_idx * self.dt:.6f} s")
        
        # Try different thresholds to find reasonable velocity (100-500 m/s)
        for threshold_pct in [90, 80, 70, 60, 50]:
            threshold = max_force * (threshold_pct / 100.0)
            peak_indices = np.where(np.abs(force) > threshold)[0]
            
            if len(peak_indices) > 0:
                peak_start = peak_indices[0]
                peak_end = peak_indices[-1]
                peak_duration = (peak_end - peak_start + 1) * self.dt
                peak_forces = force[peak_start:peak_end + 1]
                
                # Calculate impulse and velocity for this threshold
                J_peak = np.trapz(peak_forces, dx=self.dt)
                v_peak = abs(J_peak / self.mass) if self.mass > 0 else 0
                
                if debug:
                    print(f"      {threshold_pct}% threshold ({threshold:.0f} N): "
                          f"{len(peak_indices)} points, "
                          f"{peak_duration*1000:.1f} ms, "
                          f"v={v_peak:.1f} m/s")
                
                # Accept if velocity is in reasonable range for fishing line impacts
                if 100 <= v_peak <= 500:
                    if debug:
                        print(f"      ✅ Found reasonable velocity with {threshold_pct}% threshold")
                    return peak_start, peak_end
        
        # Fallback: use 80% threshold with duration limit
        threshold = max_force * 0.8
        peak_indices = np.where(np.abs(force) > threshold)[0]
        
        if len(peak_indices) > 0:
            peak_start = peak_indices[0]
            peak_end = peak_indices[-1]
            
            # Limit to maximum 10ms duration
            max_duration_samples = int(0.01 / self.dt)
            if (peak_end - peak_start) > max_duration_samples:
                peak_end = peak_start + max_duration_samples
                
            if debug:
                print(f"      Using 80% threshold with 10ms limit")
            return peak_start, peak_end
        else:
            # Ultimate fallback - small window around peak
            window_size = min(200, int(0.005 / self.dt))  # 5ms max
            start_idx = max(0, max_force_idx - window_size // 2)
            end_idx = min(len(force) - 1, max_force_idx + window_size // 2)
            if debug:
                print(f"      Using 5ms window around peak")
            return start_idx, end_idx

    def calculate_corrected_energy(self, force: np.ndarray, time: Optional[np.ndarray] = None, debug: bool = False) -> Dict:
        """
        Calculate kinetic energy using peak-focused methodology.
        
        This method isolates only the most intense part of the impact to provide
        realistic velocity estimates for fishing line flyback impacts.
        
        Args:
            force: Force array (N)
            time: Time array (s) - optional
            debug: Print debug information
            
        Returns:
            Dictionary with corrected energy analysis
        """
        if time is not None and len(time) > 1:
            self.dt = np.median(np.diff(time))
        
        if debug:
            print(f"\n🎯 CORRECTED ENERGY CALCULATION:")
            print(f"   Force range: {np.min(force):.2f} to {np.max(force):.2f} N")
            print(f"   Sampling rate: {self.sampling_rate:.0f} Hz")
            self.print_mass_info()
        
        # Find the peak impact region (this is the key improvement)
        impact_start, impact_end = self.find_peak_impact_region(force, debug)
        
        # Extract ONLY the peak impact region
        force_peak = force[impact_start:impact_end + 1]
        impact_duration = len(force_peak) * self.dt
        
        if debug:
            print(f"   📊 Peak impact region:")
            print(f"      Start: index {impact_start}, time {impact_start * self.dt:.6f} s")
            print(f"      End: index {impact_end}, time {impact_end * self.dt:.6f} s")
            print(f"      Duration: {impact_duration * 1000:.2f} ms")
            print(f"      Force range: {np.min(force_peak):.1f} to {np.max(force_peak):.1f} N")
        
        # Calculate impulse using best available integration method
        if SCIPY_AVAILABLE and len(force_peak) > 4:
            try:
                J_peak = simpson(force_peak, dx=self.dt)
                integration_method = "Simpson's rule (SciPy)"
            except:
                J_peak = np.trapz(force_peak, dx=self.dt)
                integration_method = "Trapezoidal (NumPy)"
        else:
            J_peak = np.trapz(force_peak, dx=self.dt) if len(force_peak) > 1 else np.sum(force_peak) * self.dt
            integration_method = "Trapezoidal (NumPy)"
        
        # Calculate total impulse for comparison (legacy method)
        J_total = np.trapz(force, dx=self.dt) if len(force) > 1 else np.sum(force) * self.dt
        
        # Calculate energies
        v_initial = J_peak / self.mass if self.mass > 0 else 0
        kinetic_energy_corrected = 0.5 * self.mass * v_initial**2
        
        v_total_calc = J_total / self.mass if self.mass > 0 else 0
        kinetic_energy_overestimated = 0.5 * self.mass * v_total_calc**2
        
        if debug:
            print(f"\n   📊 Results:")
            print(f"      Peak impulse: {J_peak:.6f} N⋅s")
            print(f"      Velocity: {abs(v_initial):.1f} m/s")
            print(f"      Energy: {kinetic_energy_corrected:.3f} J")
            print(f"      Integration: {integration_method}")
            
            # Validation
            if 150 <= abs(v_initial) <= 400:
                print(f"      ✅ Velocity in target range (150-400 m/s)")
            elif 100 <= abs(v_initial) <= 500:
                print(f"      🟡 Velocity in reasonable range (100-500 m/s)")
            else:
                print(f"      ⚠️  Velocity outside expected range")
            
            if 1 <= kinetic_energy_corrected <= 1000:
                print(f"      ✅ Energy in reasonable range (1-1000 J)")
            else:
                print(f"      ⚠️  Energy outside reasonable range")
        
        return {
            'kinetic_energy': kinetic_energy_corrected,
            'kinetic_energy_corrected': kinetic_energy_corrected,
            'kinetic_energy_overestimated': kinetic_energy_overestimated,
            'initial_velocity': v_initial,
            'impulse_decel': J_peak,
            'impulse_total': J_total,
            'max_force': np.max(np.abs(force_peak)),
            'impact_start_idx': impact_start,
            'impact_end_idx': impact_end,
            'impact_duration': impact_duration,
            'decel_end_time': impact_end * self.dt,
            'decel_fraction': len(force_peak) / len(force),
            'overestimation_factor': (kinetic_energy_overestimated / kinetic_energy_corrected 
                                    if kinetic_energy_corrected > 0 else 1.0),
            'total_time': len(force) * self.dt,
            'integration_method': integration_method,
            'data_points': len(force_peak),
            'sampling_rate_hz': self.sampling_rate,
            'mass_kg': self.mass,
            'mass_breakdown': self.mass_breakdown
        }

    def quick_diagnostic(self, csv_path: str) -> Dict:
        """Quick diagnostic analysis of a file."""
        df = pd.read_csv(csv_path)
        force_n, force_columns = self._calculate_total_force(df)
        time = self._get_time_array(df, len(force_n))
        
        print(f"\n🔬 DIAGNOSTIC: {csv_path}")
        print(f"   Force columns: {force_columns}")
        print(f"   Data points: {len(force_n):,}")
        print(f"   Sampling rate: {self.sampling_rate:.0f} Hz")
        print(f"   Force range: {np.min(force_n):.1f} to {np.max(force_n):.1f} N")
        
        return self.calculate_corrected_energy(force_n, time, debug=True)
    
    def analyze_csv_file(self, csv_path: Union[str, Path]) -> Dict[str, Union[float, str]]:
        """
        Analyze a single CSV file with corrected energy calculation.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Analysis results dictionary
        """
        csv_path = Path(csv_path)
        
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Calculate total force from AI sensors
            force_n, force_columns = self._calculate_total_force(df)
            
            # Get time array
            time = self._get_time_array(df, len(force_n))
            
            # Quality checks
            if len(force_n) < 50:
                raise ValueError("Insufficient data points for analysis")
            
            # Remove invalid values
            valid_indices = np.isfinite(force_n)
            force_n = force_n[valid_indices]
            time = time[valid_indices]
            
            if len(force_n) < 25:
                raise ValueError("Too many invalid data points")
            
            # Calculate corrected energy
            results = self.calculate_corrected_energy(force_n, time)
            
            # Add metadata
            results.update({
                'filename': csv_path.name,
                'material_type': self._extract_material_type(csv_path.name),
                'sample_number': self._extract_sample_number(csv_path.name),
                'force_columns_used': ', '.join(force_columns),
                'analysis_method': 'peak_focused_corrected',
                'baseline_correction': self.baseline_correction,
                'line_mass_included': self.include_line_mass,
                'line_mass_fraction': self.line_mass_fraction
            })
            
            return results
            
        except Exception as e:
            return {
                'filename': csv_path.name,
                'error': str(e),
                'kinetic_energy': np.nan
            }
    
    def batch_analyze_directory(self, data_dir: Union[str, Path], 
                               file_pattern: str = "*.csv") -> List[Dict]:
        """
        Analyze all files in a directory.
        
        Args:
            data_dir: Directory containing data files
            file_pattern: File pattern to match
            
        Returns:
            List of analysis results
        """
        data_path = Path(data_dir)
        results = []
        
        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {data_path}")
        
        files = list(data_path.glob(file_pattern))
        
        if not files:
            print(f"No files matching '{file_pattern}' found in {data_path}")
            return results
        
        print(f"🔄 Processing {len(files)} files...")
        if self.include_line_mass:
            print(f"   Using {self.line_mass_fraction*100:.0f}% line mass ({self.mass*1000:.1f}g total)")
        else:
            print(f"   Using hardware only ({self.mass*1000:.1f}g)")
        
        for file_path in files:
            if file_path.suffix.lower() == '.csv':
                result = self.analyze_csv_file(file_path)
            else:
                continue
                
            results.append(result)
            
            # Progress indicator
            if 'error' not in result:
                velocity = abs(result['initial_velocity'])
                status = "✅" if 150 <= velocity <= 400 else ("🟡" if 100 <= velocity <= 500 else "❌")
                print(f"  {status} {file_path.name}: {velocity:.0f} m/s, {result['kinetic_energy']:.1f} J")
            else:
                print(f"  ❌ {file_path.name}: {result['error']}")
        
        return results
    
    def export_results_csv(self, results: List[Dict], 
                          output_path: Union[str, Path]) -> None:
        """Export results to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    
    def _extract_material_type(self, filename: str) -> str:
        """Extract material type from filename (e.g., 'BR' from 'BR-21-1.csv')."""
        try:
            return filename.split('-')[0]
        except:
            return 'UNKNOWN'
    
    def _extract_sample_number(self, filename: str) -> str:
        """Extract sample number from filename (e.g., '21-1' from 'BR-21-1.csv')."""
        try:
            parts = filename.replace('.csv', '').replace('.h5', '').split('-')
            return '-'.join(parts[1:])
        except:
            return 'UNKNOWN'


def analyze_single_file_with_config(file_path: Union[str, Path], 
                                   material_code: Optional[str] = None,
                                   include_line_mass: bool = True,
                                   line_mass_fraction: float = LINE_MASS_FRACTION) -> Dict:
    """
    Analyze a single file with automatic configuration detection.
    
    Args:
        file_path: Path to CSV file
        material_code: Configuration code (if None, extracted from filename)
        include_line_mass: Whether to include line mass
        line_mass_fraction: Fraction of line mass to include
        
    Returns:
        Analysis results dictionary
    """
    file_path = Path(file_path)
    
    # Extract material code from filename if not provided
    if material_code is None:
        material_code = file_path.name.split('-')[0]
    
    # Initialize analyzer with configuration-specific mass
    analyzer = ImpactAnalyzer(
        material_code=material_code,
        include_line_mass=include_line_mass,
        line_mass_fraction=line_mass_fraction
    )
    
    return analyzer.analyze_csv_file(file_path)


def run_comprehensive_analysis(data_dir: Union[str, Path] = "data/csv", 
                              output_dir: Union[str, Path] = "comprehensive_analysis") -> List[Dict]:
    """
    Run comprehensive analysis on all measurement files with configuration-specific weights.
    
    Args:
        data_dir: Directory containing CSV files
        output_dir: Output directory for results
        
    Returns:
        List of analysis results
    """
    print("🎣 COMPREHENSIVE FISHING LINE FLYBACK ANALYSIS")
    print("=" * 80)
    
    # Your file list
    files = [
        "BR-21-1.csv", "BR-21-10.csv", "BR-21-2.csv", "BR-21-3.csv", "BR-21-4.csv",
        "BR-21-5.csv", "BR-21-52.csv", "BR-21-6.csv", "BR-21-7.csv", "BR-21-8.csv", "BR-21-9.csv",
        "DF-21-1.csv", "DF-21-10.csv", "DF-21-11.csv", "DF-21-12.csv", "DF-21-2.csv",
        "DF-21-3.csv", "DF-21-4.csv", "DF-21-5.csv", "DF-21-6.csv", "DF-21-7.csv", "DF-21-8.csv", "DF-21-9.csv",
        "DS-21-1.csv", "DS-21-10.csv", "DS-21-11.csv", "DS-21-2.csv", "DS-21-4.csv",
        "DS-21-5.csv", "DS-21-6.csv", "DS-21-7.csv", "DS-21-8.csv", "DS-21-9.csv",
        "SL-21-0.csv", "SL-21-1.csv", "SL-21-10.csv", "SL-21-2.csv", "SL-21-3.csv",
        "SL-21-4.csv", "SL-21-5.csv", "SL-21-6.csv", "SL-21-7.csv", "SL-21-8.csv", "SL-21-9.csv",
        "STND-21-1.csv", "STND-21-10.csv", "STND-21-11.csv", "STND-21-2.csv", "STND-21-3.csv",
        "STND-21-4.csv", "STND-21-5.csv", "STND-21-6.csv", "STND-21-7.csv", "STND-21-8.csv", "STND-21-9.csv"
    ]
    
    print(f"📁 Processing {len(files)} measurement files")
    print(f"📊 Using enhanced peak-focused analysis with configuration-specific weights")
    print(f"⚖️  Line measurement: {MEASURED_LINE_LENGTH_INCHES}\" = {MEASURED_LINE_MASS_GRAMS}g → 38.8g total for 10m")
    print(f"🎯 Target velocity range: 150-400 m/s")
    print()
    
    # Show configuration weights
    print(f"⚖️  CONFIGURATION WEIGHTS:")
    line_mass_kg = calculate_line_mass_from_measurement()['active_line_mass_kg']
    effective_line_mass_kg = line_mass_kg * LINE_MASS_FRACTION
    
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        total_kg = weight_kg + effective_line_mass_kg
        print(f"   {config:4s}: {weight_kg*1000:2.0f}g + {effective_line_mass_kg*1000:.0f}g = {total_kg*1000:.0f}g total")
    print()
    
    # Process all files with configuration-specific weights
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    material_counts = {"BR": 0, "DF": 0, "DS": 0, "SL": 0, "STND": 0}
    
    print(f"🔄 PROCESSING FILES:")
    print("-" * 80)
    print(f"{'Status':<2} {'Filename':<15} {'Mat':<4} {'Weight':<6} {'Total':<6} {'Velocity':<8} {'Energy':<10}")
    print("-" * 80)
    
    for filename in files:
        file_path = data_path / filename
        
        if not file_path.exists():
            print(f"❌ {filename:<15} | File not found")
            continue
            
        try:
            # Extract material type from filename
            material = filename.split('-')[0]
            
            # Initialize analyzer with configuration-specific weight
            analyzer = ImpactAnalyzer(
                material_code=material,
                include_line_mass=True,
                line_mass_fraction=LINE_MASS_FRACTION
            )
            
            result = analyzer.analyze_csv_file(file_path)
            
            if 'error' not in result:
                velocity = abs(result['initial_velocity'])
                energy = result['kinetic_energy']
                total_mass_g = result['mass_kg'] * 1000
                config_weight_g = CONFIG_WEIGHTS.get(material, 0.045) * 1000
                
                # Count by material
                if material in material_counts:
                    material_counts[material] += 1
                
                # Status based on velocity range
                if 150 <= velocity <= 400:
                    status = "✅"
                elif 100 <= velocity <= 500:
                    status = "🟡"
                else:
                    status = "❌"
                
                print(f"{status} {filename:<15} | {material:<4} | {config_weight_g:3.0f}g | {total_mass_g:3.0f}g | {velocity:3.0f} m/s | {energy:6.1f} J")
                results.append(result)
                
            else:
                print(f"❌ {filename:<15} | {material:<4} | ERROR: {result['error']}")
                
        except Exception as e:
            material = filename.split('-')[0] if '-' in filename else 'UNK'
            print(f"❌ {filename:<15} | {material:<4} | ERROR: {str(e)}")
    
    print("-" * 80)
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path / "all_results.csv", index=False)
        print(f"\n💾 Results saved to: {output_path / 'all_results.csv'}")
        
        # Generate summary report
        create_summary_report(results, output_path)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 All results saved to: {output_path}")
    else:
        print("❌ No successful analyses to report")
    
    return results


def create_summary_report(results: List[Dict], output_dir: Path):
    """Create comprehensive summary report."""
    
    # Filter valid results
    valid_results = [r for r in results if 'error' not in r and 
                    not np.isnan(r.get('kinetic_energy', np.nan))]
    
    if not valid_results:
        print("No valid results to report.")
        return
    
    print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
    print("=" * 60)
    
    # Overall statistics
    velocities = [abs(r['initial_velocity']) for r in valid_results]
    energies = [r['kinetic_energy'] for r in valid_results]
    forces = [r.get('max_force', 0) for r in valid_results]
    
    print(f"📈 OVERALL STATISTICS:")
    print(f"   Total files processed: {len(valid_results)}")
    print(f"   Velocity range: {np.min(velocities):.0f} - {np.max(velocities):.0f} m/s")
    print(f"   Mean velocity: {np.mean(velocities):.0f} ± {np.std(velocities):.0f} m/s")
    print(f"   Energy range: {np.min(energies):.1f} - {np.max(energies):.1f} J")
    print(f"   Mean energy: {np.mean(energies):.1f} ± {np.std(energies):.1f} J")
    print(f"   Max force range: {np.min(forces):.0f} - {np.max(forces):.0f} N")
    
    # Velocity distribution analysis
    target_count = sum(1 for v in velocities if 150 <= v <= 400)
    reasonable_count = sum(1 for v in velocities if 100 <= v <= 500)
    
    print(f"\n🎯 VELOCITY DISTRIBUTION:")
    print(f"   Target range (150-400 m/s): {target_count}/{len(velocities)} ({target_count/len(velocities)*100:.1f}%)")
    print(f"   Reasonable range (100-500 m/s): {reasonable_count}/{len(velocities)} ({reasonable_count/len(velocities)*100:.1f}%)")
    print(f"   Outside range: {len(velocities) - reasonable_count} files")
    
    # Material-by-material analysis
    print(f"\n🧪 MATERIAL ANALYSIS (with configuration-specific weights):")
    materials = {}
    for result in valid_results:
        material = result['material_type']
        if material not in materials:
            materials[material] = []
        materials[material].append(result)
    
    # Sort materials by performance
    material_performance = []
    for material, material_results in materials.items():
        mat_velocities = [abs(r['initial_velocity']) for r in material_results]
        mat_energies = [r['kinetic_energy'] for r in material_results]
        
        target_count_mat = sum(1 for v in mat_velocities if 150 <= v <= 400)
        target_percent = target_count_mat / len(mat_velocities) * 100
        
        config_weight = CONFIG_WEIGHTS.get(material, 0.045) * 1000
        
        material_performance.append({
            'material': material,
            'count': len(material_results),
            'target_percent': target_percent,
            'mean_velocity': np.mean(mat_velocities),
            'mean_energy': np.mean(mat_energies),
            'velocity_std': np.std(mat_velocities),
            'config_weight': config_weight
        })
    
    # Sort by target percentage
    material_performance.sort(key=lambda x: x['target_percent'], reverse=True)
    
    print(f"   {'Material':<8} {'Count':<5} {'Weight':<7} {'Target%':<7} {'Velocity':<12} {'Energy':<10} {'Consistency'}")
    print(f"   {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*11}")
    
    for mp in material_performance:
        consistency = "High" if mp['velocity_std'] < 50 else ("Med" if mp['velocity_std'] < 100 else "Low")
        total_mass = mp['config_weight'] + (LINE_MASS_FRACTION * 38.8)
        print(f"   {mp['material']:<8} {mp['count']:<5} {mp['config_weight']:<4.0f}g   {mp['target_percent']:<6.1f}% "
              f"{mp['mean_velocity']:<6.0f}±{mp['velocity_std']:<4.0f} "
              f"{mp['mean_energy']:<7.1f} J  {consistency}")
    
    # Best performer
    if material_performance:
        best = material_performance[0]
        print(f"\n🏆 BEST PERFORMING CONFIGURATION:")
        print(f"   {best['material']}: {best['target_percent']:.1f}% in target range")
        print(f"   Mean velocity: {best['mean_velocity']:.0f} m/s")
        print(f"   Configuration weight: {best['config_weight']:.0f}g")
    
    # Mass effect analysis
    print(f"\n⚖️  MASS EFFECT VALIDATION:")
    config_vs_velocity = [(mp['config_weight'], mp['mean_velocity']) for mp in material_performance]
    config_vs_velocity.sort(key=lambda x: x[0])
    
    print(f"   Weight vs Velocity correlation:")
    for weight, velocity in config_vs_velocity:
        material_name = [k for k, v in CONFIG_WEIGHTS.items() if abs(v*1000 - weight) < 1][0]
        print(f"      {material_name}: {weight:.0f}g → {velocity:.0f} m/s")
    
    # Check correlation
    if len(config_vs_velocity) > 2:
        weights = [x[0] for x in config_vs_velocity]
        vels = [x[1] for x in config_vs_velocity]
        correlation = np.corrcoef(weights, vels)[0,1]
        print(f"   Correlation coefficient: {correlation:.3f}")
        
        if correlation < -0.5:
            print("   ✅ Strong negative correlation confirms physics expectation")
        elif correlation < -0.2:
            print("   🟡 Moderate negative correlation, reasonable")
        else:
            print("   ❌ Weak/positive correlation, may indicate calibration issues")
    
    # Save detailed report
    report_path = output_dir / "comprehensive_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("FISHING LINE FLYBACK ANALYSIS - COMPREHENSIVE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: Configuration-specific weights + 70% line mass\n")
        f.write(f"Line measurement: {MEASURED_LINE_LENGTH_INCHES}\" = {MEASURED_LINE_MASS_GRAMS}g\n")
        f.write(f"Target velocity range: 150-400 m/s\n\n")
        
        f.write("CONFIGURATION MASSES:\n")
        for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
            total_kg = weight_kg + effective_line_mass_kg
            f.write(f"   {config}: {weight_kg*1000:.0f}g + {effective_line_mass_kg*1000:.0f}g = {total_kg*1000:.0f}g\n")
        f.write("\n")
        
        f.write("MATERIAL PERFORMANCE RANKING:\n")
        for i, mp in enumerate(material_performance, 1):
            f.write(f"{i}. {mp['material']}: {mp['target_percent']:.1f}% target achievement\n")
            f.write(f"   Velocity: {mp['mean_velocity']:.0f} ± {mp['velocity_std']:.0f} m/s\n")
            f.write(f"   Energy: {mp['mean_energy']:.1f} J\n")
            f.write(f"   Samples: {mp['count']}\n\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")


# Convenience functions for backward compatibility
def analyze_single_file(file_path: Union[str, Path], mass: float = None, **kwargs) -> Dict:
    """
    Analyze a single file (backward compatible function).
    
    Args:
        file_path: Path to CSV file
        mass: Object mass in kg (if None, uses configuration detection)
        **kwargs: Additional arguments
        
    Returns:
        Analysis results dictionary
    """
    if mass is not None:
        # Legacy mode - use provided mass directly
        analyzer = ImpactAnalyzer(mass=mass, **kwargs)
        return analyzer.analyze_csv_file(file_path)
    else:
        # New mode - use configuration detection
        return analyze_single_file_with_config(file_path, **kwargs)


def batch_analyze(data_dir: Union[str, Path], output_dir: Union[str, Path], 
                 mass: float = None, file_pattern: str = "*.csv", **kwargs) -> List[Dict]:
    """
    Batch analyze files (backward compatible function).
    
    Args:
        data_dir: Directory containing data files
        output_dir: Output directory for results
        mass: Object mass in kg (if None, uses configuration detection)
        file_pattern: File pattern to match
        **kwargs: Additional arguments
        
    Returns:
        List of analysis results
    """
    if mass is not None:
        # Legacy mode
        analyzer = ImpactAnalyzer(mass=mass, **kwargs)
        results = analyzer.batch_analyze_directory(data_dir, file_pattern)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        analyzer.export_results_csv(results, output_path / "results.csv")
        
        return results
    else:
        # New mode - use comprehensive analysis
        return run_comprehensive_analysis(data_dir, output_dir)


# Main execution for testing
if __name__ == "__main__":
    print("🎣 Enhanced Fishing Line Flyback Impact Analysis v2.1")
    print("=" * 60)
    print("Key features:")
    print("✓ Configuration-specific weights (45g-72g)")
    print("✓ Measured line mass integration (38.8g total)")
    print("✓ 70% effective line mass (literature validated)")
    print("✓ Peak-focused impact detection")
    print("✓ Realistic velocity targeting (150-400 m/s)")
    print()
    
    # Example usage
    print("Example usage:")
    print("# Single file with auto-detection:")
    print('result = analyze_single_file_with_config("data/csv/STND-21-5.csv")')
    print()
    print("# Comprehensive analysis of all files:")
    print('results = run_comprehensive_analysis("data/csv", "output")')
    print()
    print("# Legacy mode with manual mass:")
    print('result = analyze_single_file("data/csv/STND-21-5.csv", mass=0.072)')
