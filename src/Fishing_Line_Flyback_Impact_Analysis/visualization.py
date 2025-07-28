"""
Fishing Line Flyback Impact Analysis - Consolidated Visualization Module

This module contains all visualization functions for the corrected energy analysis.
Creates publication-ready plots showing methodology improvements and results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Union, Optional
import warnings
from scipy import signal

# Set publication-ready style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Configure matplotlib for better plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


class ImpactVisualizer:
    """
    Enhanced Impact Visualizer for corrected energy analysis results.
    
    Creates comprehensive visualizations showing:
    - Material comparison and performance
    - Corrected vs overestimated energy analysis
    - Force characteristics and correlations
    - Methodology validation plots
    """
    
    def __init__(self, results: List[Dict] = None):
        """
        Initialize visualizer.
        
        Args:
            results: List of analysis results from ImpactAnalyzer
        """
        self.results = results
        if results:
            self.df = self._prepare_dataframe(results)
        else:
            self.df = None
    
    def _prepare_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Prepare and clean results DataFrame for visualization."""
        df = pd.DataFrame(results)
        
        # Filter valid results
        valid_mask = (df['kinetic_energy'].notna() & 
                     np.isfinite(df['kinetic_energy']) &
                     (df['kinetic_energy'] > 0))
        
        df_clean = df[valid_mask].copy()
        
        # Remove extreme outliers for better visualization
        if len(df_clean) > 10:
            energy_q99 = df_clean['kinetic_energy'].quantile(0.99)
            df_clean = df_clean[df_clean['kinetic_energy'] <= energy_q99]
        
        return df_clean
    
    def create_executive_summary_plot(self, output_path: Optional[Path] = None) -> plt.Figure:
        """Create executive summary dashboard."""
        
        if self.df is None or len(self.df) == 0:
            raise ValueError("No valid data available for plotting")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Fishing Line Flyback Impact Analysis\nCorrected Energy Analysis Results', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        # Executive summary text
        ax_summary = fig.add_subplot(gs[0, :2])
        ax_summary.axis('off')
        
        total_files = len(self.df)
        materials = sorted(self.df['material_type'].unique()) if 'material_type' in self.df.columns else []
        energy_range = (self.df['kinetic_energy'].min(), self.df['kinetic_energy'].max())
        avg_overestimation = self.df['overestimation_factor'].mean() if 'overestimation_factor' in self.df.columns else 1.0
        
        summary_text = f"""
📊 CORRECTED ENERGY ANALYSIS SUMMARY

Dataset: {total_files} samples analyzed
Materials: {len(materials)} types ({', '.join(materials)})
Method: Deceleration-phase isolation (excludes rebound)

⚡ Energy Results:
   Range: {energy_range[0]:.2e} - {energy_range[1]:.2e} J
   Mean: {self.df['kinetic_energy'].mean():.2e} J
   
🔄 Methodology Validation:
   Avg Overestimation Factor: {avg_overestimation:.1f}x
   Rebound Successfully Excluded
   Force Units: lbf → N converted
   
🏆 Best Material: {self.df.groupby('material_type')['kinetic_energy'].mean().idxmin() if 'material_type' in self.df.columns else 'N/A'}
💪 Max Force: {self.df['max_force'].max():.0f} N
        """
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        # Energy by material
        if 'material_type' in self.df.columns:
            ax1 = fig.add_subplot(gs[0, 2:])
            sns.boxplot(data=self.df, x='material_type', y='kinetic_energy', ax=ax1)
            ax1.set_yscale('log')
            ax1.set_title('Corrected Energy by Material', fontweight='bold')
            ax1.set_ylabel('Kinetic Energy (J)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # Force vs Energy correlation
        ax2 = fig.add_subplot(gs[1, :2])
        if 'material_type' in self.df.columns:
            materials = self.df['material_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))
            for material, color in zip(materials, colors):
                material_data = self.df[self.df['material_type'] == material]
                ax2.scatter(material_data['max_force'], material_data['kinetic_energy'], 
                           label=material, alpha=0.7, s=50, color=color)
            ax2.legend()
        else:
            ax2.scatter(self.df['max_force'], self.df['kinetic_energy'], alpha=0.6, s=50)
        
        ax2.set_xlabel('Maximum Force (N)')
        ax2.set_ylabel('Kinetic Energy (J)')
        ax2.set_title('Force-Energy Relationship', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation line
        if len(self.df) > 2:
            z = np.polyfit(self.df['max_force'], self.df['kinetic_energy'], 1)
            p = np.poly1d(z)
            ax2.plot(self.df['max_force'].sort_values(), p(self.df['max_force'].sort_values()), 
                    "r--", alpha=0.8, linewidth=2)
            
            corr = np.corrcoef(self.df['max_force'], self.df['kinetic_energy'])[0,1]
            ax2.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Overestimation analysis
        if 'overestimation_factor' in self.df.columns:
            ax3 = fig.add_subplot(gs[1, 2:])
            ax3.hist(self.df['overestimation_factor'], bins=20, alpha=0.7, 
                    edgecolor='black', color='orange')
            ax3.axvline(self.df['overestimation_factor'].mean(), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mean: {self.df["overestimation_factor"].mean():.2f}x')
            ax3.set_xlabel('Overestimation Factor')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Rebound Correction Impact', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Material performance table
        if 'material_type' in self.df.columns:
            ax4 = fig.add_subplot(gs[2, :])
            material_stats = self.df.groupby('material_type').agg({
                'kinetic_energy': ['count', 'mean', 'std'],
                'max_force': 'mean',
                'overestimation_factor': 'mean'
            }).round(4)
            
            materials_ranked = material_stats.sort_values(('kinetic_energy', 'mean'))
            
            table_data = []
            for i, (material, row) in enumerate(materials_ranked.iterrows()):
                table_data.append([
                    f"#{i+1}",
                    material,
                    f"{row[('kinetic_energy', 'count')]:.0f}",
                    f"{row[('kinetic_energy', 'mean')]:.2e}",
                    f"{row[('max_force', 'mean')]:.0f}",
                    f"{row[('overestimation_factor', 'mean')]:.2f}x"
                ])
            
            table = ax4.table(cellText=table_data,
                             colLabels=['Rank', 'Material', 'Samples', 'Mean Energy (J)', 
                                       'Avg Force (N)', 'Overest. Factor'],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax4.axis('off')
            ax4.set_title('Material Performance Ranking', fontweight='bold', pad=20)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_methodology_comparison_plot(self, output_path: Optional[Path] = None) -> plt.Figure:
        """Create before/after methodology comparison plot."""
        
        if self.df is None:
            raise ValueError("No data available")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Corrected Methodology: Before vs After Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Before vs After energy comparison
        if all(col in self.df.columns for col in ['kinetic_energy_corrected', 'kinetic_energy_overestimated']):
            energy_data = pd.melt(self.df, id_vars=['material_type'], 
                                 value_vars=['kinetic_energy_corrected', 'kinetic_energy_overestimated'],
                                 var_name='method', value_name='energy')
            energy_data['method'] = energy_data['method'].map({
                'kinetic_energy_corrected': 'Corrected\n(Decel Only)',
                'kinetic_energy_overestimated': 'Original\n(Full Curve)'
            })
            
            sns.boxplot(data=energy_data, x='material_type', y='energy', hue='method', ax=ax1)
            ax1.set_yscale('log')
            ax1.set_title('Energy Calculation: Before vs After')
            ax1.set_ylabel('Kinetic Energy (J) - Log Scale')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend(title='Method')
            ax1.grid(True, alpha=0.3)
        
        # Overestimation scatter
        if all(col in self.df.columns for col in ['kinetic_energy_corrected', 'kinetic_energy_overestimated']):
            ax2.scatter(self.df['kinetic_energy_corrected'], self.df['kinetic_energy_overestimated'], 
                       alpha=0.6, s=50)
            
            # Perfect correlation line
            min_val = min(self.df['kinetic_energy_corrected'].min(), 
                         self.df['kinetic_energy_overestimated'].min())
            max_val = max(self.df['kinetic_energy_corrected'].max(), 
                         self.df['kinetic_energy_overestimated'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                    label='Perfect correlation')
            
            ax2.set_xlabel('Corrected Energy (J)')
            ax2.set_ylabel('Overestimated Energy (J)')
            ax2.set_title('Impact of Including Rebound Phase')
            ax2.legend()
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # Deceleration detection effectiveness
        if all(col in self.df.columns for col in ['total_time', 'decel_end_time']):
            ax3.scatter(self.df['total_time'], self.df['decel_end_time'], alpha=0.6, s=50)
            ax3.plot([0, self.df['total_time'].max()], [0, self.df['total_time'].max()], 
                    'r--', alpha=0.8, label='Full duration')
            ax3.set_xlabel('Total Impact Time (s)')
            ax3.set_ylabel('Deceleration End Time (s)')
            ax3.set_title('Deceleration Phase Detection')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Improvement statistics
        if 'overestimation_factor' in self.df.columns:
            improvement_stats = {
                'Mean': self.df['overestimation_factor'].mean(),
                'Median': self.df['overestimation_factor'].median(),
                'Max': self.df['overestimation_factor'].max(),
                'Min': self.df['overestimation_factor'].min()
            }
            
            bars = ax4.bar(improvement_stats.keys(), improvement_stats.values(), 
                          color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
            ax4.set_ylabel('Overestimation Factor')
            ax4.set_title('Energy Correction Statistics')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, improvement_stats.values()):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{value:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_material_analysis_plot(self, output_path: Optional[Path] = None) -> plt.Figure:
        """Create detailed material analysis plots."""
        
        if self.df is None or 'material_type' not in self.df.columns:
            raise ValueError("No material data available")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Material Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Energy distribution
        sns.violinplot(data=self.df, x='material_type', y='kinetic_energy', ax=axes[0,0])
        axes[0,0].set_yscale('log')
        axes[0,0].set_title('Energy Distribution by Material')
        axes[0,0].set_ylabel('Kinetic Energy (J)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Force characteristics
        if 'max_force' in self.df.columns:
            sns.boxplot(data=self.df, x='material_type', y='max_force', ax=axes[0,1])
            axes[0,1].set_title('Maximum Force by Material')
            axes[0,1].set_ylabel('Maximum Force (N)')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].grid(True, alpha=0.3)
        
        # Initial velocity
        if 'initial_velocity' in self.df.columns:
            sns.boxplot(data=self.df, x='material_type', y='initial_velocity', ax=axes[0,2])
            axes[0,2].set_title('Initial Velocity by Material')
            axes[0,2].set_ylabel('Initial Velocity (m/s)')
            axes[0,2].tick_params(axis='x', rotation=45)
            axes[0,2].grid(True, alpha=0.3)
        
        # Overestimation factors
        if 'overestimation_factor' in self.df.columns:
            sns.boxplot(data=self.df, x='material_type', y='overestimation_factor', ax=axes[1,0])
            axes[1,0].set_title('Overestimation Factor by Material')
            axes[1,0].set_ylabel('Overestimation Factor')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
        
        # Deceleration characteristics
        if all(col in self.df.columns for col in ['decel_end_time', 'total_time']):
            self.df['decel_fraction'] = self.df['decel_end_time'] / self.df['total_time']
            sns.boxplot(data=self.df, x='material_type', y='decel_fraction', ax=axes[1,1])
            axes[1,1].set_title('Deceleration Phase Fraction')
            axes[1,1].set_ylabel('Deceleration Fraction')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
        
        # Sample counts
        sample_counts = self.df['material_type'].value_counts()
        bars = axes[1,2].bar(sample_counts.index, sample_counts.values)
        axes[1,2].set_title('Sample Count by Material')
        axes[1,2].set_ylabel('Number of Samples')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_correlation_analysis_plot(self, output_path: Optional[Path] = None) -> plt.Figure:
        """Create correlation analysis plot."""
        
        if self.df is None:
            raise ValueError("No data available")
        
        # Select numerical columns for correlation
        numerical_cols = [
            'kinetic_energy', 'max_force', 'initial_velocity',
            'overestimation_factor', 'decel_end_time', 'total_time'
        ]
        
        available_cols = [col for col in numerical_cols if col in self.df.columns]
        
        if len(available_cols) < 3:
            raise ValueError("Insufficient numerical columns for correlation analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Parameter Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Correlation matrix
        corr_matrix = self.df[available_cols].corr()
        
        # Create custom labels
        label_map = {
            'kinetic_energy': 'Energy',
            'max_force': 'Max Force',
            'initial_velocity': 'Velocity',
            'overestimation_factor': 'Overest. Factor',
            'decel_end_time': 'Decel Time',
            'total_time': 'Total Time'
        }
        
        display_labels = [label_map.get(col, col) for col in corr_matrix.columns]
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=display_labels, yticklabels=display_labels,
                   fmt='.3f', ax=ax1, cbar_kws={'label': 'Correlation Coefficient'})
        ax1.set_title('Parameter Correlation Matrix')
        
        # Force vs Energy scatter with material coloring
        if 'material_type' in self.df.columns:
            materials = self.df['material_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))
            
            for material, color in zip(materials, colors):
                material_data = self.df[self.df['material_type'] == material]
                ax2.scatter(material_data['max_force'], material_data['kinetic_energy'],
                           label=material, alpha=0.7, s=60, color=color)
            
            ax2.legend(title='Material')
        else:
            ax2.scatter(self.df['max_force'], self.df['kinetic_energy'], alpha=0.6, s=60)
        
        ax2.set_xlabel('Maximum Force (N)')
        ax2.set_ylabel('Kinetic Energy (J)')
        ax2.set_title('Force-Energy Relationship')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(self.df) > 2:
            corr = np.corrcoef(self.df['max_force'], self.df['kinetic_energy'])[0,1]
            ax2.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_single_file_analysis_plot(self, force_data: np.ndarray, time_data: np.ndarray,
                                       analysis_result: Dict, output_path: Optional[Path] = None) -> plt.Figure:
        """Create detailed analysis plot for a single file."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Detailed Analysis: {analysis_result.get('filename', 'Unknown')}", 
                     fontsize=16)
        
        decel_end_idx = analysis_result.get('decel_end_idx', len(force_data)//2)
        dt = time_data[1] - time_data[0] if len(time_data) > 1 else 0.00001
        cumulative_impulse = np.cumsum(force_data) * dt
        
        # Force vs Time with phases
        ax1.plot(time_data, force_data, 'b-', linewidth=1.5, label='Total Force')
        ax1.axvline(time_data[decel_end_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Decel End (t={time_data[decel_end_idx]:.4f}s)')
        ax1.fill_between(time_data[:decel_end_idx+1], force_data[:decel_end_idx+1], 
                        alpha=0.3, color='green', label='Deceleration Phase')
        ax1.fill_between(time_data[decel_end_idx:], force_data[decel_end_idx:], 
                        alpha=0.3, color='orange', label='Rebound Phase')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Force vs Time with Phase Identification')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative impulse
        ax2.plot(time_data, cumulative_impulse, 'purple', linewidth=2)
        ax2.axvline(time_data[decel_end_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Max impulse: {cumulative_impulse[decel_end_idx]:.6f} N⋅s')
        ax2.axhline(cumulative_impulse[decel_end_idx], color='r', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Cumulative Impulse (N⋅s)')
        ax2.set_title('Cumulative Impulse (finds v=0 point)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Energy comparison
        energies = [analysis_result.get('kinetic_energy_corrected', 0), 
                   analysis_result.get('kinetic_energy_overestimated', 0)]
        labels = ['Corrected\n(Decel only)', 'Overestimated\n(Full curve)']
        colors = ['green', 'red']
        
        bars = ax3.bar(labels, energies, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Kinetic Energy (J)')
        ax3.set_title('Energy Calculation Comparison')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, energy in zip(bars, energies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{energy:.6f} J', ha='center', va='bottom', fontweight='bold')
        
        # Summary statistics
        ax4.axis('off')
        summary_text = f"""
ANALYSIS SUMMARY

Material: {analysis_result.get('material_type', 'Unknown')}
Sample: {analysis_result.get('sample_number', 'Unknown')}

CORRECTED RESULTS:
Energy: {analysis_result.get('kinetic_energy', 0):.6f} J
Velocity: {analysis_result.get('initial_velocity', 0):.3f} m/s

FORCE ANALYSIS:
Max Force: {analysis_result.get('max_force', 0):.1f} N
Force Range: {analysis_result.get('force_range', 0):.1f} N

METHODOLOGY:
Overestimation Factor: {analysis_result.get('overestimation_factor', 1):.2f}x
Decel End Time: {analysis_result.get('decel_end_time', 0)*1000:.2f} ms
Decel Fraction: {analysis_result.get('decel_fraction', 0)*100:.1f}%

METHOD: Deceleration-only energy calculation
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_all_plots(self, output_dir: Union[str, Path]) -> None:
        """Create all standard plots and save to directory."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🎨 Creating visualization plots...")
        
        # Executive summary
        try:
            fig1 = self.create_executive_summary_plot()
            fig1.savefig(output_path / 'executive_summary.png', dpi=300, bbox_inches='tight')
            fig1.savefig(output_path / 'executive_summary.pdf', bbox_inches='tight')
            plt.close(fig1)
            print("  ✅ Executive summary plot created")
        except Exception as e:
            print(f"  ❌ Executive summary plot failed: {e}")
        
        # Methodology comparison
        try:
            fig2 = self.create_methodology_comparison_plot()
            fig2.savefig(output_path / 'methodology_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print("  ✅ Methodology comparison plot created")
        except Exception as e:
            print(f"  ❌ Methodology comparison plot failed: {e}")
        
        # Material analysis
        try:
            fig3 = self.create_material_analysis_plot()
            fig3.savefig(output_path / 'material_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
            print("  ✅ Material analysis plot created")
        except Exception as e:
            print(f"  ❌ Material analysis plot failed: {e}")
        
        # Correlation analysis
        try:
            fig4 = self.create_correlation_analysis_plot()
            fig4.savefig(output_path / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)
            print("  ✅ Correlation analysis plot created")
        except Exception as e:
            print(f"  ❌ Correlation analysis plot failed: {e}")
        
        print(f"📁 All plots saved to: {output_path}")


# Convenience functions for easy plotting
def create_summary_plots(results: List[Dict], output_dir: Union[str, Path]) -> None:
    """Create summary plots from analysis results."""
    visualizer = ImpactVisualizer(results)
    visualizer.create_all_plots(output_dir)


def plot_single_file_analysis(csv_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> None:
    """Create analysis plot for a single CSV file."""
    from .analysis import ImpactAnalyzer
    
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze file
    analyzer = ImpactAnalyzer()
    result = analyzer.analyze_csv_file(csv_path)
    
    if 'error' in result:
        print(f"Error analyzing {csv_path}: {result['error']}")
        return
    
    # Load data for plotting
    df = pd.read_csv(csv_path)
    force_columns = [col for col in df.columns if 'AI' in col and 'lbf' in col]
    total_force_lbf = df[force_columns].sum(axis=1).values
    force_n = total_force_lbf * 4.44822  # Convert to Newtons
    
    # Apply same corrections as analyzer
    baseline = np.median(force_n[:min(1000, len(force_n)//10)])
    force_n = force_n - baseline
    
    time = np.arange(len(force_n)) * (1/100000)  # Assume 100kHz
    
    # Create plot
    visualizer = ImpactVisualizer()
    plot_path = output_dir / f"analysis_{csv_path.stem}.png"
    visualizer.create_single_file_analysis_plot(force_n, time, result, plot_path)
    
    print(f"Analysis plot saved to: {plot_path}")


def show_force_preview(force_data: np.ndarray, time_data: np.ndarray, 
                      filename: str, show_analysis_preview: bool = False, 
                      mass: float = 0.045) -> plt.Figure:
    """
    Show a preview of force vs time data before analysis.
    
    Args:
        force_data: Force array (N)
        time_data: Time array (s)
        filename: Name of the file being previewed
        show_analysis_preview: If True, show predicted deceleration endpoint
        mass: Object mass for energy estimation
        
    Returns:
        Matplotlib figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Force Data Preview: {filename}', fontsize=16, fontweight='bold')
    
    # Main force vs time plot
    axes[0,0].plot(time_data, force_data, 'b-', linewidth=1.5, alpha=0.8)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Force (N)')
    axes[0,0].set_title('Force vs Time (After Processing)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add statistics text
    max_force = np.max(force_data)
    min_force = np.min(force_data)
    rms_force = np.sqrt(np.mean(force_data**2))
    
    stats_text = f"""
DATA STATISTICS:
Max Force: {max_force:.1f} N
Min Force: {min_force:.1f} N
RMS Force: {rms_force:.1f} N
Range: {max_force - min_force:.1f} N
Duration: {time_data[-1]:.4f} s
Data Points: {len(force_data)}
    """
    
    axes[0,0].text(0.02, 0.98, stats_text.strip(), transform=axes[0,0].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Force histogram
    axes[0,1].hist(force_data, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0,1].axvline(np.mean(force_data), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(force_data):.1f} N')
    axes[0,1].axvline(np.median(force_data), color='green', linestyle='--', 
                     label=f'Median: {np.median(force_data):.1f} N')
    axes[0,1].set_xlabel('Force (N)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Force Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Show analysis preview if requested
    if show_analysis_preview:
        # Calculate cumulative impulse to show deceleration detection
        dt = time_data[1] - time_data[0] if len(time_data) > 1 else 0.00001
        cumulative_impulse = np.cumsum(force_data) * dt
        decel_end_idx = np.argmax(cumulative_impulse)
        
        # Force with deceleration prediction
        axes[1,0].plot(time_data, force_data, 'b-', linewidth=1, alpha=0.7, label='Force')
        axes[1,0].axvline(time_data[decel_end_idx], color='red', linestyle='--', linewidth=2,
                         label=f'Predicted decel end: {time_data[decel_end_idx]*1000:.1f} ms')
        axes[1,0].fill_between(time_data[:decel_end_idx+1], force_data[:decel_end_idx+1], 
                              alpha=0.3, color='green', label='Deceleration phase')
        axes[1,0].fill_between(time_data[decel_end_idx:], force_data[decel_end_idx:], 
                              alpha=0.3, color='orange', label='Rebound phase')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Force (N)')
        axes[1,0].set_title('Predicted Analysis Phases')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Cumulative impulse
        axes[1,1].plot(time_data, cumulative_impulse, 'purple', linewidth=2)
        axes[1,1].axvline(time_data[decel_end_idx], color='red', linestyle='--', 
                         label=f'Max impulse: {cumulative_impulse[decel_end_idx]:.6f} N⋅s')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Cumulative Impulse (N⋅s)')
        axes[1,1].set_title('Impulse Accumulation (finds v=0)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Quick energy estimate
        force_decel = force_data[:decel_end_idx + 1]
        J_decel = np.trapz(force_decel, dx=dt)
        energy_estimate = 0.5 * mass * (J_decel / mass)**2
        
        preview_text = f"""
ANALYSIS PREVIEW:
Decel End: {time_data[decel_end_idx]*1000:.1f} ms
Decel Fraction: {decel_end_idx/len(force_data)*100:.1f}%
Est. Energy: {energy_estimate:.6f} J
Decel Impulse: {J_decel:.6f} N⋅s
        """
        
        axes[1,1].text(0.02, 0.98, preview_text.strip(), transform=axes[1,1].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    else:
        # Signal quality analysis
        # Moving average to show trends
        window_size = max(1, len(force_data) // 100)
        if window_size > 1:
            moving_avg = np.convolve(force_data, np.ones(window_size)/window_size, mode='same')
            axes[1,0].plot(time_data, force_data, 'b-', alpha=0.5, linewidth=0.5, label='Raw')
            axes[1,0].plot(time_data, moving_avg, 'r-', linewidth=2, label=f'Moving avg ({window_size} pts)')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Force (N)')
            axes[1,0].set_title('Signal Quality Check')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Power spectral density (if data is long enough)
        if len(force_data) > 256:
            try:
                from scipy import signal as scipy_signal
                freqs, psd = scipy_signal.welch(force_data, fs=1/dt, nperseg=min(256, len(force_data)//4))
                axes[1,1].semilogy(freqs, psd)
                axes[1,1].set_xlabel('Frequency (Hz)')
                axes[1,1].set_ylabel('Power Spectral Density')
                axes[1,1].set_title('Frequency Content')
                axes[1,1].grid(True, alpha=0.3)
            except ImportError:
                # Fallback if scipy not available
                force_diff = np.diff(force_data)
                axes[1,1].plot(time_data[:-1], force_diff, 'g-', linewidth=1)
                axes[1,1].set_xlabel('Time (s)')
                axes[1,1].set_ylabel('Force Rate (N/s)')
                axes[1,1].set_title('Force Rate of Change')
                axes[1,1].grid(True, alpha=0.3)
        else:
            # Simple force derivative if not enough data for PSD
            force_diff = np.diff(force_data)
            axes[1,1].plot(time_data[:-1], force_diff, 'g-', linewidth=1)
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Force Rate (N/s)')
            axes[1,1].set_title('Force Rate of Change')
            axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def quick_energy_check_plot(results: List[Dict], output_path: Optional[Path] = None) -> plt.Figure:
    """Create quick energy reasonableness check plot."""
    
    if not results:
        raise ValueError("No results provided")
    
    valid_results = [r for r in results if 'kinetic_energy' in r and 
                    not pd.isna(r['kinetic_energy']) and r['kinetic_energy'] > 0]
    
    if not valid_results:
        raise ValueError("No valid energy results")
    
    energies = [r['kinetic_energy'] for r in valid_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Energy Reasonableness Check', fontsize=16, fontweight='bold')
    
    # Energy distribution histogram
    ax1.hist(energies, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_xlabel('Kinetic Energy (J)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Energy Distribution')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add reference lines for reasonable ranges
    ax1.axvline(1e-3, color='green', linestyle='--', alpha=0.7, label='1 mJ (low)')
    ax1.axvline(1e-1, color='orange', linestyle='--', alpha=0.7, label='100 mJ (moderate)')
    ax1.axvline(1e1, color='red', linestyle='--', alpha=0.7, label='10 J (high)')
    ax1.legend()
    
    # Energy by material with reasonableness coloring
    if len(valid_results) > 0 and 'material_type' in valid_results[0]:
        df = pd.DataFrame(valid_results)
        
        # Color code by energy range
        def get_energy_color(energy):
            if energy < 1e-1:
                return 'green'  # Realistic
            elif energy < 1e1:
                return 'orange'  # Moderate
            else:
                return 'red'    # High
        
        materials = sorted(df['material_type'].unique())
        for i, material in enumerate(materials):
            material_data = df[df['material_type'] == material]
            colors = [get_energy_color(e) for e in material_data['kinetic_energy']]
            
            ax2.scatter([i] * len(material_data), material_data['kinetic_energy'], 
                       c=colors, alpha=0.7, s=50)
        
        ax2.set_xticks(range(len(materials)))
        ax2.set_xticklabels(materials, rotation=45)
        ax2.set_ylabel('Kinetic Energy (J)')
        ax2.set_title('Energy by Material (Color = Reasonableness)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Realistic (< 100 mJ)'),
            Patch(facecolor='orange', alpha=0.7, label='Moderate (100 mJ - 10 J)'),
            Patch(facecolor='red', alpha=0.7, label='High (> 10 J)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


# Legacy compatibility functions for backward compatibility
def create_material_summary_plots(results: List[Dict], output_dir: Union[str, Path]) -> None:
    """Legacy compatibility function."""
    create_summary_plots(results, output_dir)


def create_comparison_plots(results: List[Dict], output_dir: Union[str, Path]) -> None:
    """Legacy compatibility function."""
    create_summary_plots(results, output_dir)


def create_individual_analysis_plot(force_data: np.ndarray, time_data: np.ndarray, 
                                   analysis_result: Dict, output_path: Union[str, Path]) -> None:
    """Legacy compatibility function for individual file plots."""
    visualizer = ImpactVisualizer()
    fig = visualizer.create_single_file_analysis_plot(force_data, time_data, analysis_result, Path(output_path))
    plt.close(fig)


# Additional utility functions
def create_quick_summary(results: List[Dict]) -> str:
    """Create a quick text summary of results."""
    
    if not results:
        return "No results to summarize."
    
    valid_results = [r for r in results if 'kinetic_energy' in r and 
                    not pd.isna(r['kinetic_energy']) and r['kinetic_energy'] > 0]
    
    if not valid_results:
        return "No valid energy results to summarize."
    
    energies = [r['kinetic_energy'] for r in valid_results]
    
    summary = f"""
QUICK ANALYSIS SUMMARY
{'=' * 30}
Total files: {len(results)}
Valid results: {len(valid_results)}
Success rate: {len(valid_results)/len(results)*100:.1f}%

ENERGY STATISTICS:
Min energy: {np.min(energies):.6f} J
Max energy: {np.max(energies):.6f} J  
Mean energy: {np.mean(energies):.6f} J
Median energy: {np.median(energies):.6f} J

ENERGY REASONABLENESS:
< 100 mJ (realistic): {sum(1 for e in energies if e < 0.1)} files
100 mJ - 10 J (moderate): {sum(1 for e in energies if 0.1 <= e < 10)} files
> 10 J (high): {sum(1 for e in energies if e >= 10)} files
    """
    
    # Add material breakdown if available
    if valid_results and 'material_type' in valid_results[0]:
        df = pd.DataFrame(valid_results)
        material_stats = df.groupby('material_type')['kinetic_energy'].agg(['count', 'mean']).round(6)
        
        summary += f"\nMATERIAL BREAKDOWN:\n"
        for material, stats in material_stats.iterrows():
            summary += f"{material}: {stats['count']} samples, avg {stats['mean']:.6f} J\n"
    
    return summary.strip()


if __name__ == "__main__":
    # Example usage
    print("Fishing Line Flyback Impact Analysis - Visualization Module")
    print("This module provides comprehensive plotting capabilities for analysis results.")
    print("\nMain functions:")
    print("- ImpactVisualizer: Main visualization class")
    print("- create_summary_plots: Create all standard plots")
    print("- plot_single_file_analysis: Plot individual file analysis")
    print("\nFor usage examples, see the package documentation.")
