"""
Fishing Line Flyback Impact Analysis - Main CLI Interface

Enhanced version with configuration-specific weights, comprehensive analysis,
and NEW impulse-based analysis.
"""

import click
import numpy as np
import pandas as pd
from pathlib import Path

# Import main analysis functions
from .analysis import (
    ImpactAnalyzer, 
    analyze_single_file_with_config, 
    run_comprehensive_analysis,
    CONFIG_WEIGHTS,
    LINE_MASS_FRACTION,
    MEASURED_LINE_LENGTH_INCHES,
    MEASURED_LINE_MASS_GRAMS
)

# Import impulse analysis functions
from .impulse_analysis import (
    run_impulse_analysis, 
    analyze_single_file_with_impulse,
    create_method_comparison_plots
)

# Import windowing tools
from .windowing import (
    interactive_window_tool,
    batch_window_files,
    review_window_suggestions,
    create_review_report
)



@click.group()
@click.version_option(version="2.1.0")
def main():
    """
    🎣 Fishing Line Flyback Impact Analysis v2.1
    
    Enhanced analysis with configuration-specific weights, measured line mass,
    and NEW impulse-based analysis method.
    
    Key Features:
    ✓ Configuration-specific weights (STND=45g, DF=60g, DS=72g, SL=69g, BR=45g)
    ✓ Measured line mass integration (5.5" = 0.542g → 38.8g total)
    ✓ 70% effective line mass (literature validated)
    ✓ Peak-focused impact detection for realistic velocities
    ✓ Target velocity range: 150-400 m/s
    ✓ Interactive data windowing tools
    ✓ NEW: Impulse analysis (∫ F(t) dt) for direct momentum transfer measurement
    """
    pass


# =============================================================================
# KINETIC ENERGY ANALYSIS COMMANDS
# =============================================================================

@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='comprehensive_analysis', 
              help='Output directory for results and plots')
@click.option('--show-progress', is_flag=True, default=True,
              help='Show real-time analysis progress')
def analyze_all(data_dir, output_dir, show_progress):
    """
    🚀 Run comprehensive kinetic energy analysis on all measurement files.
    
    This is the main kinetic energy analysis function that processes all your CSV files with:
    - Configuration-specific weights for each material type
    - Measured line mass (70% effective)
    - Peak-focused energy calculation
    - Comprehensive reporting and visualization
    
    DATA_DIR: Directory containing CSV measurement files
    """
    click.echo("🎣 COMPREHENSIVE FISHING LINE FLYBACK ANALYSIS")
    click.echo("=" * 60)
    click.echo(f"📁 Data directory: {data_dir}")
    click.echo(f"📊 Output directory: {output_dir}")
    click.echo()
    
    click.echo("⚖️  CONFIGURATION WEIGHTS:")
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        line_mass_effective = 0.0388 * LINE_MASS_FRACTION
        total_mass = weight_kg + line_mass_effective
        click.echo(f"   {config:4s}: {weight_kg*1000:2.0f}g + {line_mass_effective*1000:.0f}g = {total_mass*1000:.0f}g total")
    click.echo()
    
    click.echo("📚 SCIENTIFIC BASIS:")
    click.echo(f"   Line measurement: {MEASURED_LINE_LENGTH_INCHES}\" = {MEASURED_LINE_MASS_GRAMS}g")
    click.echo(f"   Effective mass: {LINE_MASS_FRACTION*100:.0f}% (literature: Cartmell & McKenzie 2008)")
    click.echo(f"   Target velocity: 150-400 m/s (Steffens & Nettleton 2019)")
    click.echo()
    
    try:
        results = run_comprehensive_analysis(data_dir, output_dir)
        
        if results:
            valid_results = [r for r in results if 'error' not in r]
            velocities = [abs(r['initial_velocity']) for r in valid_results]
            target_count = sum(1 for v in velocities if 150 <= v <= 400)
            
            click.echo()
            click.echo("🎉 ANALYSIS COMPLETE!")
            click.echo(f"✅ Successfully analyzed: {len(valid_results)}/{len(results)} files")
            click.echo(f"🎯 Target velocity range: {target_count}/{len(valid_results)} ({target_count/len(valid_results)*100:.1f}%)")
            click.echo(f"📁 Results saved to: {output_dir}")
        else:
            click.echo("❌ No files were successfully analyzed")
            
    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--material', '-m', type=click.Choice(['STND', 'DF', 'DS', 'SL', 'BR']),
              help='Material configuration code (auto-detected if not specified)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for results (optional)')
@click.option('--debug', is_flag=True, help='Show detailed debug information')
@click.option('--legacy-mass', type=float, help='Use legacy mode with manual mass (kg)')
def analyze_single(file_path, material, output, debug, legacy_mass):
    """
    🔍 Analyze a single measurement file using kinetic energy method.
    
    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"🔍 Analyzing: {file_path.name}")
    click.echo("-" * 40)
    
    try:
        if legacy_mass is not None:
            # Legacy mode with manual mass
            analyzer = ImpactAnalyzer(mass=legacy_mass)
            click.echo(f"⚖️  Using legacy mode: {legacy_mass*1000:.1f}g total mass")
            result = analyzer.analyze_csv_file(file_path)
        else:
            # New mode with configuration detection
            if material is None:
                material = file_path.name.split('-')[0]
                click.echo(f"🔍 Auto-detected material: {material}")
            
            click.echo(f"⚖️  Configuration: {material}")
            config_weight = CONFIG_WEIGHTS.get(material, 0.045)
            line_mass_effective = 0.0388 * LINE_MASS_FRACTION
            total_mass = config_weight + line_mass_effective
            click.echo(f"   Hardware: {config_weight*1000:.0f}g")
            click.echo(f"   Line (effective): {line_mass_effective*1000:.0f}g ({LINE_MASS_FRACTION*100:.0f}%)")
            click.echo(f"   Total: {total_mass*1000:.0f}g")
            click.echo()
            
            result = analyze_single_file_with_config(
                file_path, 
                material_code=material,
                include_line_mass=True,
                line_mass_fraction=LINE_MASS_FRACTION
            )
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}")
            return
        
        # Display results
        velocity = abs(result['initial_velocity'])
        energy = result['kinetic_energy']
        
        # Status based on velocity
        if 150 <= velocity <= 400:
            status = "✅ EXCELLENT"
        elif 100 <= velocity <= 500:
            status = "🟡 GOOD"
        else:
            status = "❌ REVIEW"
        
        click.echo("📊 RESULTS:")
        click.echo(f"   Velocity: {velocity:.0f} m/s {status}")
        click.echo(f"   Energy: {energy:.2f} J")
        click.echo(f"   Max Force: {result.get('max_force', 0):.0f} N")
        click.echo(f"   Impact Duration: {result.get('impact_duration', 0)*1000:.1f} ms")
        
        if debug:
            click.echo(f"\n🔧 DEBUG INFO:")
            click.echo(f"   Peak impulse: {result.get('impulse_decel', 0):.6f} N⋅s")
            click.echo(f"   Total impulse: {result.get('impulse_total', 0):.6f} N⋅s")
            click.echo(f"   Overestimation factor: {result.get('overestimation_factor', 1):.2f}x")
            click.echo(f"   Sampling rate: {result.get('sampling_rate_hz', 0):.0f} Hz")
            click.echo(f"   Data points used: {result.get('data_points', 0)}")
        
        # Validation against literature
        click.echo(f"\n📚 LITERATURE VALIDATION:")
        if 200 <= velocity <= 350:
            click.echo("   ✅ Velocity in ideal literature range (200-350 m/s)")
        elif 150 <= velocity <= 400:
            click.echo("   🟡 Velocity in acceptable range (150-400 m/s)")
        else:
            click.echo("   ❌ Velocity outside expected range")
        
        # Save results if requested
        if output:
            import json
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"\n💾 Results saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.ClickException(str(e))


# =============================================================================
# IMPULSE ANALYSIS COMMANDS (NEW)
# =============================================================================

@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='impulse_analysis', 
              help='Output directory for impulse analysis results')
@click.option('--show-progress', is_flag=True, default=True,
              help='Show real-time analysis progress')
def analyze_impulse(data_dir, output_dir, show_progress):
    """
    🎯 Run impulse-based analysis on all measurement files.
    
    This method focuses on total momentum transfer (∫ F(t) dt) rather than
    trying to estimate kinetic energy. More direct and meaningful for 
    fishing line impact effectiveness comparison.
    
    DATA_DIR: Directory containing CSV measurement files
    """
    click.echo("🎯 IMPULSE-BASED FISHING LINE ANALYSIS")
    click.echo("=" * 60)
    click.echo(f"📁 Data directory: {data_dir}")
    click.echo(f"📊 Output directory: {output_dir}")
    click.echo(f"🔬 Method: Total momentum transfer via ∫ F(t) dt")
    click.echo()
    
    try:
        results = run_impulse_analysis(data_dir, output_dir)
        
        if results:
            valid_results = [r for r in results if 'error' not in r]
            impulses = [abs(r['total_impulse']) for r in valid_results]
            
            click.echo(f"🎉 IMPULSE ANALYSIS COMPLETE!")
            click.echo(f"✅ Successfully analyzed: {len(valid_results)}/{len(results)} files")
            click.echo(f"📊 Impulse range: {min(impulses):.6f} to {max(impulses):.6f} N⋅s")
            click.echo(f"📁 Results saved to: {output_dir}")
        else:
            click.echo("❌ No files were successfully analyzed")
            
    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--material', '-m', type=click.Choice(['STND', 'DF', 'DS', 'SL', 'BR']),
              help='Material configuration code (auto-detected if not specified)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for results (optional)')
@click.option('--debug', is_flag=True, help='Show detailed debug information')
def analyze_impulse_single(file_path, material, output, debug):
    """
    🎯 Analyze single file using impulse method.
    
    Focus on total momentum transfer rather than kinetic energy estimation.
    
    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"🎯 Impulse Analysis: {file_path.name}")
    click.echo("-" * 40)
    
    try:
        if material is None:
            material = file_path.name.split('-')[0]
            click.echo(f"🔍 Auto-detected material: {material}")
        
        result = analyze_single_file_with_impulse(
            file_path, 
            material_code=material,
            include_line_mass=True,
            line_mass_fraction=LINE_MASS_FRACTION
        )
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}")
            return
        
        # Display results
        impulse = result['total_impulse']
        abs_impulse = result['total_abs_impulse']
        peak_force = result['peak_force']
        duration = result['impact_duration'] * 1000  # ms
        
        click.echo("📊 IMPULSE ANALYSIS RESULTS:")
        click.echo(f"   Total impulse: {impulse:+.6f} N⋅s")
        click.echo(f"   Absolute impulse: {abs_impulse:.6f} N⋅s")
        click.echo(f"   Peak force: {peak_force:.0f} N")
        click.echo(f"   Impact duration: {duration:.1f} ms")
        
        # Interpretation
        if impulse > 0:
            click.echo(f"   → Forward momentum transfer (typical impact)")
        else:
            click.echo(f"   → Backward momentum transfer (strong rebound)")
        
        click.echo(f"   → Momentum magnitude: {abs(impulse):.6f} N⋅s")
        
        # For comparison with kinetic energy method
        if 'equivalent_velocity' in result and not np.isnan(result['equivalent_velocity']):
            equiv_v = result['equivalent_velocity']
            equiv_ke = result['equivalent_kinetic_energy']
            click.echo(f"\n📊 EQUIVALENT METRICS (for comparison):")
            click.echo(f"   Equivalent velocity: {equiv_v:.0f} m/s")
            click.echo(f"   Equivalent kinetic energy: {equiv_ke:.3f} J")
        
        if debug:
            click.echo(f"\n🔧 DEBUG INFO:")
            click.echo(f"   Material: {result['material_type']}")
            click.echo(f"   Mass: {result['mass_kg']*1000:.1f}g")
            click.echo(f"   Sampling rate: {result['sampling_rate_hz']:.0f} Hz")
            click.echo(f"   Impact start: {result['impact_start_time']:.4f} s")
            click.echo(f"   Impact end: {result['impact_end_time']:.4f} s")
        
        # Save results if requested
        if output:
            import json
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\n💾 Results saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='method_comparison',
              help='Output directory for comparison results')
def compare_methods(data_dir, output_dir):
    """
    ⚖️  Compare kinetic energy vs impulse analysis methods.
    
    Runs both analysis methods on the same data and creates
    comparative visualizations and statistics.
    
    DATA_DIR: Directory containing CSV files
    """
    click.echo("⚖️  KINETIC ENERGY vs IMPULSE METHOD COMPARISON")
    click.echo("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run both analyses
        click.echo("🔄 Running kinetic energy analysis...")
        ke_results = run_comprehensive_analysis(data_dir, output_path / "kinetic_energy")
        
        click.echo("🔄 Running impulse analysis...")
        impulse_results = run_impulse_analysis(data_dir, output_path / "impulse")
        
        # Create comparison plots
        click.echo("🔄 Creating comparison plots...")
        create_method_comparison_plots(ke_results, impulse_results, output_path)
        
        click.echo(f"🎉 METHOD COMPARISON COMPLETE!")
        click.echo(f"📁 Results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Error during comparison: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--count', '-n', default=5, help='Number of files to check')
def quick_impulse_check(data_dir, count):
    """
    ⚡ Quick impulse check of multiple files (first N files).
    
    DATA_DIR: Directory containing CSV files
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))[:count]
    
    click.echo(f"⚡ QUICK IMPULSE CHECK ({len(csv_files)} files)")
    click.echo("-" * 60)
    
    success_count = 0
    
    for file_path in csv_files:
        try:
            material = file_path.name.split('-')[0]
            result = analyze_single_file_with_impulse(file_path, material_code=material)
            
            if 'error' not in result:
                impulse = result['total_impulse']
                abs_impulse = result['total_abs_impulse']
                peak_force = result['peak_force']
                
                direction = "→" if impulse > 0 else "←"
                click.echo(f"✅ {file_path.name:<15} | {material:<4} | {impulse:>+10.6f} N⋅s {direction} | "
                          f"{abs_impulse:>8.6f} | {peak_force:>6.0f} N")
                success_count += 1
            else:
                click.echo(f"❌ {file_path.name:<15} | ERROR: {result['error']}")
                
        except Exception as e:
            click.echo(f"❌ {file_path.name:<15} | ERROR: {e}")
    
    if success_count > 0:
        click.echo("-" * 60)
        click.echo(f"📊 Summary: {success_count}/{len(csv_files)} successful")
        click.echo(f"Legend: → = forward momentum, ← = backward momentum (rebound)")


# =============================================================================
# UTILITY COMMANDS
# =============================================================================

@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--material', '-m', type=click.Choice(['STND', 'DF', 'DS', 'SL', 'BR']),
              help='Material configuration code')
def quick_check(file_path, material):
    """
    ⚡ Quick check of a measurement file (minimal output).
    
    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    
    try:
        if material is None:
            material = file_path.name.split('-')[0]
        
        result = analyze_single_file_with_config(file_path, material_code=material)
        
        if 'error' in result:
            click.echo(f"❌ {file_path.name}: {result['error']}")
            return
        
        velocity = abs(result['initial_velocity'])
        energy = result['kinetic_energy']
        
        if 150 <= velocity <= 400:
            status = "✅"
        elif 100 <= velocity <= 500:
            status = "🟡"
        else:
            status = "❌"
        
        click.echo(f"{status} {file_path.name}: {velocity:.0f} m/s, {energy:.1f} J")
        
    except Exception as e:
        click.echo(f"❌ {file_path.name}: {e}")


@main.command()
def show_config():
    """
    ⚙️ Show current configuration and mass settings.
    """
    click.echo("⚙️ CONFIGURATION SETTINGS")
    click.echo("=" * 40)
    
    click.echo("📏 LINE MASS MEASUREMENT:")
    click.echo(f"   Sample length: {MEASURED_LINE_LENGTH_INCHES}\"")
    click.echo(f"   Sample mass: {MEASURED_LINE_MASS_GRAMS}g")
    click.echo(f"   Linear density: {MEASURED_LINE_MASS_GRAMS/(MEASURED_LINE_LENGTH_INCHES*0.0254):.1f} g/m")
    click.echo(f"   10m line mass: {38.8:.1f}g")
    click.echo(f"   Effective fraction: {LINE_MASS_FRACTION*100:.0f}%")
    click.echo(f"   Effective mass: {38.8 * LINE_MASS_FRACTION:.1f}g")
    click.echo()
    
    click.echo("⚖️  CONFIGURATION WEIGHTS:")
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        line_mass_effective = 0.0388 * LINE_MASS_FRACTION
        total_mass = weight_kg + line_mass_effective
        click.echo(f"   {config:4s}: {weight_kg*1000:2.0f}g hardware + {line_mass_effective*1000:.0f}g line = {total_mass*1000:.0f}g total")
    click.echo()
    
    click.echo("🎯 ANALYSIS METHOD COMPARISON:")
    click.echo("   KINETIC ENERGY METHOD:")
    click.echo("     • Target velocity: 150-400 m/s")
    click.echo("     • Expected energy: 1-200 J")
    click.echo("     • Focus: Initial projectile energy")
    click.echo("     • Complexity: High (phase separation needed)")
    click.echo()
    click.echo("   IMPULSE METHOD:")
    click.echo("     • Primary metric: Total impulse (N⋅s)")
    click.echo("     • Physical meaning: Momentum transfer")
    click.echo("     • Focus: Impact effectiveness")
    click.echo("     • Complexity: Low (simple integration)")
    click.echo()
    
    click.echo("📚 SCIENTIFIC REFERENCES:")
    click.echo("   Line mass fraction: Cartmell & McKenzie (2008), Irvine (1981)")
    click.echo("   Fishing line impacts: Steffens & Nettleton (2019)")
    click.echo("   Distributed mass theory: Thomson & Dahleh (1998)")
    click.echo("   High-speed impacts: Zukas (1990)")
    click.echo("   Impulse-momentum theorem: Classical mechanics")


@main.command()
def info():
    """
    ℹ️  Show package information and version details.
    """
    click.echo("🎣 FISHING LINE FLYBACK IMPACT ANALYSIS")
    click.echo("=" * 50)
    click.echo("Version: 2.1.0")
    click.echo("Enhanced with impulse-based analysis")
    click.echo()
    
    click.echo("🔬 ANALYSIS METHODS:")
    click.echo("1. KINETIC ENERGY ANALYSIS:")
    click.echo("   • Peak-focused impact detection")
    click.echo("   • Configuration-specific hardware masses")
    click.echo("   • Measured line mass integration (70% effective)")
    click.echo("   • Realistic velocity targeting (150-400 m/s)")
    click.echo("   • Focus: Estimate initial projectile energy")
    click.echo()
    click.echo("2. IMPULSE ANALYSIS (NEW):")
    click.echo("   • Total momentum transfer: ∫ F(t) dt")
    click.echo("   • Direct measurement of impact effectiveness")
    click.echo("   • Simple integration of complete force curve")
    click.echo("   • More relevant to fishing line performance")
    click.echo("   • Focus: What the fish/lure actually experiences")
    click.echo()
    
    click.echo("📊 SUPPORTED CONFIGURATIONS:")
    for config, weight_kg in sorted(CONFIG_WEIGHTS.items()):
        descriptions = {
            'STND': 'Standard',
            'DF': 'Dual Fixed',
            'DS': 'Dual Sliding', 
            'SL': 'Sliding',
            'BR': 'Breakaway'
        }
        desc = descriptions.get(config, 'Unknown')
        click.echo(f"• {config}: {desc} ({weight_kg*1000:.0f}g)")
    click.echo()
    
    click.echo("⚡ QUICK START - IMPULSE ANALYSIS (RECOMMENDED):")
    click.echo("# Single file:")
    click.echo("poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse-single data/csv/STND-21-5.csv --debug")
    click.echo()
    click.echo("# All files:")
    click.echo("poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-impulse data/csv")
    click.echo()
    click.echo("# Compare methods:")
    click.echo("poetry run python -m Fishing_Line_Flyback_Impact_Analysis compare-methods data/csv")
    click.echo()
    
    click.echo("🎯 CLASSIC WORKFLOW (KINETIC ENERGY):")
    click.echo("# Analyze all files:")
    click.echo("poetry run python -m Fishing_Line_Flyback_Impact_Analysis analyze-all data/csv")
    click.echo()
    
    click.echo("💡 RECOMMENDATION:")
    click.echo("Start with impulse analysis - it's simpler, more reliable,")
    click.echo("and more directly relevant to fishing line performance!")


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--count', '-n', default=5, help='Number of files to check')
def quick_batch_check(data_dir, count):
    """
    ⚡ Quick check of multiple files (first N files).
    
    DATA_DIR: Directory containing CSV files
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))[:count]
    
    click.echo(f"⚡ QUICK BATCH CHECK ({len(csv_files)} files)")
    click.echo("-" * 60)
    
    success_count = 0
    target_count = 0
    
    for file_path in csv_files:
        try:
            material = file_path.name.split('-')[0]
            result = analyze_single_file_with_config(file_path, material_code=material)
            
            if 'error' not in result:
                velocity = abs(result['initial_velocity'])
                energy = result['kinetic_energy']
                
                if 150 <= velocity <= 400:
                    status = "✅"
                    target_count += 1
                elif 100 <= velocity <= 500:
                    status = "🟡"
                else:
                    status = "❌"
                
                click.echo(f"{status} {file_path.name:<15} | {material:<4} | {velocity:3.0f} m/s | {energy:6.1f} J")
                success_count += 1
            else:
                click.echo(f"❌ {file_path.name:<15} | ERROR: {result['error']}")
                
        except Exception as e:
            click.echo(f"❌ {file_path.name:<15} | ERROR: {e}")
    
    if success_count > 0:
        click.echo("-" * 60)
        click.echo(f"📊 Summary: {success_count}/{len(csv_files)} successful")
        click.echo(f"🎯 Target range: {target_count}/{success_count} ({target_count/success_count*100:.1f}%)")

@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--material', '-m', type=click.Choice(['STND', 'DF', 'DS', 'SL', 'BR']),
              help='Material configuration code (auto-detected if not specified)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for results (optional)')
@click.option('--debug', is_flag=True, help='Show detailed debug information')
@click.option('--show-plot', is_flag=True, help='Show boundary validation plot')
def analyze_impulse_single(file_path, material, output, debug, show_plot):
    """
    🎯 Analyze single file using impulse method with optional visualization.
    
    Focus on total momentum transfer rather than kinetic energy estimation.
    Use --show-plot to validate integration boundaries.
    
    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"🎯 Impulse Analysis: {file_path.name}")
    click.echo("-" * 40)
    
    try:
        if material is None:
            material = file_path.name.split('-')[0]
            click.echo(f"🔍 Auto-detected material: {material}")
        
        # Use enhanced analyzer with show_plot option
        result = analyze_single_file_with_enhanced_impulse(
            file_path, 
            material_code=material,
            include_line_mass=True,
            line_mass_fraction=LINE_MASS_FRACTION,
            show_plot=show_plot
        )
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}")
            return
        
        # Display results
        impulse = result['total_impulse']
        abs_impulse = result['total_abs_impulse']
        peak_force = result['peak_force']
        duration = result['impact_duration'] * 1000  # ms
        
        click.echo("📊 IMPULSE ANALYSIS RESULTS:")
        click.echo(f"   Total impulse: {impulse:+.6f} N⋅s")
        click.echo(f"   Absolute impulse: {abs_impulse:.6f} N⋅s")
        click.echo(f"   Peak force: {peak_force:.0f} N")
        click.echo(f"   Impact duration: {duration:.1f} ms")
        
        # Interpretation
        if impulse > 0:
            click.echo(f"   → Forward momentum transfer (typical impact)")
        else:
            click.echo(f"   → Backward momentum transfer (strong rebound)")
        
        click.echo(f"   → Momentum magnitude: {abs(impulse):.6f} N⋅s")
        
        # For comparison with kinetic energy method
        if 'equivalent_velocity' in result and not np.isnan(result['equivalent_velocity']):
            equiv_v = result['equivalent_velocity']
            equiv_ke = result['equivalent_kinetic_energy']
            click.echo(f"\n📊 EQUIVALENT METRICS (for comparison):")
            click.echo(f"   Equivalent velocity: {equiv_v:.0f} m/s")
            click.echo(f"   Equivalent kinetic energy: {equiv_ke:.3f} J")
        
        if debug:
            click.echo(f"\n🔧 DEBUG INFO:")
            click.echo(f"   Material: {result['material_type']}")
            click.echo(f"   Mass: {result['mass_kg']*1000:.1f}g")
            click.echo(f"   Sampling rate: {result['sampling_rate_hz']:.0f} Hz")
            click.echo(f"   Impact start: {result['impact_start_time']:.4f} s")
            click.echo(f"   Impact end: {result['impact_end_time']:.4f} s")
        
        if show_plot:
            click.echo(f"\n📊 BOUNDARY VALIDATION:")
            click.echo(f"   Plot displayed showing integration boundaries")
            click.echo(f"   Check that the highlighted region captures the main impact")
            click.echo(f"   Duration should typically be 2-50 ms for fishing line impacts")
        
        # Save results if requested
        if output:
            import json
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\n💾 Results saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.ClickException(str(e))


# Alternative: Quick validation command for problematic files
@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--material', '-m', type=click.Choice(['STND', 'DF', 'DS', 'SL', 'BR']),
              help='Material configuration code (auto-detected if not specified)')
def validate_boundaries(file_path, material):
    """
    🔍 Quick boundary validation with visualization (always shows plot).
    
    Use this command when you need to verify integration boundaries
    for files that show suspicious results.
    
    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"🔍 Boundary Validation: {file_path.name}")
    click.echo("-" * 50)
    
    try:
        if material is None:
            material = file_path.name.split('-')[0]
            click.echo(f"🔍 Auto-detected material: {material}")
        
        # Always show plot for validation
        result = analyze_single_file_with_enhanced_impulse(
            file_path, 
            material_code=material,
            include_line_mass=True,
            line_mass_fraction=LINE_MASS_FRACTION,
            show_plot=True  # Always True for validation
        )
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}")
            return
        
        # Quick summary
        duration = result['impact_duration'] * 1000
        click.echo(f"\n⚡ QUICK VALIDATION RESULTS:")
        click.echo(f"   Impact duration: {duration:.1f} ms")
        click.echo(f"   Total impulse: {result['total_impulse']:+.6f} N⋅s")
        click.echo(f"   Peak force: {result['peak_force']:.0f} N")
        
        # Validation guidance
        click.echo(f"\n💡 VALIDATION GUIDANCE:")
        if 2 <= duration <= 50:
            click.echo(f"   ✅ Duration looks reasonable ({duration:.1f} ms)")
        elif duration < 2:
            click.echo(f"   ⚠️  Duration very short ({duration:.1f} ms) - may miss impact data")
        elif duration > 100:
            click.echo(f"   ❌ Duration very long ({duration:.1f} ms) - likely includes rebound/noise")
        else:
            click.echo(f"   🟡 Duration borderline ({duration:.1f} ms) - review plot carefully")
        
        click.echo(f"\n📊 PLOT INTERPRETATION:")
        click.echo(f"   • Top plot shows all threshold options tried")
        click.echo(f"   • Bottom plot shows selected boundary in detail")
        click.echo(f"   • Green shaded area = integration region")
        click.echo(f"   • Red dashed lines = start/end boundaries")
        click.echo(f"   • Orange dotted line = peak force location")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.ClickException(str(e))
            
@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--material', '-m', type=click.Choice(['STND', 'DF', 'DS', 'SL', 'BR']),
              help='Material configuration code (auto-detected if not specified)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for results (optional)')
@click.option('--debug', is_flag=True, help='Show detailed debug information')
@click.option('--show-plot', is_flag=True, help='Show boundary validation plot')
def analyze_impulse_single(file_path, material, output, debug, show_plot):
    """
    🎯 Analyze single file using impulse method with optional visualization.
    
    Focus on total momentum transfer rather than kinetic energy estimation.
    Use --show-plot to validate integration boundaries.
    
    FILE_PATH: Path to CSV measurement file
    """
    file_path = Path(file_path)
    click.echo(f"🎯 Impulse Analysis: {file_path.name}")
    click.echo("-" * 40)
    
    try:
        if material is None:
            material = file_path.name.split('-')[0]
            click.echo(f"🔍 Auto-detected material: {material}")
        
        # Use the enhanced analyze_single_file_with_impulse with show_plot
        result = analyze_single_file_with_impulse(
            file_path, 
            material_code=material,
            include_line_mass=True,
            line_mass_fraction=LINE_MASS_FRACTION,
            show_plot=show_plot  # This is the new parameter
        )
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}")
            return
        
        # Display results
        impulse = result['total_impulse']
        abs_impulse = result['total_abs_impulse']
        peak_force = result['peak_force']
        duration = result['impact_duration'] * 1000  # ms
        
        click.echo("📊 IMPULSE ANALYSIS RESULTS:")
        click.echo(f"   Total impulse: {impulse:+.6f} N⋅s")
        click.echo(f"   Absolute impulse: {abs_impulse:.6f} N⋅s")
        click.echo(f"   Peak force: {peak_force:.0f} N")
        click.echo(f"   Impact duration: {duration:.1f} ms")
        
        # Interpretation
        if impulse > 0:
            click.echo(f"   → Forward momentum transfer (typical impact)")
        else:
            click.echo(f"   → Backward momentum transfer (strong rebound)")
        
        click.echo(f"   → Momentum magnitude: {abs(impulse):.6f} N⋅s")
        
        # For comparison with kinetic energy method
        if 'equivalent_velocity' in result and not np.isnan(result['equivalent_velocity']):
            equiv_v = result['equivalent_velocity']
            equiv_ke = result['equivalent_kinetic_energy']
            click.echo(f"\n📊 EQUIVALENT METRICS (for comparison):")
            click.echo(f"   Equivalent velocity: {equiv_v:.0f} m/s")
            click.echo(f"   Equivalent kinetic energy: {equiv_ke:.3f} J")
        
        if debug:
            click.echo(f"\n🔧 DEBUG INFO:")
            click.echo(f"   Material: {result['material_type']}")
            click.echo(f"   Mass: {result['mass_kg']*1000:.1f}g")
            click.echo(f"   Sampling rate: {result['sampling_rate_hz']:.0f} Hz")
            click.echo(f"   Impact start: {result['impact_start_time']:.4f} s")
            click.echo(f"   Impact end: {result['impact_end_time']:.4f} s")
        
        if show_plot:
            click.echo(f"\n📊 BOUNDARY VALIDATION:")
            click.echo(f"   Plot displayed showing integration boundaries")
            click.echo(f"   Check that the highlighted region captures the main impact")
            click.echo(f"   Duration should typically be 2-50 ms for fishing line impacts")
        
        # Save results if requested
        if output:
            import json
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\n💾 Results saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
