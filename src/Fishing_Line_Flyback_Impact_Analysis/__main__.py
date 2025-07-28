"""
Fishing Line Flyback Impact Analysis - Main CLI Interface

Enhanced version with configuration-specific weights and comprehensive analysis.
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


@click.group()
@click.version_option(version="2.1.0")
def main():
    """
    🎣 Fishing Line Flyback Impact Analysis v2.1
    
    Enhanced analysis with configuration-specific weights and measured line mass.
    
    Key Features:
    ✓ Configuration-specific weights (STND=45g, DF=60g, DS=72g, SL=69g, BR=45g)
    ✓ Measured line mass integration (5.5" = 0.542g → 38.8g total)
    ✓ 70% effective line mass (literature validated)
    ✓ Peak-focused impact detection for realistic velocities
    ✓ Target velocity range: 150-400 m/s
    ✓ Interactive data windowing tools
    """
    pass


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='comprehensive_analysis', 
              help='Output directory for results and plots')
@click.option('--show-progress', is_flag=True, default=True,
              help='Show real-time analysis progress')
def analyze_all(data_dir, output_dir, show_progress):
    """
    🚀 Run comprehensive analysis on all measurement files.
    
    This is the main analysis function that processes all your CSV files with:
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
        line_mass_effective = 0.0388 * LINE_MASS_FRACTION  # 38.8g × 70%
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
    🔍 Analyze a single measurement file.
    
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
            status_color = "green"
        elif 100 <= velocity <= 500:
            status = "🟡 GOOD"
            status_color = "yellow"
        else:
            status = "❌ REVIEW"
            status_color = "red"
        
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
@click.argument('test_file', type=click.Path(exists=True))
def test_line_mass(test_file):
    """
    🧪 Test different line mass fractions on a single file.
    
    TEST_FILE: Path to CSV file for testing
    """
    click.echo("🧪 TESTING LINE MASS FRACTIONS")
    click.echo("=" * 50)
    
    test_file = Path(test_file)
    material = test_file.name.split('-')[0]
    
    click.echo(f"📁 Test file: {test_file.name}")
    click.echo(f"🔍 Material: {material}")
    click.echo(f"📏 Line measurement: {MEASURED_LINE_LENGTH_INCHES}\" = {MEASURED_LINE_MASS_GRAMS}g")
    click.echo()
    
    fractions = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0]
    
    click.echo("📊 RESULTS:")
    click.echo(f"{'Fraction':<8} {'Line Mass':<9} {'Total Mass':<10} {'Velocity':<9} {'Energy':<8} {'Status'}")
    click.echo("-" * 60)
    
    for fraction in fractions:
        try:
            result = analyze_single_file_with_config(
                test_file, 
                material_code=material,
                include_line_mass=fraction > 0,
                line_mass_fraction=fraction
            )
            
            if 'error' not in result:
                velocity = abs(result['initial_velocity'])
                energy = result['kinetic_energy']
                total_mass = result['mass_kg'] * 1000
                line_mass = 38.8 * fraction
                
                if 150 <= velocity <= 400:
                    status = "✅"
                elif 100 <= velocity <= 500:
                    status = "🟡"
                else:
                    status = "❌"
                
                click.echo(f"{fraction*100:3.0f}%     {line_mass:5.1f}g    {total_mass:7.1f}g   {velocity:6.0f} m/s {energy:6.1f} J  {status}")
            else:
                click.echo(f"{fraction*100:3.0f}%     ERROR: {result['error']}")
                
        except Exception as e:
            click.echo(f"{fraction*100:3.0f}%     ERROR: {e}")
    
    click.echo()
    click.echo("📚 INTERPRETATION:")
    click.echo("   ✅ = Target range (150-400 m/s) - Ideal for fishing line impacts")
    click.echo("   🟡 = Reasonable range (100-500 m/s) - Acceptable physics")
    click.echo("   ❌ = Outside range - May need calibration")
    click.echo("   📖 Literature recommends 60-80% for flexible tethers")


@main.command()
@click.argument('file_path', type=click.Path(exists=True))
def window_tool(file_path):
    """
    🎛️  Launch interactive window selection tool.
    
    This tool helps you visually select the correct analysis window
    to isolate the actual impact event from noise and rebound.
    
    FILE_PATH: Path to CSV file for windowing
    """
    from .windowing import interactive_window_tool
    
    click.echo("🎛️  LAUNCHING INTERACTIVE WINDOW TOOL")
    click.echo("=" * 50)
    click.echo(f"📁 File: {file_path}")
    click.echo()
    click.echo("📋 Instructions:")
    click.echo("1. Drag on upper plot to select analysis window")
    click.echo("2. Click 'Analyze' to test window with different configurations")
    click.echo("3. Look for consistent velocities (150-400 m/s) across materials")
    click.echo("4. Click 'Save Window' to save your selection")
    click.echo("5. Close plot window when done")
    click.echo()
    
    interactive_window_tool(file_path)


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--pattern', default='*.csv', help='File pattern to match')
@click.option('--output', '-o', help='Output file for suggestions')
def suggest_windows(data_dir, pattern, output):
    """
    🔍 Auto-suggest analysis windows for all files.
    
    Automatically analyzes force data to suggest optimal analysis windows
    based on force characteristics and timing.
    
    DATA_DIR: Directory containing CSV files
    """
    from .windowing import batch_window_files
    
    click.echo("🔍 AUTO-SUGGESTING ANALYSIS WINDOWS")
    click.echo("=" * 50)
    
    suggestions = batch_window_files(data_dir, pattern)
    
    if suggestions:
        click.echo(f"\n📊 SUMMARY OF SUGGESTIONS:")
        click.echo(f"{'Filename':<20} {'Duration':<10} {'Peak Force':<10} {'Peak Time':<10}")
        click.echo("-" * 60)
        
        for s in suggestions:
            click.echo(f"{s['filename']:<20} {s['duration_ms']:<8.1f}ms {s['peak_force']:<8.0f}N {s['peak_time']:<8.4f}s")
        
        avg_duration = sum(s['duration_ms'] for s in suggestions) / len(suggestions)
        click.echo(f"\nAverage suggested duration: {avg_duration:.1f} ms")
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(suggestions, f, indent=2)
            click.echo(f"💾 Suggestions saved to: {output}")
    else:
        click.echo("❌ No valid suggestions generated")


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--windows-file', type=click.Path(exists=True),
              help='JSON file with window definitions')
def apply_windows(data_dir, windows_file):
    """
    ✂️  Apply saved windows to create windowed CSV files.
    
    Takes window definitions (from window-tool or suggest-windows) and
    creates new CSV files containing only the selected data ranges.
    
    DATA_DIR: Directory containing original CSV files
    """
    if not windows_file:
        # Look for default windows file
        windows_file = Path(data_dir) / "batch_window_suggestions.json"
        if not windows_file.exists():
            click.echo("❌ No windows file found. Use suggest-windows first or specify --windows-file")
            return
    
    click.echo("✂️  APPLYING WINDOWS TO CREATE WINDOWED FILES")
    click.echo("=" * 50)
    
    try:
        import json
        with open(windows_file, 'r') as f:
            windows = json.load(f)
        
        click.echo(f"📄 Loading windows from: {windows_file}")
        click.echo(f"🔄 Processing {len(windows)} files...")
        click.echo()
        
        success_count = 0
        data_path = Path(data_dir)
        
        for window in windows:
            try:
                original_file = data_path / window['filename']
                if not original_file.exists():
                    click.echo(f"❌ {window['filename']}: Original file not found")
                    continue
                
                # Load original data
                df = pd.read_csv(original_file)
                
                # Calculate indices from times
                analyzer = ImpactAnalyzer()
                force_data, _ = analyzer._calculate_total_force(df)
                time_data = analyzer._get_time_array(df, len(force_data))
                
                start_idx = np.searchsorted(time_data, window['start_time'])
                end_idx = np.searchsorted(time_data, window['end_time'])
                
                # Extract windowed data
                df_windowed = df.iloc[start_idx:end_idx+1].copy()
                
                # Reset time column if exists
                time_cols = [col for col in df_windowed.columns if 'time' in col.lower()]
                if time_cols:
                    time_col = time_cols[0]
                    df_windowed[time_col] = df_windowed[time_col] - df_windowed[time_col].iloc[0]
                
                # Save windowed file
                output_file = data_path / f"{original_file.stem}_windowed.csv"
                df_windowed.to_csv(output_file, index=False)
                
                click.echo(f"✅ {window['filename']}: {len(df_windowed)} samples → {output_file.name}")
                success_count += 1
                
            except Exception as e:
                click.echo(f"❌ {window['filename']}: {e}")
        
        click.echo(f"\n🎉 Successfully created {success_count}/{len(windows)} windowed files")
        
    except Exception as e:
        click.echo(f"❌ Error reading windows file: {e}")


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='windowed_analysis',
              help='Output directory for windowed analysis')
def analyze_windowed(data_dir, output_dir):
    """
    📊 Analyze all windowed CSV files.
    
    Runs the comprehensive analysis on files ending with '_windowed.csv'
    which should contain properly selected analysis windows.
    
    DATA_DIR: Directory containing windowed CSV files
    """
    click.echo("📊 ANALYZING WINDOWED FILES")
    click.echo("=" * 40)
    
    data_path = Path(data_dir)
    windowed_files = list(data_path.glob("*_windowed.csv"))
    
    if not windowed_files:
        click.echo("❌ No windowed CSV files found.")
        click.echo("💡 Use 'suggest-windows' and 'apply-windows' first")
        return
    
    click.echo(f"🔄 Found {len(windowed_files)} windowed files")
    click.echo()
    
    # Run analysis on windowed files
    try:
        # Temporarily rename files to remove _windowed suffix for analysis
        renamed_files = []
        for windowed_file in windowed_files:
            original_name = windowed_file.name.replace('_windowed.csv', '.csv')
            temp_name = windowed_file.parent / f"temp_{original_name}"
            windowed_file.rename(temp_name)
            renamed_files.append((temp_name, windowed_file))
        
        # Run comprehensive analysis
        results = run_comprehensive_analysis(str(data_path), output_dir)
        
        # Restore original windowed names
        for temp_file, original_windowed in renamed_files:
            temp_file.rename(original_windowed)
        
        # Update results with windowing info
        if results:
            valid_results = [r for r in results if 'error' not in r]
            velocities = [abs(r['initial_velocity']) for r in valid_results]
            target_count = sum(1 for v in velocities if 150 <= v <= 400)
            
            click.echo(f"\n🎉 WINDOWED ANALYSIS COMPLETE!")
            click.echo(f"✅ Successfully analyzed: {len(valid_results)}/{len(results)} files")
            click.echo(f"🎯 Target velocity range: {target_count}/{len(valid_results)} ({target_count/len(valid_results)*100:.1f}%)")
            
            if target_count/len(valid_results) > 0.8:
                click.echo("🟢 Excellent windowing results!")
            elif target_count/len(valid_results) > 0.6:
                click.echo("🟡 Good windowing results")
            else:
                click.echo("🔴 Consider reviewing window selections")
        
    except Exception as e:
        # Restore names in case of error
        for temp_file, original_windowed in renamed_files:
            if temp_file.exists():
                temp_file.rename(original_windowed)
        click.echo(f"❌ Error during analysis: {e}")


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
    
    click.echo("🎯 TARGET RANGES:")
    click.echo("   Ideal velocity: 200-350 m/s (literature)")
    click.echo("   Target velocity: 150-400 m/s (fishing line)")
    click.echo("   Reasonable velocity: 100-500 m/s (physics)")
    click.echo("   Expected energy: 1-200 J (for these masses)")
    click.echo()
    
    click.echo("📚 SCIENTIFIC REFERENCES:")
    click.echo("   Line mass fraction: Cartmell & McKenzie (2008), Irvine (1981)")
    click.echo("   Fishing line impacts: Steffens & Nettleton (2019)")
    click.echo("   Distributed mass theory: Thomson & Dahleh (1998)")
    click.echo("   High-speed impacts: Zukas (1990)")


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
@click.option('--mass', type=float, help='Object mass in kg')
def preview(file_path, mass):
    """
    👀 Preview force data without full analysis.
    
    FILE_PATH: Path to CSV or HDF5 data file
    """
    click.echo(f"👀 Previewing: {file_path.name}")
    click.echo("-" * 40)
    
    try:
        # Initialize analyzer
        analyzer = ImpactAnalyzer(mass=mass or 0.045, baseline_correction=True)
        
        # Load and process data
        if file_path.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            force_n, force_columns = analyzer._calculate_total_force(df)
            time = analyzer._get_time_array(df, len(force_n))
            
            click.echo(f"📊 Data loaded:")
            click.echo(f"   Force columns: {', '.join(force_columns)}")
            click.echo(f"   Data points: {len(force_n)}")
            click.echo(f"   Duration: {time[-1]:.4f} s")
            click.echo(f"   Force range: {np.min(force_n):.1f} - {np.max(force_n):.1f} N")
            click.echo(f"   Sampling rate: {analyzer.sampling_rate:.0f} Hz")
            
        elif file_path.suffix.lower() == '.h5':
            import h5py
            with h5py.File(file_path, 'r') as f:
                if 'SUM' in f:
                    force_lbf = f['SUM'][:]
                    force_source = "SUM channel"
                else:
                    ai_datasets = [key for key in f.keys() if key.startswith('AI')]
                    if ai_datasets:
                        force_lbf = sum(f[key][:] for key in ai_datasets)
                        force_source = f"Sum of {len(ai_datasets)} AI channels"
                    else:
                        raise ValueError("No suitable force data found in HDF5 file")
                
                force_n = analyzer._convert_lbf_to_n(force_lbf)
                force_n = analyzer._apply_baseline_correction(force_n)
                time = np.arange(len(force_n)) * analyzer.dt
                
                click.echo(f"📊 Data loaded:")
                click.echo(f"   Source: {force_source}")
                click.echo(f"   Data points: {len(force_n)}")
                click.echo(f"   Duration: {time[-1]:.4f} s")
                click.echo(f"   Force range: {np.min(force_n):.1f} - {np.max(force_n):.1f} N")
        
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Basic analysis preview
        max_force = np.max(np.abs(force_n))
        max_force_time = time[np.argmax(np.abs(force_n))]
        
        click.echo(f"\n🔍 Quick analysis:")
        click.echo(f"   Peak force: {max_force:.1f} N at t={max_force_time:.4f} s")
        
        # Estimate velocity without full analysis
        simple_impulse = np.trapz(force_n, dx=analyzer.dt)
        est_velocity = abs(simple_impulse / analyzer.mass)
        
        click.echo(f"   Estimated velocity: {est_velocity:.0f} m/s (rough calculation)")
        
        if est_velocity > 1000:
            click.echo("   ⚠️  Very high velocity estimate - may need mass correction")
        elif est_velocity < 50:
            click.echo("   ⚠️  Very low velocity estimate - check data quality")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@main.command()
def info():
    """
    ℹ️  Show package information and version details.
    """
    click.echo("🎣 FISHING LINE FLYBACK IMPACT ANALYSIS")
    click.echo("=" * 50)
    click.echo("Version: 2.1.0")
    click.echo("Enhanced with configuration-specific weights")
    click.echo()
    
    click.echo("🔬 METHODOLOGY:")
    click.echo("• Peak-focused impact detection")
    click.echo("• Configuration-specific hardware masses")
    click.echo("• Measured line mass integration (70% effective)")
    click.echo("• Automatic force unit conversion (lbf → N)")
    click.echo("• Baseline correction and outlier filtering")
    click.echo("• Realistic velocity targeting (150-400 m/s)")
    click.echo("• Interactive data windowing for precise analysis")
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
    
    click.echo("🎛️  DATA WINDOWING TOOLS:")
    click.echo("• Interactive window selection with visual feedback")
    click.echo("• Automatic window suggestions based on force characteristics")
    click.echo("• Batch processing for multiple files")
    click.echo("• Real-time analysis validation across configurations")
    click.echo()
    
    click.echo("📚 SCIENTIFIC BASIS:")
    click.echo("• Line mass theory: Cartmell & McKenzie (2008)")
    click.echo("• Distributed systems: Thomson & Dahleh (1998)")
    click.echo("• Cable impact loading: Irvine (1981)")
    click.echo("• High-speed impacts: Zukas (1990)")
    click.echo("• Fishing line analysis: Steffens & Nettleton (2019)")
    click.echo()
    
    click.echo("🚀 WINDOWING WORKFLOW:")
    click.echo("1. python -m Fishing_Line_Flyback_Impact_Analysis suggest-windows data/csv")
    click.echo("2. python -m Fishing_Line_Flyback_Impact_Analysis window-tool data/csv/problematic_file.csv")
    click.echo("3. python -m Fishing_Line_Flyback_Impact_Analysis apply-windows data/csv")
    click.echo("4. python -m Fishing_Line_Flyback_Impact_Analysis analyze-windowed data/csv")
    click.echo()
    
    click.echo("⚡ QUICK START:")
    click.echo("# Analyze all files (current method):")
    click.echo("python -m Fishing_Line_Flyback_Impact_Analysis analyze-all data/csv")
    click.echo()
    click.echo("# With windowing (recommended for problematic data):")
    click.echo("python -m Fishing_Line_Flyback_Impact_Analysis suggest-windows data/csv")
    click.echo("python -m Fishing_Line_Flyback_Impact_Analysis apply-windows data/csv")
    click.echo("python -m Fishing_Line_Flyback_Impact_Analysis analyze-windowed data/csv")

@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--suggestions-file', type=click.Path(exists=True),
              help='JSON file with window suggestions')
@click.option('--save-report', is_flag=True, help='Save detailed report to file')
def review_windows(data_dir, suggestions_file, save_report):
    """
    🔍 Review automatically suggested analysis windows.
    
    Analyzes the quality of automatic window suggestions and identifies
    files that need manual review using the interactive window tool.
    
    DATA_DIR: Directory containing CSV files and suggestions
    """
    from .windowing import review_window_suggestions, create_review_report
    
    click.echo("🔍 REVIEWING WINDOW SUGGESTIONS")
    click.echo("=" * 50)
    
    if save_report:
        review_results = create_review_report(data_dir)
    else:
        review_results = review_window_suggestions(data_dir, suggestions_file)
    
    if review_results:
        click.echo(f"\n📋 QUICK SUMMARY:")
        click.echo(f"   Total suggestions: {review_results['total_suggestions']}")
        click.echo(f"   Priority review needed: {len(review_results.get('priority_files', []))}")
        click.echo(f"   Potential issues found:")
        click.echo(f"     • Short windows (<3ms): {review_results.get('short_windows', 0)}")
        click.echo(f"     • Max duration (20ms): {review_results.get('max_windows', 0)}")
        click.echo(f"     • Low force (<1kN): {review_results.get('low_force', 0)}")
        click.echo(f"     • Late peaks (>100s): {review_results.get('late_peaks', 0)}")


if __name__ == "__main__":
    main()
