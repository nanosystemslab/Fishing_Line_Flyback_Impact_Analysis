#!/usr/bin/env python3

"""Fishing Line Flyback Impact Analysis Tool.

Unified command-line interface for analyzing and
visualizing fishing line flyback impact properties.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from . import __version__
from .analysis import ImpactAnalyzer
from .visualization import ImpactVisualizer


def setup_logging(verbosity: int) -> None:
    """Setup logging configuration."""
    log_fmt = "%(levelname)s - %(module)s - %(funcName)s @%(lineno)d: %(message)s"
    logging.basicConfig(
        filename=None, 
        format=log_fmt, 
        level=logging.getLevelName(verbosity)
    )


def parse_command_line() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize fishing line flyback impact properties",
        prog="Fishing_Line_Flyback_Impact_Analysis",
    )

    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        dest="verbosity", help="Verbose output (use -vv for more verbose)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Single file analysis command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze impact properties from sensor data"
    )
    analyze_parser.add_argument(
        "-i", "--input", nargs="+", required=True, 
        help="Path(s) to input CSV or H5 files"
    )
    analyze_parser.add_argument(
        "-o", "--output", default="out", 
        help="Output directory (default: out)"
    )
    analyze_parser.add_argument(
        "--param-y", choices=["SUM", "S1", "S2", "S3", "S4", "Max", "All"],
        default="SUM", help="Y-axis parameter for plotting (default: SUM)"
    )
    analyze_parser.add_argument(
        "--param-x", choices=["time"], default="time",
        help="X-axis parameter for plotting (default: time)"
    )
    analyze_parser.add_argument(
        "--show-all-sensors", action="store_true",
        help="Plot all individual sensors on same graph"
    )

    # Post-processing command for results files
    postprocess_parser = subparsers.add_parser(
        "postprocess", help="Post-process results from text file"
    )
    postprocess_parser.add_argument(
        "-i", "--input", required=True,
        help="Path to results text file"
    )
    postprocess_parser.add_argument(
        "-o", "--output", default="out",
        help="Output directory (default: out)"
    )
    postprocess_parser.add_argument(
        "--plot-type", choices=["box", "violin", "dual", "all"], 
        default="all", help="Type of plots to generate (default: all)"
    )
    postprocess_parser.add_argument(
        "--generate-table", action="store_true",
        help="Also generate LaTeX table for publication"
    )

    # Batch processing command  
    batch_parser = subparsers.add_parser(
        "batch", help="Batch process multiple files"
    )
    batch_parser.add_argument(
        "-d", "--data-dir", required=True,
        help="Directory containing data files"
    )
    batch_parser.add_argument(
        "-o", "--output", default="out",
        help="Output directory (default: out)"
    )
    batch_parser.add_argument(
        "--summary", action="store_true",
        help="Generate summary statistics and report"
    )

    # Table generation command
    table_parser = subparsers.add_parser(
        "table", help="Generate LaTeX table from results file"
    )
    table_parser.add_argument(
        "-i", "--input", required=True,
        help="Path to results text file"
    )
    table_parser.add_argument(
        "-o", "--output", default="out",
        help="Output directory (default: out)"
    )

    # Visualization command for output data
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize output data from CSV files"
    )
    visualize_parser.add_argument(
        "-i", "--input", nargs="+", required=True,
        help="Path(s) to output CSV files"
    )
    visualize_parser.add_argument(
        "-o", "--output", default="out",
        help="Output directory (default: out)"
    )
    visualize_parser.add_argument(
        "--x-param", choices=["D", "L", "KE", "V"], default="D",
        help="X-axis parameter (default: D)"
    )
    visualize_parser.add_argument(
        "--y-param", choices=["D", "L", "KE", "V"], default="KE", 
        help="Y-axis parameter (default: KE)"
    )

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["verbosity"] = max(0, 30 - 10 * args_dict["verbosity"])

    return args_dict


def handle_analyze_command(args: Dict[str, Any]) -> int:
    """Handle the analyze command for single/multiple files."""
    analyzer = ImpactAnalyzer()
    visualizer = ImpactVisualizer(output_dir=args["output"])

    try:
        results = []
        
        for input_file in args["input"]:
            logging.info(f"Processing file: {input_file}")
            
            # Load and analyze file
            df = analyzer.load_file(input_file)
            properties = analyzer.calculate_impact_properties(
                df, param_y=args["param_y"], param_x=args["param_x"]
            )
            
            # Create time series plot
            visualizer.plot_time_series(
                df, 
                param_y=args["param_y"], 
                param_x=args["param_x"],
                show_all_sensors=args["show_all_sensors"]
            )
            
            # Store results - use the metadata from df.meta
            properties["filepath"] = input_file
            properties["filename"] = df.meta.fname  # Use df.meta.fname instead of properties['filename']
            results.append(properties)
            
            # Print results to console
            print(f"{df.meta.fname},J={properties['impulse_Ns']:.3f},"
                  f"F={properties['max_force_N']:.2f}")

        # Save results to text file for later post-processing
        results_file = Path(args["output"]) / "run_results.txt"
        with open(results_file, "w") as f:
            for result in results:
                f.write(f"{result['filename']},J={result['impulse_Ns']:.3f},"
                       f"F={result['max_force_N']:.2f}\n")

        logging.info(f"Analysis complete. Results saved to {args['output']}")
        return 0

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        import traceback
        logging.error(traceback.format_exc())  # Add detailed error info
        return 1


def handle_postprocess_command(args: Dict[str, Any]) -> int:
    """Handle the postprocess command for results files."""
    analyzer = ImpactAnalyzer()
    visualizer = ImpactVisualizer(output_dir=args["output"])

    try:
        # Load results file
        results_df = analyzer.load_results_file(args["input"])
        
        # Calculate summary statistics
        summary_stats = analyzer.calculate_summary_stats(results_df)
        
        # Save summary to CSV
        analyzer.save_results_to_csv(summary_stats, args["output"])
        
        # Generate plots based on plot type
        if args["plot_type"] in ["box", "all"]:
            visualizer.plot_box_plots(results_df, param="F")
            visualizer.plot_box_plots(results_df, param="J")
            
        if args["plot_type"] in ["violin", "all"]:
            visualizer.plot_violin_plots(results_df, param="F")
            visualizer.plot_violin_plots(results_df, param="J")
            
        if args["plot_type"] in ["dual", "all"]:
            visualizer.plot_dual_box_plots(results_df, save_format="png")
            visualizer.plot_dual_box_plots(results_df, save_format="svg")

        # Generate summary report
        analyzer.generate_summary_report(summary_stats, args["output"])

        # Generate summary table if requested
        if args.get("generate_table", False):
            analyzer.generate_summary_table(results_df, args["output"])

        logging.info(f"Post-processing complete. Results saved to {args['output']}")
        return 0

    except Exception as e:
        logging.error(f"Post-processing failed: {e}")
        return 1


def handle_batch_command(args: Dict[str, Any]) -> int:
    """Handle the batch processing command."""
    analyzer = ImpactAnalyzer()
    visualizer = ImpactVisualizer(output_dir=args["output"])

    try:
        data_dir = Path(args["data_dir"])
        if not data_dir.exists():
            logging.error(f"Data directory {data_dir} does not exist")
            return 1

        # Find all CSV and H5 files
        file_patterns = ["*.csv", "*.h5"]
        all_files = []
        for pattern in file_patterns:
            all_files.extend(data_dir.rglob(pattern))

        if not all_files:
            logging.error(f"No data files found in {data_dir}")
            return 1

        logging.info(f"Found {len(all_files)} files to process")
        
        results = []
        for file_path in all_files:
            try:
                logging.info(f"Processing: {file_path}")
                result = analyzer.process_single_file(str(file_path))
                results.append(result)
                
                # Print progress
                print(f"{result['filename']},J={result['impulse_Ns']:.3f},"
                      f"F={result['max_force_N']:.2f}")
                      
            except Exception as e:
                logging.warning(f"Failed to process {file_path}: {e}")
                continue

        # Save all results
        results_file = Path(args["output"]) / "batch_results.txt"
        with open(results_file, "w") as f:
            for result in results:
                f.write(f"{result['filename']},J={result['impulse_Ns']:.3f},"
                       f"F={result['max_force_N']:.2f}\n")

        if args["summary"]:
            # Create DataFrame from results for summary analysis
            results_data = []
            for result in results:
                results_data.append({
                    'fname': result['filename'],
                    'test_type': result['config'],
                    'J': result['impulse_Ns'],
                    'F': result['max_force_N']
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Calculate and save summary statistics
            summary_stats = analyzer.calculate_summary_stats(results_df)
            analyzer.save_results_to_csv(summary_stats, args["output"])
            analyzer.generate_summary_report(summary_stats, args["output"])
            
            # Create summary plots
            visualizer.create_summary_plots(results_df)

        logging.info(f"Batch processing complete. Results saved to {args['output']}")
        return 0

    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        return 1


def handle_table_command(args: Dict[str, Any]) -> int:
    """Handle the table generation command."""
    analyzer = ImpactAnalyzer()

    try:
        # Load results file
        results_df = analyzer.load_results_file(args["input"])
        
        # Generate and print human-readable table
        config_stats = analyzer.generate_summary_table(results_df, args["output"])
        
        logging.info(f"Summary table generated and saved to {args['output']}")
        
        return 0

    except Exception as e:
        logging.error(f"Table generation failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


def handle_visualize_command(args: Dict[str, Any]) -> int:
    """Handle the visualize command for output data."""
    visualizer = ImpactVisualizer(output_dir=args["output"])

    try:
        for input_file in args["input"]:
            visualizer.plot_output_data(
                filepath=input_file, 
                x_param=args["x_param"], 
                y_param=args["y_param"]
            )

        logging.info(f"Visualization complete. Results saved to {args['output']}")
        return 0

    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_command_line()
    setup_logging(args["verbosity"])

    command_handlers = {
        "analyze": handle_analyze_command,
        "postprocess": handle_postprocess_command, 
        "batch": handle_batch_command,
        "table": handle_table_command,
        "visualize": handle_visualize_command,
    }

    handler = command_handlers.get(args["command"])
    if handler:
        return handler(args)
    else:
        logging.error(f"Unknown command: {args['command']}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(1)
