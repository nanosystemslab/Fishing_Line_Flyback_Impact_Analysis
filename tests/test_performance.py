"""Performance tests for fishing line analysis package."""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Fishing_Line_Flyback_Impact_Analysis import ImpulseAnalyzer
from Fishing_Line_Flyback_Impact_Analysis import analyze_single_file_with_impulse
from Fishing_Line_Flyback_Impact_Analysis import run_impulse_analysis


@pytest.mark.slow
class TestPerformance:
    """Performance tests for various components."""

    def create_large_dataset(self, n_points=1000000):
        """Create large dataset for performance testing."""
        time_array = np.linspace(0, 10, n_points)  # 10 seconds
        force1 = np.random.normal(0, 50, n_points)
        force2 = np.random.normal(0, 25, n_points)

        # Add multiple impact events
        for i in range(5):
            start = i * n_points // 6 + n_points // 10
            end = start + n_points // 20
            magnitude = 1000 + i * 200

            if end <= n_points:
                force1[start:end] = magnitude * np.sin(
                    np.linspace(0, np.pi, end - start)
                )

        return {
            "AI_Channel_1_lbf": force1,
            "AI_Channel_2_lbf": force2,
            "Time": time_array,
        }

    def test_large_dataset_analysis_performance(self, benchmark_timer, memory_monitor):
        """Test performance with large dataset (1M points)."""
        # Create large dataset
        test_data = self.create_large_dataset(1000000)

        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / "large_test.csv"

        try:
            # Save to CSV
            df = pd.DataFrame(test_data)
            df.to_csv(file_path, index=False)

            # Test analysis performance
            memory_monitor.start()
            benchmark_timer.start()

            result = analyze_single_file_with_impulse(file_path, show_plot=False)

            elapsed = benchmark_timer.stop()
            memory_stats = memory_monitor.stop()

            # Verify result
            assert "error" not in result
            assert "total_impulse" in result

            # Performance assertions
            assert elapsed < 30.0, f"Analysis took too long: {elapsed:.2f}s"
            assert (
                memory_stats["increase_mb"] < 500
            ), f"Memory usage too high: {memory_stats['increase_mb']:.1f}MB"

            print(
                f"Large dataset analysis: {elapsed:.2f}s, {memory_stats['increase_mb']:.1f}MB"
            )

        finally:
            if file_path.exists():
                file_path.unlink()

    def test_boundary_detection_performance(self, benchmark_timer):
        """Test boundary detection performance with various data sizes."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        sizes = [1000, 10000, 100000, 1000000]
        times = []

        for size in sizes:
            # Create test data
            force = np.random.normal(0, 10, size)
            force[size // 3 : 2 * size // 3] = 1000  # Clear impact

            benchmark_timer.start()
            start_idx, end_idx = analyzer.find_impact_boundaries(force)
            elapsed = benchmark_timer.stop()

            times.append(elapsed)

            # Verify result
            assert 0 <= start_idx < end_idx < size

            print(f"Boundary detection {size:,} points: {elapsed:.4f}s")

        # Performance should scale reasonably
        # 1M points should not take more than 100x longer than 1K points
        ratio = times[-1] / times[0]  # 1M vs 1K
        assert ratio < 100, f"Performance scaling too poor: {ratio:.1f}x"

    def test_impulse_calculation_performance(self, benchmark_timer):
        """Test impulse calculation performance."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        sizes = [1000, 10000, 100000]

        for size in sizes:
            time_array = np.linspace(0, size * 1e-5, size)  # Realistic time scale
            force = np.random.normal(0, 10, size)
            force[size // 3 : 2 * size // 3] = 1000

            benchmark_timer.start()
            result = analyzer.calculate_impulse_metrics(force, time_array)
            elapsed = benchmark_timer.stop()

            assert "error" not in result
            assert (
                elapsed < 1.0
            ), f"Impulse calculation too slow for {size} points: {elapsed:.3f}s"

            print(f"Impulse calculation {size:,} points: {elapsed:.4f}s")

    def test_batch_analysis_performance(self, benchmark_timer, memory_monitor):
        """Test batch analysis performance with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple test files
            n_files = 20
            points_per_file = 50000

            print(
                f"Creating {n_files} test files with {points_per_file:,} points each..."
            )

            for i in range(n_files):
                material = ["STND", "DF", "DS", "SL", "BR"][i % 5]
                test_data = self.create_large_dataset(points_per_file)

                file_path = temp_path / f"{material}-21-{i + 1}.csv"
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)

            # Test batch analysis
            memory_monitor.start()
            benchmark_timer.start()

            results = run_impulse_analysis(str(temp_path), str(temp_path / "output"))

            elapsed = benchmark_timer.stop()
            memory_stats = memory_monitor.stop()

            # Verify results
            assert len(results) == n_files
            valid_results = [r for r in results if "error" not in r]
            assert len(valid_results) == n_files

            # Performance assertions
            time_per_file = elapsed / n_files
            assert (
                time_per_file < 5.0
            ), f"Average time per file too high: {time_per_file:.2f}s"
            assert (
                memory_stats["increase_mb"] < 1000
            ), f"Memory usage too high: {memory_stats['increase_mb']:.1f}MB"

            print(
                f"Batch analysis {n_files} files: {elapsed:.2f}s total, {time_per_file:.2f}s per file"
            )
            print(f"Memory usage: {memory_stats['increase_mb']:.1f}MB")

    def test_memory_efficiency(self, memory_monitor):
        """Test memory efficiency with repeated analyses."""
        memory_monitor.start()

        # Run multiple analyses to check for memory leaks
        temp_dir = Path(tempfile.gettempdir())

        for i in range(10):
            test_data = self.create_large_dataset(100000)
            file_path = temp_dir / f"memory_test_{i}.csv"

            try:
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)

                result = analyze_single_file_with_impulse(file_path, show_plot=False)
                assert "error" not in result

                # Check memory periodically
                if i % 3 == 0:
                    memory_monitor.update()

            finally:
                if file_path.exists():
                    file_path.unlink()

        memory_stats = memory_monitor.stop()

        # Memory should not grow excessively
        assert (
            memory_stats["increase_mb"] < 200
        ), f"Potential memory leak: {memory_stats['increase_mb']:.1f}MB increase"

        print(
            f"Memory efficiency test: {memory_stats['increase_mb']:.1f}MB increase over 10 iterations"
        )

    def test_file_io_performance(self, benchmark_timer):
        """Test file I/O performance with different file sizes."""
        from Fishing_Line_Flyback_Impact_Analysis.shared import calculate_total_force
        from Fishing_Line_Flyback_Impact_Analysis.shared import load_csv_file

        sizes = [1000, 10000, 100000]

        for size in sizes:
            test_data = self.create_large_dataset(size)
            temp_dir = Path(tempfile.gettempdir())
            file_path = temp_dir / f"io_test_{size}.csv"

            try:
                # Write performance
                df = pd.DataFrame(test_data)
                benchmark_timer.start()
                df.to_csv(file_path, index=False)
                write_time = benchmark_timer.stop()

                # Read performance
                benchmark_timer.start()
                df_loaded = load_csv_file(file_path)
                read_time = benchmark_timer.stop()

                # Processing performance
                benchmark_timer.start()
                force, columns = calculate_total_force(df_loaded)
                process_time = benchmark_timer.stop()

                print(
                    f"File I/O {size:,} points: write={write_time:.3f}s, read={read_time:.3f}s, process={process_time:.3f}s"
                )

                # Performance assertions
                assert write_time < 5.0, f"Write too slow: {write_time:.3f}s"
                assert read_time < 5.0, f"Read too slow: {read_time:.3f}s"
                assert process_time < 1.0, f"Processing too slow: {process_time:.3f}s"

            finally:
                if file_path.exists():
                    file_path.unlink()


@pytest.mark.slow
class TestScalability:
    """Test scalability with varying parameters."""

    def test_scaling_with_data_size(self):
        """Test how performance scales with data size."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        sizes = [1000, 2000, 5000, 10000, 20000, 50000]
        times = []

        for size in sizes:
            time_array = np.linspace(0, 0.1, size)
            force = np.random.normal(0, 100, size)
            force[size // 3 : 2 * size // 3] = 1000

            start_time = time.time()
            result = analyzer.calculate_impulse_metrics(force, time_array)
            elapsed = time.time() - start_time

            times.append(elapsed)
            assert "error" not in result

        # Check that scaling is roughly linear
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1]

            # Time ratio should not be much worse than size ratio
            assert (
                time_ratio < size_ratio * 2
            ), f"Poor scaling at size {sizes[i]}: {time_ratio:.2f}x vs {size_ratio:.2f}x"

    def test_scaling_with_file_count(self):
        """Test how batch analysis scales with number of files."""
        file_counts = [1, 2, 5, 10]
        times_per_file = []

        for count in file_counts:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create test files
                for i in range(count):
                    test_data = {
                        "AI_Channel_1_lbf": np.random.normal(0, 10, 5000),
                        "AI_Channel_2_lbf": np.random.normal(0, 5, 5000),
                        "Time": np.linspace(0, 0.05, 5000),
                    }
                    # Add impact
                    test_data["AI_Channel_1_lbf"][1500:3500] = 1000

                    file_path = temp_path / f"STND-21-{i + 1}.csv"
                    df = pd.DataFrame(test_data)
                    df.to_csv(file_path, index=False)

                # Time batch analysis
                start_time = time.time()
                results = run_impulse_analysis(
                    str(temp_path), str(temp_path / "output")
                )
                elapsed = time.time() - start_time

                time_per_file = elapsed / count
                times_per_file.append(time_per_file)

                assert len(results) == count
                valid_results = [r for r in results if "error" not in r]
                assert len(valid_results) == count

        # Time per file should remain roughly constant
        avg_time = np.mean(times_per_file)
        for tpf in times_per_file:
            assert (
                abs(tpf - avg_time) < avg_time * 0.5
            ), f"Poor scaling: time per file varies too much"

    def test_memory_scaling(self, memory_monitor):
        """Test memory usage scaling with data size."""
        sizes = [10000, 50000, 100000, 200000]
        memory_usage = []

        for size in sizes:
            memory_monitor.start()

            test_data = {
                "AI_Channel_1_lbf": np.random.normal(0, 10, size),
                "AI_Channel_2_lbf": np.random.normal(0, 5, size),
                "Time": np.linspace(0, size * 1e-5, size),
            }
            test_data["AI_Channel_1_lbf"][size // 3 : 2 * size // 3] = 1000

            temp_dir = Path(tempfile.gettempdir())
            file_path = temp_dir / f"memory_scaling_{size}.csv"

            try:
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)

                result = analyze_single_file_with_impulse(file_path, show_plot=False)
                assert "error" not in result

                stats = memory_monitor.stop()
                memory_usage.append(stats["increase_mb"])

            finally:
                if file_path.exists():
                    file_path.unlink()

        # Memory usage should scale roughly linearly with data size
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            memory_ratio = memory_usage[i] / max(
                memory_usage[i - 1], 1
            )  # Avoid division by zero

            # Memory should not scale much worse than data size
            assert (
                memory_ratio < size_ratio * 3
            ), f"Poor memory scaling: {memory_ratio:.2f}x vs {size_ratio:.2f}x"


@pytest.mark.slow
class TestRobustness:
    """Test robustness under various conditions."""

    def test_extreme_data_values(self):
        """Test with extreme data values."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        test_cases = [
            # Very large forces
            {"force_scale": 1e6, "name": "very_large_forces"},
            # Very small forces
            {"force_scale": 1e-6, "name": "very_small_forces"},
            # Mixed scales
            {"force_scale": 1.0, "offset": 1e6, "name": "large_offset"},
        ]

        for case in test_cases:
            time_array = np.linspace(0, 0.1, 10000)
            force = np.random.normal(0, 10, 10000) * case.get("force_scale", 1.0)
            force += case.get("offset", 0)

            # Add impact
            force[3000:7000] = 1000 * case.get("force_scale", 1.0) + case.get(
                "offset", 0
            )

            try:
                result = analyzer.calculate_impulse_metrics(force, time_array)

                # Should either succeed or fail gracefully
                if "error" not in result:
                    assert "total_impulse" in result
                    assert np.isfinite(result["total_impulse"])
                    print(f"Extreme data test '{case['name']}': SUCCESS")
                else:
                    print(
                        f"Extreme data test '{case['name']}': Failed gracefully - {result['error']}"
                    )

            except Exception as e:
                # Should not crash with unhandled exceptions
                pytest.fail(f"Extreme data test '{case['name']}' crashed: {e}")

    def test_noisy_data_robustness(self):
        """Test robustness with very noisy data."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        noise_levels = [0.1, 1.0, 10.0, 100.0]  # Relative to signal

        for noise_level in noise_levels:
            time_array = np.linspace(0, 0.1, 10000)

            # Signal
            signal = np.zeros(10000)
            signal[4000:6000] = 1000  # Clear impact

            # Add noise
            noise = np.random.normal(0, 1000 * noise_level, 10000)
            force = signal + noise

            result = analyzer.calculate_impulse_metrics(force, time_array)

            # Should handle gracefully
            if "error" not in result:
                # For very noisy data, results might not be accurate but should be finite
                assert np.isfinite(result["total_impulse"])
                assert np.isfinite(result["peak_force"])
                print(f"Noise level {noise_level}x: Analysis succeeded")
            else:
                print(f"Noise level {noise_level}x: Failed gracefully")

    def test_long_duration_robustness(self):
        """Test with very long duration data."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Test with different durations
        durations = [1.0, 10.0, 100.0]  # seconds

        for duration in durations:
            n_points = int(duration * 10000)  # 10kHz effective sampling
            time_array = np.linspace(0, duration, n_points)
            force = np.random.normal(0, 10, n_points)

            # Add impact somewhere in the middle
            impact_start = n_points // 2
            impact_end = impact_start + min(1000, n_points // 10)
            force[impact_start:impact_end] = 1000

            start_time = time.time()
            result = analyzer.calculate_impulse_metrics(force, time_array)
            elapsed = time.time() - start_time

            if "error" not in result:
                assert "total_impulse" in result
                print(
                    f"Duration {duration}s ({n_points:,} points): {elapsed:.2f}s analysis time"
                )

                # Should complete in reasonable time
                assert (
                    elapsed < 60
                ), f"Analysis too slow for {duration}s data: {elapsed:.2f}s"
            else:
                print(f"Duration {duration}s: Failed gracefully - {result['error']}")

    def test_concurrent_analysis_safety(self):
        """Test thread safety with concurrent analyses."""
        import queue
        import threading

        def analyze_worker(work_queue, result_queue):
            """Worker function for concurrent analysis."""
            while True:
                try:
                    work_item = work_queue.get(timeout=1)
                    if work_item is None:
                        break

                    material, data = work_item
                    analyzer = ImpulseAnalyzer(material_code=material)

                    time_array = np.linspace(0, 0.1, 5000)
                    force = data + np.random.normal(0, 5, 5000)
                    force[1500:3500] = 1000 + np.random.normal(0, 100, 2000)

                    result = analyzer.calculate_impulse_metrics(force, time_array)
                    result_queue.put((material, result))

                except queue.Empty:
                    break
                except Exception as e:
                    result_queue.put((None, {"error": str(e)}))

        # Create work items
        work_queue = queue.Queue()
        result_queue = queue.Queue()

        materials = ["STND", "DF", "DS", "SL", "BR"] * 4  # 20 total jobs
        for material in materials:
            base_data = np.random.normal(0, 10, 5000)
            work_queue.put((material, base_data))

        # Start workers
        n_workers = 4
        threads = []
        for _ in range(n_workers):
            thread = threading.Thread(
                target=analyze_worker, args=(work_queue, result_queue)
            )
            thread.start()
            threads.append(thread)

        # Signal workers to stop
        for _ in range(n_workers):
            work_queue.put(None)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Collect results
        results = []
        while not result_queue.empty():
            material, result = result_queue.get()
            results.append((material, result))

        # Verify results
        assert len(results) == len(
            materials
        ), f"Expected {len(materials)} results, got {len(results)}"

        successful = [r for m, r in results if "error" not in r]
        assert (
            len(successful) > len(materials) * 0.8
        ), "Too many failures in concurrent analysis"

        print(f"Concurrent analysis: {len(successful)}/{len(results)} successful")


@pytest.mark.slow
class TestStressTests:
    """Stress tests for extreme conditions."""

    def test_memory_stress(self, memory_monitor):
        """Stress test memory usage with many large analyses."""
        memory_monitor.start()

        n_iterations = 50
        points_per_iteration = 100000

        for i in range(n_iterations):
            # Create large dataset
            time_array = np.linspace(0, 1.0, points_per_iteration)
            force = np.random.normal(0, 50, points_per_iteration)

            # Add multiple impacts
            for j in range(5):
                start = j * points_per_iteration // 6 + points_per_iteration // 10
                end = start + points_per_iteration // 50
                if end <= points_per_iteration:
                    force[start:end] = 1000 + j * 100

            # Analyze
            analyzer = ImpulseAnalyzer(material_code="STND")
            result = analyzer.calculate_impulse_metrics(force, time_array)

            assert "error" not in result

            # Check memory every 10 iterations
            if i % 10 == 0:
                memory_monitor.update()
                current_stats = memory_monitor.stop()

                # Should not use excessive memory
                assert (
                    current_stats["increase_mb"] < 1000
                ), f"Memory usage too high at iteration {i}: {current_stats['increase_mb']:.1f}MB"

                # Restart monitoring
                memory_monitor.start()

        final_stats = memory_monitor.stop()
        print(f"Memory stress test: {final_stats['increase_mb']:.1f}MB final increase")

    def test_file_system_stress(self):
        """Stress test file system operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create many files rapidly
            n_files = 100
            files_created = []

            start_time = time.time()

            for i in range(n_files):
                test_data = {
                    "AI_Channel_1_lbf": np.random.normal(0, 10, 5000),
                    "AI_Channel_2_lbf": np.random.normal(0, 5, 5000),
                    "Time": np.linspace(0, 0.05, 5000),
                }
                test_data["AI_Channel_1_lbf"][1500:3500] = 1000 + i * 10

                file_path = temp_path / f"stress_test_{i:03d}.csv"
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)
                files_created.append(file_path)

            creation_time = time.time() - start_time

            # Batch analyze all files
            start_time = time.time()
            results = run_impulse_analysis(str(temp_path), str(temp_path / "output"))
            analysis_time = time.time() - start_time

            # Verify results
            assert len(results) == n_files
            valid_results = [r for r in results if "error" not in r]
            success_rate = len(valid_results) / len(results)

            assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"

            print(
                f"File system stress: {n_files} files, {creation_time:.2f}s creation, {analysis_time:.2f}s analysis"
            )

    def test_boundary_detection_stress(self):
        """Stress test boundary detection with difficult cases."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        difficult_cases = [
            # Multiple peaks
            {
                "name": "multiple_peaks",
                "peaks": [(1000, 2000, 1000), (3000, 4000, 800), (6000, 7000, 1200)],
            },
            # Very weak signal
            {"name": "weak_signal", "peaks": [(4000, 5000, 50)]},
            # Very short impact
            {"name": "short_impact", "peaks": [(4990, 5010, 2000)]},
            # Very long impact
            {"name": "long_impact", "peaks": [(1000, 8000, 500)]},
            # Asymmetric impact
            {"name": "asymmetric", "peaks": [(3000, 7000, 1000)], "asymmetric": True},
        ]

        for case in difficult_cases:
            force = np.random.normal(0, 20, 10000)  # Noisy baseline

            for start, end, magnitude in case["peaks"]:
                if case.get("asymmetric", False):
                    # Asymmetric profile
                    peak_pos = start + (end - start) // 4
                    rise_indices = np.arange(start, peak_pos)
                    fall_indices = np.arange(peak_pos, end)

                    if len(rise_indices) > 0:
                        rise_profile = (
                            magnitude
                            * ((rise_indices - start) / (peak_pos - start)) ** 2
                        )
                        force[rise_indices] += rise_profile

                    if len(fall_indices) > 0:
                        fall_profile = magnitude * np.exp(
                            -3 * (fall_indices - peak_pos) / (end - peak_pos)
                        )
                        force[fall_indices] += fall_profile
                else:
                    # Symmetric profile
                    indices = np.arange(start, end)
                    if len(indices) > 0:
                        profile = magnitude * np.sin(
                            np.linspace(0, np.pi, len(indices))
                        )
                        force[indices] += profile

            try:
                start_idx, end_idx = analyzer.find_impact_boundaries(force)

                # Should find reasonable boundaries
                assert 0 <= start_idx < end_idx < len(force)

                # Should capture at least one of the peaks
                captured_any_peak = False
                for peak_start, peak_end, _ in case["peaks"]:
                    if not (end_idx < peak_start or start_idx > peak_end):
                        captured_any_peak = True
                        break

                if case["name"] != "weak_signal":  # Weak signal might not be detected
                    assert (
                        captured_any_peak
                    ), f"Failed to capture any peak in {case['name']}"

                print(f"Boundary stress test '{case['name']}': SUCCESS")

            except Exception as e:
                print(f"Boundary stress test '{case['name']}': FAILED - {e}")
                if case["name"] not in ["weak_signal", "short_impact"]:
                    # These cases might legitimately fail
                    raise

    def test_numerical_stability_stress(self):
        """Test numerical stability with edge cases."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        edge_cases = [
            # All zeros
            {"name": "all_zeros", "force": np.zeros(1000)},
            # All ones
            {"name": "all_ones", "force": np.ones(1000)},
            # Alternating values
            {
                "name": "alternating",
                "force": np.array([(-1) ** i for i in range(1000)]),
            },
            # Single spike
            {"name": "single_spike", "force": np.zeros(1000)},
            # Step function
            {
                "name": "step_function",
                "force": np.concatenate([np.zeros(500), np.ones(500) * 1000]),
            },
        ]

        # Modify single spike case
        edge_cases[3]["force"][500] = 10000

        for case in edge_cases:
            time_array = np.linspace(0, 0.1, len(case["force"]))

            try:
                result = analyzer.calculate_impulse_metrics(case["force"], time_array)

                # Should handle gracefully
                if "error" not in result:
                    # Results should be finite
                    assert np.isfinite(
                        result["total_impulse"]
                    ), f"Non-finite impulse in {case['name']}"
                    assert np.isfinite(
                        result["peak_force"]
                    ), f"Non-finite force in {case['name']}"
                    print(f"Numerical stability '{case['name']}': SUCCESS")
                else:
                    print(f"Numerical stability '{case['name']}': Failed gracefully")

            except Exception as e:
                print(f"Numerical stability '{case['name']}': Exception - {e}")
                # Some edge cases might legitimately fail
                if case["name"] not in ["all_zeros", "alternating"]:
                    raise


if __name__ == "__main__":
    # Run a quick performance check
    print("Running quick performance check...")

    # Simple timing test
    analyzer = ImpulseAnalyzer(material_code="STND")

    sizes = [1000, 10000, 100000]
    for size in sizes:
        time_array = np.linspace(0, 0.1, size)
        force = np.random.normal(0, 10, size)
        force[size // 3 : 2 * size // 3] = 1000

        start_time = time.time()
        result = analyzer.calculate_impulse_metrics(force, time_array)
        elapsed = time.time() - start_time

        print(f"{size:,} points: {elapsed:.4f}s")

    print("Performance check complete.")
