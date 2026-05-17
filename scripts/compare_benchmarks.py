#!/usr/bin/env python3
"""
minllama Benchmark Comparison Tool

Usage:
  compare_benchmarks.py baseline.json current.json
  compare_benchmarks.py --dir baseline_dir current_dir
  compare_benchmarks.py --baseline-branch main --current-branch feature/x
"""

import json
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any
import statistics


class BenchmarkResult:
    def __init__(self, data: Dict[str, Any], source: str):
        self.data = data
        self.source = source

    def get_perf_metric(self, metric_name: str) -> float:
        """Extract performance metric (e.g., perplexity, tokens_per_sec)"""
        if metric_name == "perplexity":
            # Extract from metrics.perplexity
            metrics = self.data.get("metrics", {})
            perplexity = metrics.get("perplexity", {})
            return perplexity.get("value", 0.0)
        elif metric_name == "tokens_per_sec":
            metrics = self.data.get("metrics", {})
            tokens_per_sec = metrics.get("tokens_per_sec", {})
            return tokens_per_sec.get("value", 0.0)
        return 0.0

    def get_speedup(self, other: 'BenchmarkResult', metric_name: str) -> float:
        """Calculate speedup/ improvement ratio"""
        value1 = self.get_perf_metric(metric_name)
        value2 = other.get_perf_metric(metric_name)

        if value1 == 0 or value2 == 0:
            return 0.0

        return value1 / value2

    def print_summary(self):
        """Print summary of benchmark results"""
        print(f"\n{self.source}:")
        print(f"  Perplexity: {self.get_perf_metric('perplexity')}")
        print(f"  Tokens/s: {self.get_perf_metric('tokens_per_sec')}")

    @staticmethod
    def from_json_file(filepath: str) -> 'BenchmarkResult':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return BenchmarkResult(data, filepath)


def compare_json_files(baseline_file: str, current_file: str, metric_name: str = "perplexity"):
    """Compare two JSON benchmark files"""
    baseline = BenchmarkResult.from_json_file(baseline_file)
    current = BenchmarkResult.from_json_file(current_file)

    print("=" * 50)
    print("Benchmark Comparison")
    print("=" * 50)

    baseline.print_summary()
    current.print_summary()

    speedup = baseline.get_speedup(current, metric_name)
    print(f"\nSpeedup: {speedup:.4f}x")

    # Interpret results
    if speedup < 0.95:
        print(f"⚠️  WARNING: Performance regression detected (>5%)")
        print(f"   Baseline: {baseline.get_perf_metric(metric_name)}")
        print(f"   Current:  {current.get_perf_metric(metric_name)}")
        return "FAILED"
    elif speedup > 1.05:
        print(f"✓ Performance improvement: {((speedup - 1) * 100):.2f}%")
        print(f"   Baseline: {baseline.get_perf_metric(metric_name)}")
        print(f"   Current:  {current.get_perf_metric(metric_name)}")
        return "PASSED"
    else:
        print(f"= No significant change")
        print(f"   Baseline: {baseline.get_perf_metric(metric_name)}")
        print(f"   Current:  {current.get_perf_metric(metric_name)}")
        return "UNCHANGED"


def compare_json_dirs(baseline_dir: str, current_dir: str, metric_name: str = "perplexity"):
    """Compare benchmark results from two directories"""
    baseline_file = Path(baseline_dir) / "baseline.json"
    current_file = Path(current_dir) / "baseline.json"

    if not baseline_file.exists():
        print(f"Error: {baseline_file} not found")
        return None

    if not current_file.exists():
        print(f"Error: {current_file} not found")
        return None

    return compare_json_files(str(baseline_file), str(current_file), metric_name)


def compare_branches(baseline_branch: str, current_branch: str, metric_name: str = "perplexity"):
    """Compare benchmarks between two git branches"""
    # Store current branch
    original_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        text=True
    ).strip()

    try:
        # Checkout baseline branch
        print(f"Checking out {baseline_branch}...")
        subprocess.run(["git", "checkout", baseline_branch], check=True, capture_output=True)

        # Build baseline
        print(f"Building {baseline_branch}...")
        subprocess.run(["cmake", "-S", ".", "-B", "build-baseline", "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", "build-baseline", "-j", "$(nproc)"], check=True)

        # Run baseline benchmark
        print(f"Running benchmark for {baseline_branch}...")
        result_file = "benchmark-results/baseline/baseline.json"
        subprocess.run([
            "./build-baseline/minllama_bench",
            "--model", "models/SmolLM-1.7B-Instruct/model.safetensors",
            "--prompt", "The future of AI is exciting.",
            "--max-new-tokens", "50",
            "--threads", "2",
            "--output", result_file
        ], check=True)

        # Checkout current branch
        print(f"Checking out {current_branch}...")
        subprocess.run(["git", "checkout", current_branch], check=True, capture_output=True)

        # Build current
        print(f"Building {current_branch}...")
        subprocess.run(["cmake", "-S", ".", "-B", "build-release", "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", "build-release", "-j", "$(nproc)"], check=True)

        # Run current benchmark
        print(f"Running benchmark for {current_branch}...")
        result_file = "benchmark-results/current/baseline.json"
        subprocess.run([
            "./build-release/minllama_bench",
            "--model", "models/SmolLM-1.7B-Instruct/model.safetensors",
            "--prompt", "The future of AI is exciting.",
            "--max-new-tokens", "50",
            "--threads", "2",
            "--output", result_file
        ], check=True)

        # Compare results
        print(f"\nComparing {baseline_branch} vs {current_branch}...")
        return compare_json_files(
            "benchmark-results/baseline/baseline.json",
            "benchmark-results/current/baseline.json",
            metric_name
        )

    finally:
        # Restore original branch
        print(f"\nRestoring {original_branch}...")
        subprocess.run(["git", "checkout", original_branch], check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Compare minllama benchmark results")
    parser.add_argument("baseline", help="Baseline JSON file or directory")
    parser.add_argument("current", help="Current JSON file or directory")
    parser.add_argument("--metric", choices=["perplexity", "tokens_per_sec"], default="perplexity",
                        help="Metric to compare (default: perplexity)")
    parser.add_argument("--branch", action="store_true", help="Compare git branches instead of files")

    args = parser.parse_args()

    if args.branch:
        if len(sys.argv) < 4:
            print("Error: --branch requires two branch names")
            sys.exit(1)
        compare_branches(sys.argv[2], sys.argv[3], args.metric)
    else:
        if Path(args.baseline).is_dir() and Path(args.current).is_dir():
            result = compare_json_dirs(args.baseline, args.current, args.metric)
        else:
            result = compare_json_files(args.baseline, args.current, args.metric)


if __name__ == "__main__":
    main()
