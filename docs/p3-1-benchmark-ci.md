# P3-1: Benchmark CI Infrastructure

## Overview

This document describes the Benchmark CI infrastructure setup for minllama project.

## Purpose

Automated testing and benchmarking infrastructure to:
1. Ensure code correctness with comprehensive test suite
2. Detect performance regressions automatically
3. Track benchmark results over time
4. Enable CI/CD pipeline for quality assurance

## Infrastructure Components

### 1. GitHub Actions Workflows

#### `ci.yml` - Main CI Workflow

Runs on every push and pull request:
- **Test Suite**: Executes all unit tests with Debug and Release builds
- **Benchmark Suite**: Runs all benchmark tools
- **Code Quality**: Checks formatting and static analysis

**Coverage Matrix**:
- GCC 11 and 12
- Debug and Release builds
- Ubuntu latest

#### `benchmark-ci.yml` - Benchmark CI Workflow

Dedicated workflow for performance regression detection:
- **Benchmark Comparison**: Compares results between branches
- **Performance Tracking**: Maintains baseline metrics
- **Artifact Upload**: Saves benchmark results for review

**Key Features**:
- Downloads SmolLM model for real-world testing
- Compares current branch vs baseline (main/develop)
- Detects >5% performance regressions
- Uploads results as artifacts (30-day retention)

### 2. Local Benchmark Tools

#### `scripts/run_benchmarks.sh`

Convenient script to run all benchmarks locally:

```bash
./scripts/run_benchmarks.sh [all|attention|kv-tiling|smol-model]
```

**Supported Benchmarks**:
- `attention_simd_bench`: Attention kernel performance
- `kv_read_tiling_bench`: KV cache read optimization testing
- `minllama_bench`: End-to-end model generation performance

#### `scripts/compare_benchmarks.py`

Python script to compare benchmark results:

```bash
# Compare two JSON files
python3 scripts/compare_benchmarks.py baseline.json current.json

# Compare two directories
python3 scripts/compare_benchmarks.py --dir baseline/ current/

# Compare git branches
python3 scripts/compare_benchmarks.py --branch main feature/x

# Compare specific metric
python3 scripts/compare_benchmarks.py baseline.json current.json --metric tokens_per_sec
```

### 3. Benchmark Results Storage

Results are stored in:
- `.github/workflows/benchmark-ci.yml`: Workflow-generated artifacts
- `benchmark-results/`: Local benchmark storage
- JSON format for machine-readable comparison

## Running Benchmarks Locally

### Prerequisites

```bash
# Install dependencies
sudo apt-get install cmake g++ libssl-dev wget

# Build project
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j$(nproc)
```

### Running Benchmarks

```bash
# Run all benchmarks
./scripts/run_benchmarks.sh

# Run specific benchmark type
./scripts/run_benchmarks.sh attention
./scripts/run_benchmarks.sh kv-tiling
./scripts/run_benchmarks.sh smol-model
```

### Comparing Results

```bash
# After running benchmarks, compare to baseline
python3 scripts/compare_benchmarks.py --dir benchmark-results/baseline benchmark-results/current
```

## Performance Gate

### Regression Threshold

- **Warning**: >5% performance regression in any metric
- **Fail**: Benchmark CI marked as failed

### Metrics Tracked

1. **Perplexity**: Lower is better
2. **Tokens Per Second**: Higher is better
3. **Attention Kernel Time**: Lower is better
4. **KV Cache Read Time**: Lower is better

## Integration with Development Workflow

### For Codex Tasks

When implementing a new optimization:

1. **Implement**: Make changes to codebase
2. **Test**: Run test suite locally
   ```bash
   ctest --test-dir build-release --output-on-failure
   ```
3. **Benchmark**: Run benchmarks
   ```bash
   ./scripts/run_benchmarks.sh
   ```
4. **Compare**: Check for regressions
   ```bash
   python3 scripts/compare_benchmarks.py --dir benchmark-results/old benchmark-results/new
   ```
5. **Commit**: If no regressions detected
6. **Push**: Triggers CI
7. **Review**: Check CI results in GitHub Actions

### CI Checks

CI will verify:
- ✅ All tests pass
- ✅ Code formatting follows conventions
- ✅ No performance regressions detected
- ✅ Static analysis clean

## Benchmark Tools

### 1. `minllama_bench`

End-to-end model generation benchmark.

**Usage**:
```bash
./minllama_bench \
  --model <model_path> \
  --prompt "Your prompt" \
  --max-new-tokens <n> \
  --threads <1|2> \
  --output <output.json>
```

**Metrics**:
- Perplexity
- Tokens per second
- Generation time

### 2. `attention_simd_bench`

Attention kernel performance benchmark.

**Usage**:
```bash
./attention_simd_bench
```

**Metrics**:
- Kernel execution time
- Memory bandwidth utilization

### 3. `kv_read_tiling_bench`

KV cache read optimization benchmark.

**Usage**:
```bash
./kv_read_tiling_bench
```

**Metrics**:
- QK read time
- AV read time
- Total attention-read time
- Speedup over baseline

## Troubleshooting

### Build Issues

```bash
# Clean and rebuild
rm -rf build-release
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j$(nproc)
```

### Benchmark Failures

- Check model file exists and is valid
- Verify all dependencies are installed
- Check memory availability (benchmarks can be memory-intensive)

### Comparison Issues

- Ensure JSON files are valid and have same structure
- Check that both runs used same parameters
- Verify model and tokenizer are identical

## Maintenance

### Updating Baseline

To update baseline from main branch:
```bash
git checkout main
./scripts/run_benchmarks.sh
mv benchmark-results benchmark-results-new
git checkout -p feature/x
mv benchmark-results-new benchmark-results
```

### Adding New Benchmarks

1. Add benchmark executable to CMakeLists.txt
2. Add benchmark to run_benchmarks.sh
3. Update CI workflows if needed
4. Document new benchmark in this file

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CMake Testing](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [Performance Testing Best Practices](https://github.com/github/octo-best-practices-performance-testing)

## Status

- ✅ CI workflow created (`ci.yml`)
- ✅ Benchmark CI workflow created (`benchmark-ci.yml`)
- ✅ Local benchmark runner script created
- ✅ Benchmark comparison tool created
- ✅ Documentation completed
- 🔄 Next: Enable workflows and test
