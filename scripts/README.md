# minllama Scripts

This directory contains utility scripts for development, testing, and benchmarking.

## Scripts

### `run_benchmarks.sh`

Convenient script to run all benchmarks locally.

**Usage**:
```bash
./run_benchmarks.sh [all|attention|kv-tiling|smol-model]
```

**Parameters**:
- `all`: Run all benchmark tools (default)
- `attention`: Only run attention SIMD benchmark
- `kv-tiling`: Only run KV tiling benchmark
- `smol-model`: Only run SmolLM model benchmark

**Requirements**:
- Project must be built in `build-release` directory
- Dependencies: cmake, g++, libssl-dev

### `compare_benchmarks.py`

Python script to compare benchmark results between versions or branches.

**Usage**:
```bash
# Compare two JSON files
python3 compare_benchmarks.py baseline.json current.json

# Compare two directories
python3 compare_benchmarks.py --dir baseline/ current/

# Compare git branches
python3 compare_benchmarks.py --branch main feature/x

# Compare specific metric
python3 compare_benchmarks.py baseline.json current.json --metric tokens_per_sec
```

**Features**:
- Extracts performance metrics from JSON results
- Calculates speedup/improvement ratios
- Detects performance regressions (>5% warning)
- Provides clear reporting

**Requirements**:
- Python 3.6+
- Two benchmark result JSON files

### `test_ci.sh` (To be created)

Script to run CI checks locally.

**Purpose**:
- Run all tests
- Check code formatting
- Run quick benchmarks
- Detect regressions

**Usage** (when implemented):
```bash
./test_ci.sh [quick|full]
```

## Benchmark Tools

The following benchmark executables are built alongside the project:

- `minllama_bench`: End-to-end model generation benchmark
- `attention_simd_bench`: Attention kernel performance benchmark
- `kv_read_tiling_bench`: KV cache read optimization benchmark

All tools are built to `build-release/` directory.

## Documentation

See `../docs/p3-1-benchmark-ci.md` for comprehensive CI infrastructure documentation.

## Contributing

When adding new scripts:
1. Follow existing naming conventions
2. Add usage documentation in this file
3. Update scripts/README.md if needed
4. Ensure scripts are executable
5. Add shebang line for bash/python scripts
6. Include error handling
