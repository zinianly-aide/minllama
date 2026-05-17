# P3-1: Benchmark CI - Quick Start Guide

## What Was Implemented

✅ **GitHub Actions CI Workflows**
- `ci.yml`: Main CI pipeline (tests, benchmarks, code quality)
- `benchmark-ci.yml`: Performance regression detection

✅ **Local Benchmark Tools**
- `scripts/run_benchmarks.sh`: Run all benchmarks
- `scripts/compare_benchmarks.py`: Compare benchmark results
- `scripts/README.md`: Script documentation

✅ **Documentation**
- `docs/p3-1-benchmark-ci.md`: Comprehensive CI guide
- `docs/p3-1-quickstart.md`: This file

## How to Use

### 1. Set Up Locally

```bash
cd minllama
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j$(nproc)
```

### 2. Run All Benchmarks

```bash
./scripts/run_benchmarks.sh
```

This runs:
- Attention SIMD benchmark
- KV tiling benchmark
- SmolLM model benchmark

### 3. Compare Results

```bash
# Create baseline
./scripts/run_benchmarks.sh
mkdir benchmark-results/baseline
mv benchmark-results/*.json benchmark-results/baseline/

# Make changes and run again
./scripts/run_benchmarks.sh
mkdir benchmark-results/current
mv benchmark-results/*.json benchmark-results/current/

# Compare
python3 scripts/compare_benchmarks.py --dir benchmark-results/baseline benchmark-results/current
```

### 4. Check CI on GitHub

After pushing code, check:
1. Go to GitHub Actions tab
2. Look for "CI" workflow
3. Verify all tests pass
4. Check benchmark comparison if applicable

## Performance Gates

**Critical Threshold**: >5% performance regression

The CI system will:
- ✅ Allow if improvement >5%
- ⚠️ Warning if regression <5%
- ❌ Fail if regression >5%

## Common Commands

### Build
```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j$(nproc)
```

### Run Tests
```bash
ctest --test-dir build-release --output-on-failure
```

### Run Specific Benchmark
```bash
./build-release/attention_simd_bench
./build-release/kv_read_tiling_bench
./build-release/minllama_bench --help
```

### Compare Two Files
```bash
python3 scripts/compare_benchmarks.py old.json new.json
```

## Troubleshooting

### Build fails
```bash
rm -rf build-release
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j$(nproc)
```

### Benchmark fails
- Check model file exists
- Verify dependencies are installed
- Ensure sufficient memory

### Comparison issues
- Check JSON files are valid
- Ensure same parameters used
- Verify both files have same structure

## Next Steps

1. **Enable GitHub Workflows**
   - Push this branch to GitHub
   - Enable Actions for the repository

2. **Test CI Locally**
   - Run `scripts/test_ci.sh quick` (when implemented)
   - Verify workflows execute correctly

3. **Add Benchmarks**
   - Document any new benchmarks
   - Update CI workflows if needed
   - Add to run_benchmarks.sh

4. **Establish Baseline**
   - Run benchmarks on main branch
   - Store results in benchmark-results/baseline/
   - Use as reference for future comparisons

## Support

For detailed documentation, see:
- `docs/p3-1-benchmark-ci.md` - Full CI infrastructure guide
- `scripts/README.md` - Script documentation

## Status

✅ Implementation complete
✅ Documentation complete
🔄 Ready for testing and deployment
