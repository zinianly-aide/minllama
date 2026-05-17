#!/bin/bash

# minllama Benchmark Runner
# Usage: ./scripts/run_benchmarks.sh [benchmark_type]
#   benchmark_type: all | attention | kv-tiling | smol-model (default: all)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build-release"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Build directory not found. Please run 'cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release' first${NC}"
    exit 1
fi

# Benchmark types
BENCH_ALL="attention_simd_bench kv_read_tiling_bench"
BENCH_ATTENTION="attention_simd_bench"
BENCH_KV="kv_read_tiling_bench"
BENCH_SMOLOLM="minllama_bench"

# Parse arguments
BENCHMARK_TYPE="${1:-all}"

echo -e "${GREEN}=== minllama Benchmark Runner ===${NC}"
echo "Build directory: $BUILD_DIR"
echo "Benchmark type: $BENCHMARK_TYPE"
echo ""

# Run specific benchmark or all
if [ "$BENCHMARK_TYPE" = "all" ]; then
    echo -e "${YELLOW}Running all benchmarks...${NC}"
    echo ""

    # Attention SIMD benchmarks
    echo "1. Attention SIMD Benchmark"
    echo "-------------------------"
    "$BUILD_DIR"/attention_simd_bench || true
    echo ""

    # KV tiling benchmarks
    echo "2. KV Read Tiling Benchmark"
    echo "---------------------------"
    "$BUILD_DIR"/kv_read_tiling_bench || true
    echo ""

    # SmolLM benchmarks
    echo "3. SmolLM Model Benchmark"
    echo "-------------------------"
    "$BUILD_DIR"/minllama_bench --help || true
    echo ""

elif [ "$BENCHMARK_TYPE" = "attention" ]; then
    echo -e "${YELLOW}Running attention SIMD benchmark...${NC}"
    echo ""
    "$BUILD_DIR"/attention_simd_bench || true

elif [ "$BENCHMARK_TYPE" = "kv-tiling" ]; then
    echo -e "${YELLOW}Running KV tiling benchmark...${NC}"
    echo ""
    "$BUILD_DIR"/kv_read_tiling_bench || true

elif [ "$BENCHMARK_TYPE" = "smol-model" ]; then
    echo -e "${YELLOW}Running SmolLM benchmark...${NC}"
    echo ""
    "$BUILD_DIR"/minllama_bench --help || true

else
    echo -e "${RED}Unknown benchmark type: $BENCHMARK_TYPE${NC}"
    echo "Usage: $0 [all | attention | kv-tiling | smol-model]"
    exit 1
fi

echo -e "${GREEN}=== Benchmark run complete ===${NC}"
