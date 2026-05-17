#!/bin/bash

# minllama CI Test Script
# Runs CI checks locally to verify setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build-release"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== minllama CI Test Suite ===${NC}"
echo ""

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found.${NC}"
    echo "Please run: cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release"
    exit 1
fi

# Run tests
echo -e "${YELLOW}1. Running tests...${NC}"
cd "$BUILD_DIR"
if ! ctest --output-on-failure --verbose; then
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Tests passed${NC}"
echo ""

# Check code formatting
echo -e "${YELLOW}2. Checking code formatting...${NC}
if [ -n "$(find $PROJECT_DIR -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run -Werror 2>&1)" ]; then
    echo -e "${RED}Code formatting issues found!${NC}"
    find $PROJECT_DIR -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run -Werror
    exit 1
fi
echo -e "${GREEN}✓ Code formatting OK${NC}"
echo ""

# Run quick benchmarks
echo -e "${YELLOW}3. Running quick benchmarks...${NC}
# Only run attention SIMD benchmark for quick check
if [ -f "$BUILD_DIR/attention_simd_bench" ]; then
    "$BUILD_DIR/attention_simd_bench" || true
    echo -e "${GREEN}✓ Benchmarks completed${NC}"
else
    echo -e "${YELLOW}⚠ Benchmarks not found (expected for quick test)${NC}"
fi
echo ""

echo -e "${GREEN}=== CI Check Complete ===${NC}"
echo "All checks passed! 🎉"
