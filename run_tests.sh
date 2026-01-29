#!/usr/bin/env bash
#
# run_tests.sh - Comprehensive test runner for Heterodyne Detector
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh cuda         # Run only CUDA tests
#   ./run_tests.sh cpu          # Run only CPU tests
#   ./run_tests.sh quick        # Run quick tests (skip slow ones)
#   ./run_tests.sh coverage     # Run with coverage report

set -e  # Exit on error

# Colors
RED='\033[0.31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Heterodyne Detector - Test Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest not found. Installing...${NC}"
    pip install pytest pytest-asyncio pytest-cov
fi

# Determine test mode
MODE="${1:-all}"

# Base pytest command
PYTEST_CMD="python -m pytest -v"

# Test selection based on mode
case "$MODE" in
    cuda)
        echo -e "${YELLOW}Running CUDA-specific tests...${NC}"
        PYTEST_CMD="$PYTEST_CMD -m 'not skipif' tests/test_cuda_streams.py"
        ;;
    cpu)
        echo -e "${YELLOW}Running CPU tests (CUDA disabled)...${NC}"
        export CUDA_VISIBLE_DEVICES=""
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
    quick)
        echo -e "${YELLOW}Running quick tests (skipping slow)...${NC}"
        PYTEST_CMD="$PYTEST_CMD -m 'not slow' tests/"
        ;;
    coverage)
        echo -e "${YELLOW}Running tests with coverage...${NC}"
        PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html --cov-report=term tests/"
        ;;
    streams)
        echo -e "${YELLOW}Running CUDA streams tests only...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/test_cuda_streams.py"
        ;;
    errors)
        echo -e "${YELLOW}Running error recovery tests only...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/test_error_recovery.py"
        ;;
    memory)
        echo -e "${YELLOW}Running memory leak tests only...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/test_memory_leaks.py"
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests only...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/test_integration_comprehensive.py"
        ;;
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [all|cuda|cpu|quick|coverage|streams|errors|memory|integration]"
        exit 1
        ;;
esac

# Print system info
echo ""
echo -e "${BLUE}System Information:${NC}"
python -c "
import torch
import sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
"
echo ""

# Run tests
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Running Tests${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Execute
if $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ TESTS FAILED${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    exit 1
fi
