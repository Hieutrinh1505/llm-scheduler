#!/bin/bash

# vLLM Server Startup Script
# Reads configuration from .env file

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}vLLM Server Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Look for .env file in multiple locations
ENV_FILE=""
if [ -f "../.env" ]; then
    ENV_FILE="../.env"
elif [ -f "../../.env" ]; then
    ENV_FILE="../../.env"
elif [ -f ".env" ]; then
    ENV_FILE=".env"
fi

# Load environment variables from .env file
if [ -n "$ENV_FILE" ]; then
    echo -e "${GREEN}Loading configuration from: $ENV_FILE${NC}"

    # Export variables from .env file
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo -e "${YELLOW}Warning: No .env file found, using defaults${NC}"
fi

# Set defaults if not specified
LLM_MODEL=${LLM_MODEL:-"gpt2"}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-"0.0.0.0"}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.9}

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Model: $LLM_MODEL"
echo "  Port: $VLLM_PORT"
echo "  Host: $VLLM_HOST"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vLLM not installed${NC}"
    echo "Install with: pip install vllm"
    exit 1
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA available${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}! CUDA not available, using CPU (slow)${NC}"
fi

echo ""
echo -e "${GREEN}Starting vLLM server...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Start vLLM server using the newer 'vllm serve' command
vllm serve "$LLM_MODEL" \
    --port "$VLLM_PORT" \
    --host "$VLLM_HOST" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
