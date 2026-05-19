#!/bin/bash

# Model download script for minllama testing
# This script will download test models from Hugging Face

set -e

# Download directory
DOWNLOAD_DIR="/tmp"
LOG_FILE="/tmp/download_models.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Function to download a model with retry
download_model() {
    local url="$1"
    local output="$2"
    local name="$3"
    
    log "Downloading $name..."
    
    # Try multiple methods
    for method in "curl" "wget" "aria2c"; do
        log "Trying $method..."
        case $method in
            curl)
                if curl -L -f --progress-bar -o "$output" "$url"; then
                    log "✅ $name downloaded successfully using curl"
                    return 0
                fi
                ;;
            wget)
                if wget -q --show-progress -O "$output" "$url"; then
                    log "✅ $name downloaded successfully using wget"
                    return 0
                fi
                ;;
            aria2c)
                if aria2c -x 16 -s 16 -k 1M --check-certificate=false --file-allocation=none -o "$output" "$url"; then
                    log "✅ $name downloaded successfully using aria2c"
                    return 0
                fi
                ;;
        esac
        log "❌ $method failed for $name"
    done
    
    log "❌ All download methods failed for $name"
    return 1
}

# List of models to download
MODELS=(
    "google/gemma-2b-it/gemma-2b-it.Q4_0.gguf:/tmp/gemma-2b-it.gguf:Gemma-2B-IT"
    "Qwen/Qwen1.5-1.5B/qwen1.5-1.5b.Q4_0.gguf:/tmp/qwen-1.5b.gguf:Qwen1.5-1.5B"
    "TheBloke/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2.Q4_0.gguf:/tmp/mistral-7b.gguf:Mistral-7B-Instruct"
    "lmms-lab/llama-3-8b-instruct-v2/llama-3-8b-instruct-v2.Q4_0.gguf:/tmp/llama-3-8b.gguf:Llama-3-8B-Instruct"
)

# Check tools availability
log "Checking download tools..."
TOOLS_AVAILABLE=""
if command -v curl >/dev/null 2>&1; then TOOLS_AVAILABLE+=",curl"; fi
if command -v wget >/dev/null 2>&1; then TOOLS_AVAILABLE+=",wget"; fi
if command -v aria2c >/dev/null 2>&1; then TOOLS_AVAILABLE+=",aria2c"; fi

TOOLS_AVAILABLE=${TOOLS_AVAILABLE#,}
log "Available tools: $TOOLS_AVAILABLE"

if [ -z "$TOOLS_AVAILABLE" ]; then
    log "❌ No download tools available!"
    exit 1
fi

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

# Download models
DOWNLOAD_SUCCESS=0
TOTAL_MODELS=${#MODELS[@]}

for model in "${MODELS[@]}"; do
    IFS=':' read -r url output name <<< "$model"
    
    if [ -f "$output" ]; then
        log "ℹ️  $name already exists, skipping..."
        DOWNLOAD_SUCCESS=$((DOWNLOAD_SUCCESS + 1))
        continue
    fi
    
    if download_model "$url" "$output" "$name"; then
        DOWNLOAD_SUCCESS=$((DOWNLOAD_SUCCESS + 1))
    else
        log "❌ Failed to download $name"
    fi
done

# Summary
log "Download Summary:"
log "  Total models: $TOTAL_MODELS"
log "  Successfully downloaded: $DOWNLOAD_SUCCESS"
log "  Failed: $((TOTAL_MODELS - DOWNLOAD_SUCCESS))"

if [ $DOWNLOAD_SUCCESS -eq $TOTAL_MODELS ]; then
    log "🎉 All models downloaded successfully!"
    exit 0
else
    log "⚠️  Some models failed to download"
    exit 1
fi