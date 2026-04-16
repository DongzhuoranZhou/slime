#!/bin/bash

# Keep running search_agent.py until line count reaches 1153
DATASET="longdocurl"
AGENT_MODEL_NAME="openai/gemini-2.5-pro"
EMBED_DEVICE="cuda:3"

LOG_DIR="${AGENTIC_MEMORY_LOG_DIR:-/lc3T/AgenticMemory/logs}"
LOG_FILE="${LOG_DIR}/${DATASET}_${AGENT_MODEL_NAME#*/}.jsonl"

while true; do
    LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    echo "Current line count in $LOG_FILE: $LINE_COUNT"
    
    if [ "$LINE_COUNT" -ge 1153 ]; then
        echo "Line count has reached 1153 or more. Stopping."
        break
    fi
    
    echo "Line count is less than 1153. Running search_agent.py..."
    python search_agent.py \
        --dataset "$DATASET" \
        --agent_model_name "$AGENT_MODEL_NAME" \
        --embed_device "$EMBED_DEVICE" 
    
    # Optional: Add a small delay between runs to avoid tight loops
    sleep 10
done