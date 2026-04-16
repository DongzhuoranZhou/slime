#!/bin/bash

# Keep running search_agent.py until line count reaches 961
DATASET="mmlongdoc"
AGENT_MODEL_NAME="openai/gemini-3-pro-preview"
EMBED_DEVICE="cuda:3"

LOG_DIR="${AGENTIC_MEMORY_LOG_DIR:-/lc3T/AgenticMemory/logs}"
LOG_FILE="${LOG_DIR}/${DATASET}_${AGENT_MODEL_NAME#*/}.jsonl"

while true; do
    LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    echo "Current line count in $LOG_FILE: $LINE_COUNT"
    
    if [ "$LINE_COUNT" -ge 961 ]; then
        echo "Line count has reached 961 or more. Stopping."
        break
    fi
    
    echo "Line count is less than 961. Running search_agent.py..."
    python search_agent.py \
        --dataset "$DATASET" \
        --agent_model_name "$AGENT_MODEL_NAME" \
        --embed_device "$EMBED_DEVICE" 
    
    # Optional: Add a small delay between runs to avoid tight loops
    sleep 10
done