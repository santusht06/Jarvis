#!/bin/bash
# Start the AI README Maintenance Bot
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/bot.py"
