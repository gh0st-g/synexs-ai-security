#!/bin/bash
###############################################################################
# Synexs Biological Organism Startup Script
#
# This script launches the complete biological organism that manages:
# - All Synexs processes as "cells"
# - Adaptive immune system for threats
# - Genetic evolution of agents
# - Metabolic resource management
# - Organism lifecycle (aging, reproduction, evolution)
#
# Usage:
#   ./start_biological_organism.sh
#
# The organism will:
# 1. Stop any existing Synexs processes
# 2. Initialize biological systems
# 3. Start all managed processes
# 4. Begin organism lifecycle
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  SYNEXS BIOLOGICAL ORGANISM - STARTUP"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Working Directory: $SCRIPT_DIR"
echo "Python: $(which python3)"
echo "Virtual Env: $SCRIPT_DIR/synexs_env/bin/python3"
echo ""

# Check if biological system files exist
if [ ! -f "synexs_main_biological.py" ]; then
    echo "❌ ERROR: synexs_main_biological.py not found"
    echo "   Please ensure biological system files are present"
    exit 1
fi

# Check required biological modules
for module in synexs_biological_organism.py synexs_genetic_recombination.py synexs_adaptive_immune_system.py synexs_cell_differentiation.py synexs_metabolism_engine.py; do
    if [ ! -f "$module" ]; then
        echo "⚠️  WARNING: $module not found"
    fi
done

echo "✓ Biological system files present"
echo ""

# Stop existing Synexs processes
echo "🛑 Stopping existing Synexs processes..."

# Stop honeypot
if pgrep -f "honeypot_server.py" > /dev/null; then
    echo "  Stopping honeypot_server..."
    pkill -f "honeypot_server.py" || true
fi

# Stop swarm
if pgrep -f "ai_swarm_fixed.py" > /dev/null; then
    echo "  Stopping ai_swarm_fixed..."
    pkill -f "ai_swarm_fixed.py" || true
fi

# Stop old biological organism if running
if pgrep -f "synexs_main_biological.py" > /dev/null; then
    echo "  Stopping old biological organism..."
    pkill -f "synexs_main_biological.py" || true
fi

sleep 2
echo "✓ Existing processes stopped"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p datasets/honeypot
mkdir -p datasets/genomes
mkdir -p datasets/agents
echo "✓ Directories ready"
echo ""

# Start the biological organism
echo "════════════════════════════════════════════════════════════════════"
echo "  STARTING BIOLOGICAL ORGANISM"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Use virtual environment python
PYTHON_BIN="$SCRIPT_DIR/synexs_env/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "⚠️  Virtual environment not found, using system python3"
    PYTHON_BIN="python3"
fi

# Start organism in foreground (use screen/tmux for background)
$PYTHON_BIN synexs_main_biological.py

# Note: Script will not reach here unless organism stops
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  BIOLOGICAL ORGANISM STOPPED"
echo "════════════════════════════════════════════════════════════════════"
