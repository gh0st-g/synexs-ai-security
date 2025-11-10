#!/usr/bin/env python3
"""
Synexs Core Orchestrator v2.1
Coordinates all cells, integrates AI decisions, manages pipeline
"""

import os
import sys
import time
import subprocess
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json
import signal
import psutil
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from synexs_model import load_model, predict_action, ACTIONS
    AI_MODEL_AVAILABLE = True
except ImportError:
    logging.error(f"⚠️ AI model not available")
    AI_MODEL_AVAILABLE = False

# Configuration
WORK_DIR = Path("/root/synexs") if Path("/root/synexs").exists() else Path("/app")
CELLS_DIR = WORK_DIR / "cells"
LOG_FILE = WORK_DIR / "synexs_core.log"
CYCLE_INTERVAL = 60  # seconds between cycles
TIMEOUT = 600  # cell execution timeout in seconds

# Cell execution order (phases)
# NOTE: cell_002.py is a continuous generator (infinite loop) - runs separately as standalone service
CELL_PHASES = {
    "generation": ["cell_001.py"],  # cell_002.py runs independently
    "processing": ["cell_004.py", "cell_010_parser.py"],
    "classification": ["cell_006.py"],
    "evolution": ["cell_014_mutator.py", "cell_015_replicator.py"],
    "feedback": ["cell_016_feedback_loop.py"],
}

# Setup logging with rotation
handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=50*1024*1024,  # 50MB
    backupCount=3
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logging.basicConfig(handlers=[handler], level=logging.INFO)

# ==================== Cell Execution ====================
class CellExecutor:
    """Manages cell execution"""

    def __init__(self):
        self.stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
        self.model = None
        self.vocab = None

        # Load AI model if available
        if AI_MODEL_AVAILABLE:
            try:
                self.model, self.vocab = load_model()
                logging.info(f"✅ AI model loaded (vocab size: {len(self.vocab)})")
            except Exception as e:
                logging.error(f"⚠️ Could not load AI model: {e}", exc_info=True)

    def execute_cell(self, cell_path: Path) -> Dict:
        """
        Execute a single cell script
        Args:
            cell_path: Path to cell script
        Returns:
            Dict with execution results
        """
        cell_name = cell_path.name
        start_time = time.time()

        # Check if cell exists
        if not cell_path.exists():
            logging.warning(f"⏭️ Skipping {cell_name} (not found)")
            self.stats["skipped"] += 1
            return {"status": "skipped", "reason": "not_found"}

        try:
            # Execute cell
            result = subprocess.run(
                ["python3", str(cell_path)],
                cwd=str(WORK_DIR),
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )

            duration = time.time() - start_time
            self.stats["total_executions"] += 1

            if result.returncode == 0:
                self.stats["successful"] += 1
                logging.info(f"✅ {cell_name} completed ({duration:.2f}s)")
                return {
                    "status": "success",
                    "duration": duration,
                    "output": result.stdout[-200:] if result.stdout else ""
                }
            else:
                self.stats["failed"] += 1
                logging.error(f"❌ {cell_name} failed: {result.stderr[:200]}")
                return {
                    "status": "failed",
                    "duration": duration,
                    "error": result.stderr[:200]
                }

        except subprocess.TimeoutExpired:
            self.stats["failed"] += 1
            logging.error(f"⏱️ {cell_name} timeout after {TIMEOUT}s")
            return {"status": "timeout", "timeout": TIMEOUT}

        except Exception as e:
            self.stats["failed"] += 1
            logging.error(f"❌ {cell_name} error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def execute_phase(self, phase_name: str, cells: List[str]) -> Dict:
        """
        Execute all cells in a phase
        Args:
            phase_name: Name of the phase
            cells: List of cell filenames
        Returns:
            Phase execution summary
        """
        logging.info(f"📋 Phase: {phase_name}")
        results = {}

        for cell_name in cells:
            cell_path = CELLS_DIR / cell_name
            result = self.execute_cell(cell_path)
            results[cell_name] = result

        successes = sum(1 for r in results.values() if r["status"] == "success")
        logging.info(f"✅ Phase '{phase_name}' complete: {successes}/{len(cells)} succeeded")

        return {"phase": phase_name, "cells": results, "success_count": successes}

    def run_cycle(self, cycle_num: int):
        """Execute one full orchestration cycle"""
        logging.info("=" * 60)
        logging.info(f"🔄 Cycle #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 60)

        cycle_start = time.time()
        phase_results = {}

        # Execute each phase in order
        for phase_name, cells in CELL_PHASES.items():
            phase_result = self.execute_phase(phase_name, cells)
            phase_results[phase_name] = phase_result

        # Log summary
        cycle_duration = time.time() - cycle_start
        total_cells = sum(len(cells) for cells in CELL_PHASES.values())
        successful = sum(pr["success_count"] for pr in phase_results.values())

        logging.info("=" * 60)
        logging.info(f"📊 Cycle #{cycle_num} Summary:")
        logging.info(f"   Duration: {cycle_duration:.2f}s")
        logging.info(f"   Cells: {successful}/{total_cells} succeeded")
        logging.info(f"   Cumulative: {self.stats['successful']} successes, {self.stats['failed']} failures")
        logging.info("=" * 60)

        return phase_results

    def get_stats(self) -> Dict:
        """Get execution statistics"""
        return self.stats.copy()

# ==================== Health Monitoring ====================
def check_system_health() -> Dict:
    """Check system health metrics"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"❌ Error checking system health: {e}", exc_info=True)
        return {"error": str(e)}

# ==================== Main Loop ====================
def main():
    """Main orchestrator loop"""
    print("=" * 60)
    print("🧠 Synexs Core Orchestrator v2.1")
    print("=" * 60)
    print(f"📁 Work directory: {WORK_DIR}")
    print(f"🔬 Cells directory: {CELLS_DIR}")
    print(f"📝 Log file: {LOG_FILE}")
    print(f"⏱️  Cycle interval: {CYCLE_INTERVAL}s")
    print(f"🤖 AI model: {'✅ Available' if AI_MODEL_AVAILABLE else '❌ Not available'}")
    print("=" * 60)

    # Verify cells directory exists
    if not CELLS_DIR.exists():
        logging.error(f"❌ Cells directory not found: {CELLS_DIR}")
        sys.exit(1)

    # Count available cells
    available_cells = list(CELLS_DIR.glob("*.py"))
    print(f"📦 Found {len(available_cells)} cell files")

    # Initialize executor
    executor = CellExecutor()
    logging.info(f"🚀 Orchestrator started - PID: {os.getpid()}")

    cycle_count = 0

    def handle_signal(signum, frame):
        logging.info("⛔ Shutdown requested")
        print("\n⛔ Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            cycle_count += 1

            # Run orchestration cycle
            try:
                executor.run_cycle(cycle_count)
            except Exception as e:
                logging.error(f"❌ Error running orchestration cycle: {e}", exc_info=True)

            # Health check every 10 cycles
            if cycle_count % 10 == 0:
                try:
                    health = check_system_health()
                    if "cpu_percent" in health:
                        logging.info(f"💊 Health: CPU={health['cpu_percent']:.1f}% MEM={health['memory_percent']:.1f}% DISK={health['disk_percent']:.1f}%")
                except Exception as e:
                    logging.error(f"❌ Error checking system health: {e}", exc_info=True)

            # Sleep until next cycle
            print(f"\n⏸️  Sleeping {CYCLE_INTERVAL}s until next cycle...")
            time.sleep(CYCLE_INTERVAL)

    except KeyboardInterrupt:
        logging.info("⛔ Shutdown requested")
        print("\n⛔ Shutting down...")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        stats = executor.get_stats()
        logging.info(f"📊 Final stats: {stats}")
        print(f"\n📊 Total executions: {stats['total_executions']}")
        print(f"✅ Successful: {stats['successful']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"⏭️  Skipped: {stats['skipped']}")

if __name__ == "__main__":
    main()