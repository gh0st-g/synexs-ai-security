#!/usr/bin/env python3
"""
Synexs Core Orchestrator v2.2
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
        self.ai_engine = None

        # Load AI model if available
        if AI_MODEL_AVAILABLE:
            try:
                self.model, self.vocab = load_model()
                logging.info(f"✅ AI model loaded (vocab size: {len(self.vocab)})")

                # Initialize AI Decision Engine
                self.ai_engine = AIDecisionEngine(self.model, self.vocab)
                logging.info(f"✅ AI Decision Engine initialized")

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

    def pre_classification_hook(self):
        """
        Pre-execution hook: Generate AI decisions before classification phase
        This runs BEFORE cell_006 classification in shadow mode
        """
        if not self.ai_engine:
            return

        try:
            # Load sequences from refined directory
            refined_dir = WORK_DIR / "datasets" / "refined"
            sequences = []

            if refined_dir.exists():
                for fname in refined_dir.glob("*.json"):
                    try:
                        with open(fname, 'r') as f:
                            data = json.load(f)
                        # Handle different formats
                        seq_data = data.get("sequences", []) if isinstance(data, dict) else data if isinstance(data, list) else []
                        for entry in seq_data:
                            if isinstance(entry, dict):
                                seq = entry.get("sequence") or entry.get("refined") or ""
                            elif isinstance(entry, str):
                                seq = entry
                            else:
                                continue
                            if seq:
                                sequences.append(seq)
                    except Exception as e:
                        logging.error(f"Error loading {fname}: {e}")

            if sequences:
                logging.info(f"🤖 Generating AI decisions for {len(sequences)} sequences...")
                decisions = self.ai_engine.generate_decisions(sequences)
                self.ai_engine.save_decisions_for_cells(decisions)

                # Log AI metrics
                sources = {}
                for d in decisions:
                    source = d.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1

                avg_confidence = sum(d.get("confidence", 0) for d in decisions) / len(decisions) if decisions else 0
                logging.info(f"📊 AI Metrics: {sources}, Avg Confidence: {avg_confidence:.3f}")
            else:
                logging.info("⏭️  No sequences found for AI decision generation")

        except Exception as e:
            logging.error(f"❌ Error in pre-classification hook: {e}", exc_info=True)

    def post_feedback_hook(self):
        """
        Post-execution hook: Collect training data after feedback phase
        This runs AFTER cell_016 feedback loop completes
        """
        if not self.ai_engine:
            return

        try:
            # Load feedback results to determine outcomes
            decisions_file = WORK_DIR / "datasets" / "decisions" / "decisions.json"

            if decisions_file.exists():
                with open(decisions_file, 'r') as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    decisions = data.get("decisions", [])
                elif isinstance(data, list):
                    decisions = data
                else:
                    decisions = []

                # Collect training samples from decisions
                for decision in decisions[:10]:  # Limit to avoid overwhelming
                    if isinstance(decision, dict):
                        seq = decision.get("sequence", "")
                        action = decision.get("action", decision.get("decision", ""))
                        if seq and action:
                            # Use action as both prediction and outcome for now
                            # In production, outcome would come from actual execution results
                            self.ai_engine.collect_training_sample(seq, action, action)

                logging.info(f"📊 Collected {min(len(decisions), 10)} training samples")

        except Exception as e:
            logging.error(f"❌ Error in post-feedback hook: {e}", exc_info=True)

    def run_cycle(self, cycle_num: int):
        """Execute one full orchestration cycle with AI integration"""
        logging.info("=" * 60)
        logging.info(f"🔄 Cycle #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 60)

        cycle_start = time.time()
        phase_results = {}

        # Execute each phase in order
        for phase_name, cells in CELL_PHASES.items():
            # Pre-classification hook: Generate AI decisions
            if phase_name == "classification":
                self.pre_classification_hook()

            phase_result = self.execute_phase(phase_name, cells)
            phase_results[phase_name] = phase_result

            # Post-feedback hook: Collect training data
            if phase_name == "feedback":
                self.post_feedback_hook()

        # Check for retraining trigger
        if self.ai_engine and self.ai_engine.should_retrain(cycle_num):
            logging.info(f"🔄 Retraining trigger: cycle {cycle_num}")
            self.ai_engine.trigger_retraining()

        # Log summary
        cycle_duration = time.time() - cycle_start
        total_cells = sum(len(cells) for cells in CELL_PHASES.values())
        successful = sum(pr["success_count"] for pr in phase_results.values())

        logging.info("=" * 60)
        logging.info(f"📊 Cycle #{cycle_num} Summary:")
        logging.info(f"   Duration: {cycle_duration:.2f}s")
        logging.info(f"   Cells: {successful}/{total_cells} succeeded")
        logging.info(f"   Cumulative: {self.stats['successful']} successes, {self.stats['failed']} failures")

        # Log AI stats if available
        if self.ai_engine:
            logging.info(f"   Training buffer: {len(self.ai_engine.training_buffer)} samples")

        logging.info("=" * 60)

        return phase_results

    def get_stats(self) -> Dict:
        """Get execution statistics"""
        return self.stats.copy()

# ==================== AI Decision Engine ====================
class AIDecisionEngine:
    """
    AI-powered decision engine with shadow mode support
    Makes predictions using the trained model with fallback to rules
    """

    def __init__(self, model, vocab, config_path="ai_config.json"):
        self.model = model
        self.vocab = vocab
        self.config = self._load_config(config_path)
        self.training_buffer = []
        self.decisions_log = WORK_DIR / "ai_decisions_log.jsonl"
        self.ai_decisions_dir = WORK_DIR / "datasets" / "ai_decisions"
        self.ai_decisions_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"🤖 AI Decision Engine initialized")
        logging.info(f"   Shadow mode: {self.config['ai_mode']['shadow_mode']}")
        logging.info(f"   Fallback enabled: {self.config['ai_mode']['fallback_enabled']}")
        logging.info(f"   Auto-retrain: {self.config['training']['auto_retrain']}")

    def _load_config(self, config_path):
        """Load AI configuration"""
        try:
            with open(WORK_DIR / config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load AI config, using defaults: {e}")
            return {
                "ai_mode": {"enabled": True, "shadow_mode": True, "fallback_enabled": True},
                "training": {"auto_retrain": True, "retrain_every_n_cycles": 100}
            }

    def predict_with_confidence(self, sequence: str) -> tuple:
        """
        Predict action for a sequence with confidence score
        Returns: (action, confidence, source)
        """
        if not self.model or not self.vocab:
            return self._rule_based_decision(sequence), 0.0, "rule"

        try:
            import torch
            tokens = sequence.split()
            unk_idx = self.vocab.get("<UNK>", 0)
            token_ids = [self.vocab.get(tok.upper(), unk_idx) for tok in tokens]
            x = torch.tensor([token_ids], dtype=torch.long)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                confidence = confidence.item()
                pred_idx = pred_idx.item()

            # Map index to action
            from synexs_model import IDX2ACTION
            action = IDX2ACTION.get(pred_idx, "SCAN")

            # Check confidence threshold
            threshold = self.config.get("confidence_threshold", 0.6)
            if confidence < threshold and self.config["ai_mode"]["fallback_enabled"]:
                fallback_action = self._rule_based_decision(sequence)
                logging.debug(f"Low confidence ({confidence:.2f}), using fallback: {fallback_action}")
                return fallback_action, confidence, "fallback"

            return action, confidence, "ai"

        except Exception as e:
            logging.error(f"AI prediction error: {e}")
            return self._rule_based_decision(sequence), 0.0, "error"

    def _rule_based_decision(self, sequence: str) -> str:
        """Fallback rule-based decision logic"""
        tokens = sequence.upper().split()

        # Simple heuristics
        if "ATTACK" in tokens or "EXPLOIT" in tokens:
            return "DEFEND"
        elif "MUTATE" in tokens:
            return "MUTATE"
        elif "REPLICATE" in tokens:
            return "REPLICATE"
        elif "EVADE" in tokens or "FLAG" in tokens:
            return "EVADE"
        else:
            return "SCAN"

    def generate_decisions(self, sequences: List[str]) -> List[Dict]:
        """
        Generate AI decisions for sequences
        Returns list of decisions with metadata
        """
        decisions = []

        for seq in sequences:
            action, confidence, source = self.predict_with_confidence(seq)

            decision = {
                "sequence": seq,
                "action": action,
                "confidence": confidence,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }

            decisions.append(decision)

            # Log decision
            self._log_decision(decision)

        return decisions

    def _log_decision(self, decision: Dict):
        """Log AI decision to audit trail"""
        try:
            with open(self.decisions_log, 'a') as f:
                f.write(json.dumps(decision) + "\n")
        except Exception as e:
            logging.error(f"Error logging AI decision: {e}")

    def save_decisions_for_cells(self, decisions: List[Dict]):
        """Save AI decisions for cells to consume"""
        output_file = self.ai_decisions_dir / "latest_decisions.json"
        try:
            with open(output_file, 'w') as f:
                json.dump({"decisions": decisions, "generated_at": datetime.now().isoformat()}, f, indent=2)
            logging.info(f"💾 Saved {len(decisions)} AI decisions to {output_file}")
        except Exception as e:
            logging.error(f"Error saving AI decisions: {e}")

    def collect_training_sample(self, sequence: str, predicted_action: str, actual_outcome: str):
        """Collect training sample from execution results"""
        sample = {
            "sequence": sequence,
            "predicted_action": predicted_action,
            "actual_outcome": actual_outcome,
            "timestamp": datetime.now().isoformat()
        }

        self.training_buffer.append(sample)

        # Save buffer periodically
        if len(self.training_buffer) >= 100:
            self._flush_training_buffer()

    def _flush_training_buffer(self):
        """Save training buffer to disk"""
        if not self.training_buffer:
            return

        training_dir = WORK_DIR / "datasets" / "core_training"
        training_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        output_file = training_dir / f"training_batch_{timestamp}.json"

        try:
            with open(output_file, 'w') as f:
                json.dump(self.training_buffer, f, indent=2)
            logging.info(f"💾 Saved {len(self.training_buffer)} training samples to {output_file}")
            self.training_buffer = []
        except Exception as e:
            logging.error(f"Error saving training buffer: {e}")

    def should_retrain(self, cycle_num: int) -> bool:
        """Check if retraining should be triggered"""
        if not self.config["training"]["auto_retrain"]:
            return False

        retrain_interval = self.config["training"].get("retrain_every_n_cycles", 100)
        return cycle_num > 0 and cycle_num % retrain_interval == 0

    def trigger_retraining(self):
        """Trigger model retraining"""
        logging.info("🔄 Triggering model retraining...")

        try:
            # Flush any remaining training samples
            self._flush_training_buffer()

            # Execute training cell
            trainer_path = CELLS_DIR / "cell_016_model_trainer.py"
            if trainer_path.exists():
                result = subprocess.run(
                    ["python3", str(trainer_path)],
                    cwd=str(WORK_DIR),
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if result.returncode == 0:
                    logging.info("✅ Model retraining completed successfully")
                    # Reload model
                    self.model, self.vocab = load_model()
                    logging.info("✅ Model reloaded with new weights")
                    return True
                else:
                    logging.error(f"❌ Model retraining failed: {result.stderr[:200]}")
                    return False
            else:
                logging.warning(f"⚠️  Trainer not found: {trainer_path}")
                return False

        except Exception as e:
            logging.error(f"❌ Error during retraining: {e}", exc_info=True)
            return False

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
    print("🧠 Synexs Core Orchestrator v2.2")
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