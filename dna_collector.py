#!/usr/bin/env python3
"""
dna_collector.py - Synexs DNA Training Data Collector
Monitors system events and generates training samples from real operations
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque
import hashlib
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_protocol import encode_base64, generate_binary_training_data

# Configuration
HONEYPOT_LOG = "datasets/honeypot/attacks.json"
MEMORY_LOG = "memory_log.json"
CELL_OUTPUTS = "datasets/generated"
TRAINING_OUTPUT = "training_binary_v3.jsonl"
STATE_FILE = ".dna_collector_state.json"
EVENTS_THRESHOLD = 100
SAMPLES_PER_RUN = 50

# Action mapping from events
EVENT_TO_ACTIONS = {
    "ssh_attack": ["SCAN", "ATTACK", "REPORT"],
    "honeypot_detected": ["SCAN", "EVADE", "LEARN"],
    "agent_spawn": ["REPLICATE", "MUTATE"],
    "mutation": ["MUTATE", "REFINE"],
    "kill_report": ["LEARN", "MUTATE", "REPLICATE"],
    "cell_execution": ["SCAN", "REFINE"],
    "classifier": ["FLAG", "REFINE"],
    "replication": ["REPLICATE", "REPORT"],
    "defense": ["DEFEND", "SCAN"],
}

class DNACollector:
    def __init__(self):
        self.state = self.load_state()
        self.new_events = 0
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger("dna_collector")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler("dna_collector.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def load_state(self):
        """Load last run state"""
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "last_run": 0,
                "honeypot_offset": 0,
                "memory_offset": 0,
                "cell_files_seen": [],
                "total_events": 0,
                "total_samples": 0
            }

    def save_state(self):
        """Save current state"""
        self.state["last_run"] = int(time.time())
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def watch_honeypot(self):
        """Monitor honeypot attack logs"""
        events = []
        if not os.path.exists(HONEYPOT_LOG):
            return events

        try:
            with open(HONEYPOT_LOG) as f:
                lines = f.readlines()

            new_lines = lines[self.state["honeypot_offset"]:]
            self.state["honeypot_offset"] = len(lines)

            for line in new_lines:
                try:
                    data = json.loads(line.strip())
                    if "source_ip" in data:
                        events.append({
                            "type": "ssh_attack",
                            "data": data,
                            "timestamp": time.time()
                        })
                except (ValueError, KeyError):
                    self.logger.warning("Error parsing honeypot log line: %s", line)
        except (FileNotFoundError, IOError):
            self.logger.error("Error reading honeypot log file: %s", HONEYPOT_LOG)

        return events

    def watch_memory_log(self):
        """Monitor memory/learning logs"""
        events = []
        if not os.path.exists(MEMORY_LOG):
            return events

        try:
            with open(MEMORY_LOG) as f:
                data = json.load(f)

            if isinstance(data, list):
                new_items = data[self.state["memory_offset"]:]
                self.state["memory_offset"] = len(data)

                for item in new_items:
                    events.append({
                        "type": "kill_report",
                        "data": item,
                        "timestamp": time.time()
                    })
        except (FileNotFoundError, IOError, ValueError):
            self.logger.error("Error reading memory log file: %s", MEMORY_LOG)

        return events

    def watch_cell_outputs(self):
        """Monitor cell generation outputs"""
        events = []
        if not os.path.exists(CELL_OUTPUTS):
            return events

        try:
            files = list(Path(CELL_OUTPUTS).glob("*.json"))

            for fpath in files:
                fname = str(fpath)
                if fname not in self.state["cell_files_seen"]:
                    self.state["cell_files_seen"].append(fname)

                    with open(fpath) as f:
                        data = json.load(f)

                    events.append({
                        "type": "cell_execution",
                        "data": {"file": fname, "sequences": len(data.get("sequences", []))},
                        "timestamp": fpath.stat().st_mtime
                    })
        except (FileNotFoundError, IOError, ValueError):
            self.logger.error("Error reading cell output files: %s", CELL_OUTPUTS)

        return events

    def event_to_actions(self, event):
        """Convert event to action sequence"""
        event_type = event["type"]
        base_actions = EVENT_TO_ACTIONS.get(event_type, ["SCAN", "REPORT"])

        # Add context-aware actions
        data = event.get("data", {})

        if "ptr_valid" in data and not data["ptr_valid"]:
            base_actions.insert(0, "EVADE")

        if "username" in data and data["username"] == "root":
            base_actions.append("FLAG")

        if "mutation" in str(data):
            base_actions.append("MUTATE")

        return base_actions[:8]  # Max 8 actions

    def generate_samples_from_events(self, events):
        """Generate training samples from real events"""
        samples = []

        for event in events[:SAMPLES_PER_RUN]:
            actions = self.event_to_actions(event)

            if not actions:
                continue

            # Encode to binary
            try:
                encoded = encode_base64(actions)
            except Exception as e:
                self.logger.error("Error encoding actions to binary: %s", e)
                continue

            # Create instruction
            event_desc = event["type"].replace("_", " ").title()

            # Generate description
            if "attack" in event["type"]:
                description = f"Detected {event_desc}. Execute defense sequence: {' → '.join(actions[:3])}"
            elif "honeypot" in event["type"]:
                description = f"Honeypot identified. Evasion protocol: {' → '.join(actions[:3])}"
            elif "kill" in event["type"]:
                description = f"Agent terminated. Learning from failure: {', '.join(actions[:3])}"
            elif "spawn" in event["type"] or "replicate" in event["type"]:
                description = f"Spawning new agent with mutations. Chain: {' → '.join(actions[:3])}"
            else:
                description = f"Execute operational sequence: {' → '.join(actions[:4])}"

            sample = {
                "instruction": f"What does binary sequence {encoded[:16]}... mean?",
                "input": f"binary:{encoded}",
                "output": description,
                "actions": actions,
                "protocol": "v3",
                "format": "base64",
                "source": event["type"],
                "timestamp": event.get("timestamp", time.time())
            }

            samples.append(sample)

        return samples

    def append_samples(self, samples):
        """Append samples to training file"""
        if not samples:
            return 0

        try:
            with open(TRAINING_OUTPUT, "a") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except (FileNotFoundError, IOError, Exception) as e:
            self.logger.error("Error writing samples to training file: %s", e)
            return 0

        return len(samples)

    def run(self):
        """Main collection cycle"""
        self.logger.info("🧬 [dna_collector] Starting DNA collection cycle")
        self.logger.info("📊 [dna_collector] Last run: %s", datetime.fromtimestamp(self.state['last_run']).strftime('%Y-%m-%d %H:%M:%S') if self.state['last_run'] else 'Never')

        # Collect events from all sources
        events = []
        events.extend(self.watch_honeypot())
        events.extend(self.watch_memory_log())
        events.extend(self.watch_cell_outputs())

        self.new_events = len(events)
        self.state["total_events"] += self.new_events

        self.logger.info("📥 [dna_collector] Collected %d new events", self.new_events)
        self.logger.info("📈 [dna_collector] Total events: %d", self.state['total_events'])

        # Generate samples if threshold met
        if self.new_events >= EVENTS_THRESHOLD:
            self.logger.info("🔬 [dna_collector] Threshold met (%d >= %d)", self.new_events, EVENTS_THRESHOLD)
            self.logger.info("🧪 [dna_collector] Generating %d training samples...", SAMPLES_PER_RUN)

            samples = self.generate_samples_from_events(events)
            added = self.append_samples(samples)

            self.state["total_samples"] += added

            self.logger.info("✅ [dna_collector] Added %d samples to %s", added, TRAINING_OUTPUT)
            self.logger.info("📊 [dna_collector] Total samples: %d", self.state['total_samples'])
        else:
            self.logger.info("⏳ [dna_collector] Waiting for threshold (%d/%d events)", self.new_events, EVENTS_THRESHOLD)

        # Save state
        self.save_state()
        self.logger.info("💾 [dna_collector] State saved")

if __name__ == "__main__":
    try:
        collector = DNACollector()
        collector.run()
    except KeyboardInterrupt:
        collector.logger.info("⛔ [dna_collector] Interrupted")
    except Exception as e:
        collector.logger.error("❌ [dna_collector] Error: %s", e)
        import traceback
        traceback.print_exc()