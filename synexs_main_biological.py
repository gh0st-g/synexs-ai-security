#!/usr/bin/env python3
"""
Synexs Biological Organism - PRODUCTION INTEGRATION
Complete digital organism running your entire Synexs system

This replaces manual process management with a living, evolving organism that:
- Manages all Synexs processes as "cells"
- Responds to threats with adaptive immunity
- Evolves agents through sexual reproduction
- Maintains homeostasis through metabolism
- Ages, learns, adapts, and reproduces

Usage:
    python3 synexs_main_biological.py

The organism will:
1. Initialize all biological systems
2. Start existing Synexs processes (honeypot, swarm, listener)
3. Monitor health and resources
4. Trigger immune responses to threats
5. Evolve new agent generations
6. Maintain detailed organism state
"""

import os
import sys
import time
import json
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import queue

# Add synexs to path
sys.path.append('/root/synexs')

# Import biological systems
from synexs_biological_organism import SynexsBiologicalOrganism
from synexs_adaptive_immune_system import Antigen, ImmuneResponse
from synexs_cell_differentiation import DifferentiationSignal, CellType
from synexs_metabolism_engine import ResourceType, MetabolicProcess, MetabolicState

# Configuration
WORK_DIR = Path('/root/synexs')
ORGANISM_ID = "synexs_production_alpha"
STATE_FILE = WORK_DIR / "organism_state.json"
LOG_FILE = WORK_DIR / "biological_organism.log"
THREAT_QUEUE_FILE = WORK_DIR / "datasets/honeypot/attacks.json"

# Process management
MANAGED_PROCESSES = {
    'honeypot': {
        'command': ['/root/synexs/synexs_env/bin/python3', 'honeypot_server.py'],
        'cwd': '/root/synexs',
        'cell_type': CellType.DEFENDER,
        'critical': True
    },
    'swarm': {
        'command': ['/root/synexs/synexs_env/bin/python3', 'ai_swarm_fixed.py'],
        'cwd': '/root/synexs',
        'cell_type': CellType.EXECUTOR,
        'critical': True
    },
    'listener': {
        'command': ['python3', 'listener.py'],
        'cwd': '/root/synexs',
        'cell_type': CellType.SCOUT,
        'critical': True
    }
}

class BiologicalOrchestrator:
    """
    Master orchestrator for biological Synexs organism

    Manages the entire organism lifecycle:
    - Process management (cells)
    - Threat response (immune system)
    - Evolution (genetics)
    - Resource management (metabolism)
    """

    def __init__(self, organism_id: str):
        self.organism_id = organism_id
        self.running = False
        self.processes: Dict[str, subprocess.Popen] = {}
        self.threat_queue = queue.Queue()

        # Create the biological organism
        print("=" * 70)
        print("  SYNEXS BIOLOGICAL ORGANISM - PRODUCTION MODE")
        print("=" * 70)
        print(f"Organism ID: {organism_id}")
        print(f"Working Directory: {WORK_DIR}")
        print(f"State File: {STATE_FILE}")
        print("=" * 70)
        print()

        self.organism = SynexsBiologicalOrganism(organism_id)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n✅ Biological organism initialized!")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n\n🛑 Shutdown signal received - gracefully stopping organism...")
        self.stop()
        sys.exit(0)

    def start_process(self, name: str, config: Dict[str, Any]) -> bool:
        """Start a managed process (cell)"""
        try:
            print(f"\n🔷 Starting {name}...")

            # Check if already running
            if name in self.processes and self.processes[name].poll() is None:
                print(f"  ✓ Already running (PID {self.processes[name].pid})")
                return True

            # Start process
            proc = subprocess.Popen(
                config['command'],
                cwd=config['cwd'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            # Wait a moment to check it didn't immediately crash
            time.sleep(0.5)
            if proc.poll() is not None:
                print(f"  ❌ Process crashed immediately")
                return False

            self.processes[name] = proc
            print(f"  ✓ Started successfully (PID {proc.pid})")

            # Register as cell in organism
            cell_type = config.get('cell_type', CellType.EXECUTOR)
            self.organism.cell_system.emit_signal(DifferentiationSignal.COORDINATION_REQUIRED)

            return True

        except Exception as e:
            print(f"  ❌ Failed to start: {e}")
            return False

    def check_process_health(self, name: str) -> bool:
        """Check if a process is healthy"""
        if name not in self.processes:
            return False

        proc = self.processes[name]
        if proc.poll() is not None:
            return False

        return True

    def restart_process(self, name: str):
        """Restart a failed process"""
        print(f"\n🔄 Restarting {name}...")

        # Kill old process if exists
        if name in self.processes:
            try:
                self.processes[name].kill()
                self.processes[name].wait(timeout=5)
            except:
                pass
            del self.processes[name]

        # Start new process
        config = MANAGED_PROCESSES[name]
        success = self.start_process(name, config)

        if success:
            print(f"  ✅ {name} restarted successfully")
            # Trigger immune response (self-healing)
            self.organism.health = min(1.0, self.organism.health + 0.05)
        else:
            print(f"  ❌ {name} restart failed")
            # Health penalty
            self.organism.health -= 0.1
            self.organism.health = max(0.0, self.organism.health)

    def monitor_threats(self):
        """Monitor for new threats from honeypot"""
        try:
            if not THREAT_QUEUE_FILE.exists():
                return

            # Read threat data (JSONL format - one JSON object per line)
            threats = []
            with open(THREAT_QUEUE_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            threats.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines

            if not threats or len(threats) == 0:
                return

            # Process recent threats (last 10)
            recent_threats = threats[-10:]

            for threat in recent_threats:
                # Create threat signature
                threat_sig = f"{threat.get('type', 'unknown')}_{threat.get('ip', 'unknown')}"

                # Check if already processed (basic dedup)
                if hasattr(self, '_processed_threats') and threat_sig in self._processed_threats:
                    continue

                # Mount immune response
                print(f"\n🦠 New threat detected: {threat.get('type', 'unknown')} from {threat.get('ip', 'unknown')}")

                threat_data = {
                    'type': threat.get('type', 'unknown'),
                    'source_ip': threat.get('ip', 'unknown'),
                    'payload': threat.get('payload', ''),
                    'timestamp': threat.get('timestamp', time.time())
                }

                response_id = self.organism.encounter_threat(threat_data)

                # Mark as success (honeypot blocked it)
                self.organism.resolve_threat(response_id, success=True)

                # Track processed threats
                if not hasattr(self, '_processed_threats'):
                    self._processed_threats = set()
                self._processed_threats.add(threat_sig)

                # Limit memory
                if len(self._processed_threats) > 1000:
                    self._processed_threats = set(list(self._processed_threats)[-500:])

        except Exception as e:
            print(f"⚠️ Threat monitoring error: {e}")

    def lifecycle_cycle(self):
        """
        One organism lifecycle cycle
        Runs every minute
        """
        # 1. Age the organism
        self.organism.age_cycle()

        # 2. Monitor process health
        for name, config in MANAGED_PROCESSES.items():
            if not self.check_process_health(name):
                if config.get('critical'):
                    print(f"⚠️ Critical process {name} is down!")
                    self.restart_process(name)
                    # Trigger defense signal
                    self.organism.cell_system.emit_signal(DifferentiationSignal.DEFENSE_NEEDED)

        # 3. Monitor threats
        self.monitor_threats()

        # 4. Check organism health
        state = self.organism.get_organism_state()

        if state.health < 0.3:
            print(f"\n⚠️ Organism health CRITICAL: {state.health:.1%}")
            # Increase defenders
            self.organism.cell_system.emit_signal(DifferentiationSignal.DEFENSE_NEEDED)
            self.organism.cell_system.auto_differentiate_population()

        # 5. Resource management
        metabolic = self.organism.metabolism.get_resource_status()

        # Check for resource starvation
        for res_type, pool in self.organism.metabolism.resources.items():
            if pool.critical:
                print(f"⚠️ Resource {res_type.value} is CRITICAL: {pool.percentage:.1f}%")
                # Trigger stress response
                self.organism.metabolism.trigger_stress_response()

        # 6. Evolution - attempt reproduction periodically
        if state.age % 500 == 0 and state.fitness > 0.6:
            print("\n🔬 Organism mature enough for reproduction...")
            offspring = self.organism.reproduce()
            if offspring:
                print(f"  ✅ New generation created!")
                print(f"     Genome: {' → '.join(offspring.genome[:5])}...")
                # Save offspring genome
                self._save_offspring(offspring)

        # 7. Export state periodically
        if state.age % 60 == 0:  # Every hour
            self.organism.export_organism_state(str(STATE_FILE))
            self._log_status()

    def _save_offspring(self, offspring):
        """Save offspring genome for future use"""
        offspring_file = WORK_DIR / f"datasets/genomes/generation_{self.organism.generation}.json"
        offspring_file.parent.mkdir(parents=True, exist_ok=True)

        offspring_data = {
            'agent_id': offspring.agent_id,
            'generation': offspring.generation,
            'genome': offspring.genome,
            'parent_ids': offspring.parent_ids,
            'fitness': offspring.fitness,
            'timestamp': datetime.now().isoformat()
        }

        with open(offspring_file, 'w') as f:
            json.dump(offspring_data, f, indent=2)

    def _log_status(self):
        """Log detailed organism status"""
        with open(LOG_FILE, 'a') as f:
            state = self.organism.get_organism_state()
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'age': state.age,
                'generation': state.generation,
                'health': state.health,
                'fitness': state.fitness,
                'metabolic_state': state.metabolic_state,
                'immune_status': state.immune_status,
                'process_count': len([p for p in self.processes.values() if p.poll() is None])
            }
            f.write(json.dumps(log_entry) + '\n')

    def start(self):
        """Start the biological organism"""
        print("\n" + "=" * 70)
        print("  STARTING BIOLOGICAL ORGANISM")
        print("=" * 70)

        self.running = True

        # Start all managed processes
        for name, config in MANAGED_PROCESSES.items():
            self.start_process(name, config)

        print("\n" + "=" * 70)
        print("  ORGANISM FULLY OPERATIONAL")
        print("=" * 70)
        print(self.organism.get_detailed_status())

        # Main lifecycle loop
        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1

                if cycle_count % 60 == 0:  # Every hour
                    print("\n" + "=" * 70)
                    print(f"  LIFECYCLE CYCLE {cycle_count} (Age: {self.organism.age})")
                    print("=" * 70)
                    print(self.organism.get_detailed_status())

                # Run lifecycle cycle
                self.lifecycle_cycle()

                # Sleep 1 minute between cycles
                time.sleep(60)

            except Exception as e:
                print(f"\n❌ Lifecycle error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)

    def stop(self):
        """Stop the organism gracefully"""
        print("\n🛑 Stopping biological organism...")
        self.running = False

        # Export final state
        self.organism.export_organism_state(str(STATE_FILE))

        print("  ✓ Organism state saved")

        # Stop all managed processes
        for name, proc in self.processes.items():
            try:
                print(f"  Stopping {name}...")
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()

        print("✅ Organism stopped successfully")


def main():
    """Main entry point"""
    # Create organism
    orchestrator = BiologicalOrchestrator(ORGANISM_ID)

    # Start organism
    orchestrator.start()


if __name__ == "__main__":
    main()
