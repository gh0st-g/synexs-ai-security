#!/usr/bin/env python3
"""
Synexs Resource Metabolism Engine
Energy and resource management for digital organisms

Biological analogy:
- ATP/Energy: Computational resources (CPU, memory, network)
- Metabolism: Resource consumption and generation
- Homeostasis: Self-balancing resource allocation
- Catabolism: Breaking down complex operations
- Anabolism: Building complex structures from resources
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class ResourceType(Enum):
    """Types of resources in the digital organism"""
    ENERGY = "energy"           # CPU/processing power
    MEMORY = "memory"           # RAM/storage
    BANDWIDTH = "bandwidth"     # Network capacity
    ATTENTION = "attention"     # Focus/priority
    TIME = "time"               # Execution time budget

class MetabolicState(Enum):
    """Current metabolic state of organism"""
    RESTING = "resting"         # Minimal resource use
    ACTIVE = "active"           # Normal operations
    STRESSED = "stressed"       # High resource demand
    EXHAUSTED = "exhausted"     # Resource depleted
    RECOVERING = "recovering"   # Replenishing resources

@dataclass
class ResourcePool:
    """Pool of available resources"""
    resource_type: ResourceType
    current: float              # Current available
    maximum: float              # Maximum capacity
    regeneration_rate: float    # How fast it replenishes
    consumption_rate: float     # Current usage rate
    critical_threshold: float   # Alert level (0.0-1.0)

    @property
    def percentage(self) -> float:
        """Get current resource level as percentage"""
        return (self.current / self.maximum * 100) if self.maximum > 0 else 0.0

    @property
    def critical(self) -> bool:
        """Check if resource level is below critical threshold"""
        current_ratio = self.current / self.maximum if self.maximum > 0 else 0.0
        return current_ratio < self.critical_threshold

@dataclass
class MetabolicProcess:
    """A metabolic process consuming/producing resources"""
    process_id: str
    name: str
    resource_cost: Dict[ResourceType, float]
    resource_yield: Dict[ResourceType, float]
    duration: float
    priority: int
    started_at: Optional[float] = None
    completed: bool = False

class MetabolismEngine:
    """
    Manages resource metabolism for Synexs organism

    Key responsibilities:
    - Track resource pools (energy, memory, bandwidth)
    - Allocate resources to processes
    - Maintain homeostasis (balance)
    - Trigger stress responses when depleted
    - Optimize resource efficiency
    """

    def __init__(self, initial_resources: Dict[ResourceType, float] = None):
        # Initialize resource pools
        self.resources = self._initialize_resources(initial_resources)

        # Metabolic processes (running operations)
        self.active_processes: Dict[str, MetabolicProcess] = {}
        self.process_queue: List[MetabolicProcess] = []

        # Metabolic state
        self.state = MetabolicState.RESTING
        self.stress_level = 0.0  # 0.0 - 1.0

        # Homeostasis targets (ideal resource levels)
        self.homeostasis_targets = {
            ResourceType.ENERGY: 0.7,
            ResourceType.MEMORY: 0.6,
            ResourceType.BANDWIDTH: 0.5,
            ResourceType.ATTENTION: 0.7,
            ResourceType.TIME: 0.8
        }

        # Statistics
        self.total_energy_consumed = 0.0
        self.total_operations = 0
        self.efficiency_score = 1.0

    def _initialize_resources(self, initial: Optional[Dict[ResourceType, float]]) -> Dict[ResourceType, ResourcePool]:
        """Initialize resource pools"""
        defaults = {
            ResourceType.ENERGY: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.BANDWIDTH: 100.0,
            ResourceType.ATTENTION: 100.0,
            ResourceType.TIME: 100.0
        }

        pools = {}
        for res_type in ResourceType:
            initial_val = initial.get(res_type, defaults[res_type]) if initial else defaults[res_type]

            pools[res_type] = ResourcePool(
                resource_type=res_type,
                current=initial_val,
                maximum=initial_val,
                regeneration_rate=initial_val * 0.05,  # 5% per cycle
                consumption_rate=0.0,
                critical_threshold=0.2  # 20%
            )

        return pools

    def allocate_resources(self, process: MetabolicProcess) -> bool:
        """
        Attempt to allocate resources for a process
        Returns True if successful, False if insufficient resources
        """
        # Check if we have enough resources
        for res_type, cost in process.resource_cost.items():
            pool = self.resources[res_type]
            if pool.current < cost:
                # Insufficient resources
                return False

        # Allocate resources
        for res_type, cost in process.resource_cost.items():
            self.resources[res_type].current -= cost
            self.resources[res_type].consumption_rate += cost / process.duration

        # Start process
        process.started_at = time.time()
        self.active_processes[process.process_id] = process
        self.total_operations += 1

        return True

    def complete_process(self, process_id: str):
        """Complete a process and yield resources"""
        if process_id not in self.active_processes:
            return

        process = self.active_processes[process_id]

        # Yield resources
        for res_type, yield_amt in process.resource_yield.items():
            pool = self.resources[res_type]
            pool.current = min(pool.maximum, pool.current + yield_amt)

        # Update consumption rate
        for res_type, cost in process.resource_cost.items():
            self.resources[res_type].consumption_rate -= cost / process.duration

        # Mark complete
        process.completed = True
        del self.active_processes[process_id]

    def regenerate_resources(self, delta_time: float = 1.0):
        """
        Regenerate resources over time (anabolism)
        Like how organisms regenerate ATP
        """
        for pool in self.resources.values():
            # Calculate regeneration
            regen = pool.regeneration_rate * delta_time

            # Apply stress penalty
            if self.state == MetabolicState.STRESSED:
                regen *= 0.5  # 50% slower under stress
            elif self.state == MetabolicState.EXHAUSTED:
                regen *= 0.2  # 80% slower when exhausted

            # Regenerate
            pool.current = min(pool.maximum, pool.current + regen)

    def assess_metabolic_state(self) -> MetabolicState:
        """
        Assess current metabolic state based on resource levels
        """
        # Calculate average resource level
        avg_level = np.mean([
            pool.current / pool.maximum
            for pool in self.resources.values()
        ])

        # Count critical resources
        critical_count = sum(
            1 for pool in self.resources.values()
            if pool.current / pool.maximum < pool.critical_threshold
        )

        # Determine state
        if critical_count >= 3:
            return MetabolicState.EXHAUSTED
        elif critical_count >= 1:
            return MetabolicState.STRESSED
        elif avg_level < 0.4:
            return MetabolicState.RECOVERING
        elif avg_level > 0.9 and len(self.active_processes) == 0:
            return MetabolicState.RESTING
        else:
            return MetabolicState.ACTIVE

    def maintain_homeostasis(self):
        """
        Maintain homeostasis - keep resources in balance
        Biological organisms constantly regulate internal state
        """
        for res_type, target in self.homeostasis_targets.items():
            pool = self.resources[res_type]
            current_ratio = pool.current / pool.maximum

            # Too low - increase regeneration
            if current_ratio < target:
                deficit = target - current_ratio
                pool.regeneration_rate *= (1 + deficit * 0.1)

            # Too high - decrease regeneration (save energy)
            elif current_ratio > target + 0.1:
                surplus = current_ratio - target
                pool.regeneration_rate *= (1 - surplus * 0.05)

    def trigger_stress_response(self):
        """
        Activate stress response when resources are low
        Like cortisol/adrenaline in biology
        """
        self.stress_level = min(1.0, self.stress_level + 0.1)

        # Emergency measures
        if self.state == MetabolicState.STRESSED:
            # Pause low-priority processes
            self._pause_low_priority_processes()

            # Reallocate resources to critical operations
            self._reallocate_to_critical()

        elif self.state == MetabolicState.EXHAUSTED:
            # Emergency shutdown of non-essential processes
            self._emergency_shutdown()

    def _pause_low_priority_processes(self):
        """Pause processes with priority < 5"""
        for proc in list(self.active_processes.values()):
            if proc.priority < 5:
                # Return resources
                for res_type, cost in proc.resource_cost.items():
                    self.resources[res_type].current += cost

                # Move to queue
                self.process_queue.append(proc)
                del self.active_processes[proc.process_id]

    def _reallocate_to_critical(self):
        """Reallocate resources to critical processes"""
        # Increase regeneration for critical resources
        for pool in self.resources.values():
            if pool.current / pool.maximum < pool.critical_threshold:
                pool.regeneration_rate *= 1.5

    def _emergency_shutdown(self):
        """Emergency shutdown of all non-critical processes"""
        critical_threshold = 8  # Priority >= 8 is critical

        for proc in list(self.active_processes.values()):
            if proc.priority < critical_threshold:
                # Return resources
                for res_type, cost in proc.resource_cost.items():
                    self.resources[res_type].current += cost

                # Cancel process
                del self.active_processes[proc.process_id]

    def calculate_efficiency(self) -> float:
        """
        Calculate metabolic efficiency
        Higher is better (more output per resource)
        """
        if self.total_energy_consumed == 0:
            return 1.0

        # Efficiency = useful work / energy consumed
        useful_work = self.total_operations
        efficiency = useful_work / max(1, self.total_energy_consumed / 10)

        # Penalty for stress
        if self.state == MetabolicState.STRESSED:
            efficiency *= 0.8
        elif self.state == MetabolicState.EXHAUSTED:
            efficiency *= 0.5

        return min(1.0, efficiency)

    def optimize_metabolism(self):
        """
        Optimize metabolic processes for efficiency
        Like how organisms adapt to conserve energy
        """
        # Identify inefficient processes
        if self.active_processes:
            avg_priority = np.mean([p.priority for p in self.active_processes.values()])

            # If running too many low-priority tasks, optimize
            if avg_priority < 5 and self.state != MetabolicState.RESTING:
                self._pause_low_priority_processes()

        # Adjust regeneration rates based on usage patterns
        for pool in self.resources.values():
            usage_ratio = pool.consumption_rate / (pool.regeneration_rate + 0.001)

            # If consuming faster than regenerating, boost regen
            if usage_ratio > 1.5:
                pool.regeneration_rate *= 1.2

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        return {
            'state': self.state.value,
            'stress_level': self.stress_level,
            'efficiency': self.calculate_efficiency(),
            'resources': {
                res_type.value: {
                    'current': pool.current,
                    'maximum': pool.maximum,
                    'percentage': (pool.current / pool.maximum) * 100,
                    'regeneration': pool.regeneration_rate,
                    'consumption': pool.consumption_rate,
                    'critical': pool.current / pool.maximum < pool.critical_threshold
                }
                for res_type, pool in self.resources.items()
            },
            'active_processes': len(self.active_processes),
            'queued_processes': len(self.process_queue)
        }

    def metabolic_cycle(self, delta_time: float = 1.0):
        """
        Run one metabolic cycle
        This should be called regularly (like a heartbeat)
        """
        # 1. Regenerate resources
        self.regenerate_resources(delta_time)

        # 2. Assess metabolic state
        old_state = self.state
        self.state = self.assess_metabolic_state()

        # 3. React to state changes
        if self.state != old_state:
            print(f"🔄 Metabolic state changed: {old_state.value} → {self.state.value}")

            if self.state in [MetabolicState.STRESSED, MetabolicState.EXHAUSTED]:
                self.trigger_stress_response()

        # 4. Maintain homeostasis
        self.maintain_homeostasis()

        # 5. Optimize metabolism
        self.optimize_metabolism()

        # 6. Update efficiency
        self.efficiency_score = self.calculate_efficiency()

        # 7. Decay stress
        if self.state in [MetabolicState.RESTING, MetabolicState.RECOVERING]:
            self.stress_level = max(0.0, self.stress_level - 0.05)

    def export_metabolic_state(self, filepath: str):
        """Export current metabolic state"""
        export_data = {
            'timestamp': time.time(),
            'state': self.state.value,
            'stress_level': self.stress_level,
            'efficiency_score': self.efficiency_score,
            'resources': {
                res_type.value: asdict(pool)
                for res_type, pool in self.resources.items()
            },
            'statistics': {
                'total_energy_consumed': self.total_energy_consumed,
                'total_operations': self.total_operations,
                'active_processes': len(self.active_processes)
            }
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Metabolic state exported to {filepath}")
        except (IOError, OSError) as e:
            print(f"❌ Error exporting metabolic state: {e}")