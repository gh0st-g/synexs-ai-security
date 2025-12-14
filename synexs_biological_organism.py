#!/usr/bin/env python3
"""
Synexs Biological Organism - Master Orchestrator
Complete digital organism with all biological systems integrated

This is the "whole organism" - integrating:
1. Genetic Recombination (Sexual reproduction, DNA)
2. Adaptive Immune System (Threat memory and defense)
3. Cell Differentiation (Dynamic specialization)
4. Resource Metabolism (Energy and homeostasis)
5. Neural Coordination (Multi-agent teamwork)

Biological Systems Integration:
┌─────────────────────────────────────────────────────────┐
│                 SYNEXS ORGANISM                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  DNA/Genetics        →  Genetic Recombination          │
│  Immune System       →  Adaptive Immunity              │
│  Cells/Tissues       →  Cell Differentiation           │
│  Metabolism          →  Resource Management            │
│  Nervous System      →  Agent Coordination             │
│  Reproduction        →  Agent Spawning                 │
│  Evolution           →  Learning & Adaptation          │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""

import time
import json
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import biological systems
sys.path.append('/root/synexs')
from synexs_genetic_recombination import GeneticRecombinator, GeneticProfile
from synexs_adaptive_immune_system import AdaptiveImmuneSystem, Antigen, ImmuneResponse
from synexs_cell_differentiation import CellDifferentiationEngine, CellType, DifferentiationSignal
from synexs_metabolism_engine import MetabolismEngine, ResourceType, MetabolicProcess, MetabolicState

@dataclass
class OrganismState:
    """Current state of the organism"""
    organism_id: str
    generation: int
    age: int
    health: float  # 0.0 - 1.0
    fitness: float  # Overall evolutionary fitness
    population_size: int
    metabolic_state: str
    immune_status: str
    threat_level: float

class SynexsBiologicalOrganism:
    """
    Complete digital organism integrating all biological systems

    The organism exhibits:
    - Lifespan and aging
    - Health and fitness
    - Reproduction and evolution
    - Immune responses to threats
    - Resource management
    - Cell specialization
    - Learning and adaptation
    """

    def __init__(self, organism_id: str = "synexs_001"):
        self.organism_id = organism_id
        self.generation = 0
        self.age = 0
        self.health = 1.0
        self.fitness = 0.5

        # Initialize biological systems
        print("🧬 Initializing biological systems...")

        self.genetics = GeneticRecombinator(mutation_rate=0.05)
        print("  ✓ Genetic system initialized")

        self.immune_system = AdaptiveImmuneSystem()
        print("  ✓ Immune system initialized")

        self.cell_system = CellDifferentiationEngine()
        print("  ✓ Cell differentiation system initialized")

        self.metabolism = MetabolismEngine()
        print("  ✓ Metabolic system initialized")

        # Create initial cell population (like embryonic development)
        self._embryonic_development()

        # Create genesis genome
        self._create_genesis_genome()

        print(f"✅ Organism {organism_id} successfully initialized!\n")

    def _embryonic_development(self):
        """Create initial cell population (embryonic development)"""
        print("\n🌱 Embryonic development...")

        # Start with stem cells
        for i in range(15):
            self.cell_system.create_stem_cell()

        # Emit differentiation signals for initial specialization
        self.cell_system.emit_signal(DifferentiationSignal.THREAT_DETECTED)
        self.cell_system.emit_signal(DifferentiationSignal.COORDINATION_REQUIRED)
        self.cell_system.emit_signal(DifferentiationSignal.LEARNING_NEEDED)

        # Initial differentiation
        diff_count = self.cell_system.auto_differentiate_population()
        print(f"  ✓ Differentiated {diff_count} cells during development")

    def _create_genesis_genome(self):
        """Create initial genetic population"""
        print("\n🧬 Creating genesis genome...")

        genesis_genomes = [
            ['SCAN', 'LEARN', 'DEFEND'],
            ['EVADE', 'REPORT', 'ADAPT'],
            ['ANALYZE', 'ATTACK', 'LEARN'],
            ['SCAN', 'EVADE', 'REPLICATE']
        ]

        for i, genome in enumerate(genesis_genomes):
            self.genetics.register_agent(
                agent_id=f"{self.organism_id}_agent_{i}",
                genome=genome,
                fitness=0.6
            )

        print(f"  ✓ Created {len(genesis_genomes)} genesis genomes")

    def encounter_threat(self, threat_data: Dict[str, Any]) -> str:
        """
        Organism encounters a threat - triggers immune response

        Returns: Response status
        """
        print(f"\n🦠 Threat encountered: {threat_data.get('type', 'unknown')}")

        # 1. Recognize threat (immune system)
        antigen = self.immune_system.recognize_threat(threat_data)

        # 2. Mount immune response
        response = self.immune_system.mount_immune_response(antigen)

        # 3. Allocate metabolic resources for immune response
        immune_process = MetabolicProcess(
            process_id=response.response_id,
            name=f"Immune response to {antigen.threat_type}",
            resource_cost={
                ResourceType.ENERGY: response.inflammation_level * 50,
                ResourceType.ATTENTION: response.inflammation_level * 30,
                ResourceType.MEMORY: 20.0
            },
            resource_yield={
                ResourceType.ENERGY: 10.0  # Small recovery after response
            },
            duration=5.0,
            priority=9  # High priority (immune response)
        )

        if self.metabolism.allocate_resources(immune_process):
            print("  ✓ Metabolic resources allocated for immune response")
        else:
            print("  ⚠️ Insufficient resources for full immune response")

        # 4. Trigger cell specialization if needed
        if antigen.danger_level > 0.7:
            print("  ⚠️ High danger - increasing defender cells")
            self.cell_system.emit_signal(DifferentiationSignal.DEFENSE_NEEDED)
            self.cell_system.auto_differentiate_population()

        # 5. Update organism health
        self.health -= antigen.danger_level * 0.1
        self.health = max(0.0, self.health)

        return response.response_id

    def resolve_threat(self, response_id: str, success: bool):
        """Resolve immune response"""
        print(f"\n{'✅' if success else '❌'} Threat resolution: {'Success' if success else 'Failure'}")

        # Report to immune system
        self.immune_system.report_outcome(response_id, success, {})

        # Complete metabolic process
        self.metabolism.complete_process(response_id)

        # Update health
        if success:
            self.health = min(1.0, self.health + 0.1)
            self.fitness += 0.05
        else:
            self.health -= 0.15
            self.fitness -= 0.02

        self.fitness = max(0.0, min(1.0, self.fitness))
        self.health = max(0.0, self.health)

    def execute_mission(self, mission_data: Dict[str, Any]) -> bool:
        """
        Execute a mission (like agent mission in Phase 1)

        Returns: Success/failure
        """
        print(f"\n🎯 Mission: {mission_data.get('type', 'unknown')}")

        # 1. Allocate metabolic resources
        mission_cost = mission_data.get('complexity', 0.5)

        mission_process = MetabolicProcess(
            process_id=f"mission_{int(time.time())}",
            name=mission_data.get('type', 'mission'),
            resource_cost={
                ResourceType.ENERGY: mission_cost * 30,
                ResourceType.BANDWIDTH: mission_cost * 20,
                ResourceType.ATTENTION: mission_cost * 25,
                ResourceType.TIME: mission_cost * 15
            },
            resource_yield={
                ResourceType.ENERGY: mission_cost * 15  # Gain from success
            },
            duration=mission_data.get('duration', 10.0),
            priority=mission_data.get('priority', 5)
        )

        if not self.metabolism.allocate_resources(mission_process):
            print("  ❌ Insufficient resources - mission aborted")
            return False

        # 2. Simulate mission execution
        success_prob = mission_data.get('success_probability', 0.5)

        # Factors affecting success
        health_factor = self.health
        fitness_factor = self.fitness
        metabolic_factor = 1.0 if self.metabolism.state == MetabolicState.ACTIVE else 0.7

        adjusted_prob = success_prob * health_factor * fitness_factor * metabolic_factor

        import random
        success = random.random() < adjusted_prob

        # 3. Complete mission
        self.metabolism.complete_process(mission_process.process_id)

        # 4. Update fitness
        if success:
            self.fitness = min(1.0, self.fitness + 0.03)
            print("  ✅ Mission successful")
        else:
            self.fitness = max(0.0, self.fitness - 0.01)
            print("  ❌ Mission failed")

        return success

    def reproduce(self) -> Optional[GeneticProfile]:
        """
        Reproduce - create offspring through sexual reproduction

        Only possible if:
        - Health > 0.5
        - Fitness > 0.6
        - Sufficient metabolic resources
        """
        print("\n🔬 Attempting reproduction...")

        # Check health and fitness
        if self.health < 0.5:
            print("  ❌ Health too low for reproduction")
            return None

        if self.fitness < 0.6:
            print("  ❌ Fitness too low for reproduction")
            return None

        # Check metabolic resources
        reproduction_process = MetabolicProcess(
            process_id=f"reproduction_{int(time.time())}",
            name="Sexual Reproduction",
            resource_cost={
                ResourceType.ENERGY: 60.0,
                ResourceType.MEMORY: 40.0,
                ResourceType.ATTENTION: 30.0
            },
            resource_yield={},
            duration=15.0,
            priority=7
        )

        if not self.metabolism.allocate_resources(reproduction_process):
            print("  ❌ Insufficient metabolic resources")
            return None

        # Select parents and create offspring
        offspring = self.genetics.evolve_generation(num_offspring=1)[0]

        # Complete reproduction process
        self.metabolism.complete_process(reproduction_process.process_id)

        # Cost of reproduction
        self.health -= 0.1

        self.generation += 1

        print(f"  ✅ Offspring created: {offspring.agent_id}")
        print(f"     Genome: {' → '.join(offspring.genome[:5])}{'...' if len(offspring.genome) > 5 else ''}")

        return offspring

    def age_cycle(self):
        """
        One aging cycle (like a day/week in biological time)

        Performs:
        - Metabolic cycle
        - Cell maintenance
        - Immune system decay
        - Aging effects
        """
        self.age += 1

        # 1. Metabolic cycle
        self.metabolism.metabolic_cycle(delta_time=1.0)

        # 2. Age-related health decline (very slow)
        if self.age % 100 == 0:
            self.health *= 0.99  # 1% decline per 100 cycles

        # 3. Immune system maintenance (memory decay)
        self.immune_system._decay_memories()

        # 4. Cell population maintenance
        needs = self.cell_system.assess_population_needs()

        # Auto-differentiate if significantly out of balance
        max_need = max(needs.values()) if needs else 0
        if max_need > 0.15:  # 15% deficit
            self.cell_system.auto_differentiate_population()

    def get_organism_state(self) -> OrganismState:
        """Get current organism state"""
        immune_status = self.immune_system.get_immune_status()

        return OrganismState(
            organism_id=self.organism_id,
            generation=self.generation,
            age=self.age,
            health=self.health,
            fitness=self.fitness,
            population_size=len(self.cell_system.cells),
            metabolic_state=self.metabolism.state.value,
            immune_status=f"{immune_status['memory_cells']} memories, {immune_status['antibodies_available']} antibodies",
            threat_level=self.immune_system.total_encounters / max(1, self.age) if self.age > 0 else 0
        )

    def get_detailed_status(self) -> str:
        """Get comprehensive organism status"""
        state = self.get_organism_state()

        status = f"""
╔════════════════════════════════════════════════════════════╗
║           SYNEXS BIOLOGICAL ORGANISM STATUS                ║
╚════════════════════════════════════════════════════════════╝

ORGANISM: {state.organism_id}
Generation: {state.generation}
Age: {state.age} cycles

VITAL SIGNS:
  Health:        {'█' * int(state.health * 20)} {state.health:.1%}
  Fitness:       {'█' * int(state.fitness * 20)} {state.fitness:.1%}

SYSTEMS STATUS:
"""

        # Metabolic status
        metabolic = self.metabolism.get_resource_status()
        status += f"  Metabolism:    {metabolic['state'].upper()} (efficiency: {metabolic['efficiency']:.2f})\n"

        # Immune status
        immune = self.immune_system.get_immune_status()
        status += f"  Immune System: {immune['memory_cells']} memory cells, {immune['success_rate']:.1%} success rate\n"

        # Cell population
        cell_count = len(self.cell_system.cells)
        status += f"  Cell Count:    {cell_count} cells\n"

        # Genetic diversity
        status += f"  Gene Pool:     {len(self.genetics.gene_pool)} genomes\n"

        status += "\n" + "=" * 60

        return status

    def export_organism_state(self, filepath: str):
        """Export complete organism state"""
        export_data = {
            'organism_id': self.organism_id,
            'generation': self.generation,
            'age': self.age,
            'health': self.health,
            'fitness': self.fitness,
            'genetics': {
                'generation': self.genetics.generation,
                'population_size': len(self.genetics.gene_pool)
            },
            'immune_system': self.immune_system.get_immune_status(),
            'metabolism': self.metabolism.get_resource_status(),
            'cells': {
                'total': len(self.cell_system.cells),
                'by_type': {}
            }
        }

        # Count cells by type
        for cell in self.cell_system.cells.values():
            cell_type = cell.cell_type.value
            export_data['cells']['by_type'][cell_type] = \
                export_data['cells']['by_type'].get(cell_type, 0) + 1

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Organism state exported to {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SYNEXS BIOLOGICAL ORGANISM - Complete Digital Life Form")
    print("=" * 60)

    # Create organism
    organism = SynexsBiologicalOrganism("synexs_alpha")

    # Show initial status
    print(organism.get_detailed_status())

    # Simulate life
    print("\n" + "=" * 60)
    print("  SIMULATING ORGANISM LIFESPAN")
    print("=" * 60)

    # Age 10 cycles
    print("\n⏰ Aging 10 cycles...")
    for i in range(10):
        organism.age_cycle()

        if i % 3 == 0:
            print(f"  Cycle {i+1}: Health={organism.health:.2f}, Fitness={organism.fitness:.2f}")

    # Encounter threats
    print("\n🦠 Encountering threats...")

    threat1 = {
        'type': 'honeypot',
        'ptr_record': 'honeypot.test',
        'indicators': ['fake_vulns', 'bait_files'],
        'detection_likelihood': 0.75
    }

    response1 = organism.encounter_threat(threat1)
    time.sleep(0.5)
    organism.resolve_threat(response1, success=True)

    # Execute missions
    print("\n🎯 Executing missions...")

    mission1 = {
        'type': 'reconnaissance',
        'complexity': 0.4,
        'success_probability': 0.7,
        'duration': 5.0,
        'priority': 6
    }

    organism.execute_mission(mission1)

    mission2 = {
        'type': 'attack',
        'complexity': 0.8,
        'success_probability': 0.5,
        'duration': 10.0,
        'priority': 8
    }

    organism.execute_mission(mission2)

    # Attempt reproduction
    print("\n🔬 Attempting reproduction...")
    offspring = organism.reproduce()

    if offspring:
        print("  New generation created!")

    # Final status
    print("\n" + "=" * 60)
    print("  FINAL ORGANISM STATUS")
    print("=" * 60)
    print(organism.get_detailed_status())

    # Detailed system reports
    print("\n" + organism.genetics.get_population_report())
    print("\n" + organism.cell_system.get_population_report())

    # Export
    organism.export_organism_state('/root/synexs/organism_state.json')

    print("\n✅ Biological organism simulation complete!")
