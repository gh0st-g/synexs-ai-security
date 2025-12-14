#!/usr/bin/env python3
"""
Synexs Genetic Recombination System
Sexual reproduction for digital organisms - combines DNA from two successful agents

This implements biological sexual reproduction:
- Two parent agents contribute genetic material (action sequences)
- Crossover: DNA segments exchange between parents
- Genetic diversity: Creates offspring with novel combinations
- Fitness-based selection: Only successful agents reproduce
"""

import random
import json
import uuid
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from binary_protocol import encode_base64, decode_base64

@dataclass
class GeneticProfile:
    """DNA profile of an agent"""
    agent_id: str
    generation: int
    parent_ids: List[str]  # Both parents for sexual reproduction
    genome: List[str]  # Action sequences (DNA)
    fitness_score: float  # Survival/success rate
    mutations: int  # Total mutations accumulated
    age: int  # Generations survived
    traits: Dict[str, float]  # Phenotypic traits (expressed behaviors)
    epigenetic_markers: Dict[str, bool]  # Learned behaviors that persist

class GeneticRecombinator:
    """
    Implements sexual reproduction for digital organisms
    Combines genetic material from two successful parents
    """

    def __init__(self, mutation_rate=0.05):
        self.mutation_rate = mutation_rate
        self.gene_pool = {}  # agent_id -> GeneticProfile
        self.generation = 0
        self.population_history = []
        self.fitness_threshold = 0.6  # Minimum fitness to reproduce

    def register_agent(self, agent_id: str, genome: List[str],
                      parent_ids: List[str] = None, fitness: float = 0.5):
        """Register a new agent in the gene pool"""
        profile = GeneticProfile(
            agent_id=agent_id,
            generation=self.generation,
            parent_ids=parent_ids or [],
            genome=genome,
            fitness_score=fitness,
            mutations=0,
            age=0,
            traits=self._calculate_traits(genome),
            epigenetic_markers={}
        )
        self.gene_pool[agent_id] = profile
        return profile

    def _calculate_traits(self, genome: List[str]) -> Dict[str, float]:
        """
        Calculate phenotypic traits from genome
        Traits are emergent properties of action sequences
        """
        traits = {
            'aggression': 0.0,      # Offensive action frequency
            'stealth': 0.0,         # Evasion action frequency
            'intelligence': 0.0,    # Learning/analysis frequency
            'cooperation': 0.0,     # Communication frequency
            'resilience': 0.0       # Defense action frequency
        }

        if not genome:
            return traits

        # Map actions to traits
        action_trait_map = {
            'ATTACK': 'aggression',
            'SCAN': 'intelligence',
            'EVADE': 'stealth',
            'DEFEND': 'resilience',
            'REPORT': 'cooperation',
            'LEARN': 'intelligence',
            'ENCRYPT': 'stealth',
            'SYNC': 'cooperation'
        }

        # Calculate trait values
        for action in genome:
            if action in action_trait_map:
                trait = action_trait_map[action]
                traits[trait] += 1.0 / len(genome)

        return traits

    def sexual_reproduction(self, parent1_id: str, parent2_id: str) -> GeneticProfile:
        """
        Sexual reproduction: Combine DNA from two parents

        Process:
        1. Select parents based on fitness
        2. Crossover: Exchange DNA segments
        3. Mutation: Random changes
        4. Create offspring with combined traits
        """
        parent1 = self.gene_pool[parent1_id]
        parent2 = self.gene_pool[parent2_id]

        # Genetic crossover
        offspring_genome = self._crossover(parent1.genome, parent2.genome)

        # Apply mutations
        offspring_genome = self._mutate(offspring_genome)

        # Create offspring profile
        offspring_id = str(uuid.uuid4())[:8]
        offspring = GeneticProfile(
            agent_id=offspring_id,
            generation=self.generation + 1,
            parent_ids=[parent1_id, parent2_id],
            genome=offspring_genome,
            fitness_score=0.5,  # Initial fitness (to be determined)
            mutations=parent1.mutations + parent2.mutations,
            age=0,
            traits=self._calculate_traits(offspring_genome),
            epigenetic_markers=self._inherit_epigenetics(parent1, parent2)
        )

        self.gene_pool[offspring_id] = offspring
        return offspring

    def _crossover(self, genome1: List[str], genome2: List[str],
                   method: str = 'multi_point') -> List[str]:
        """
        Genetic crossover - exchange DNA segments between parents

        Methods:
        - single_point: One crossover point
        - multi_point: Multiple crossover points
        - uniform: Each gene chosen randomly from either parent
        """
        if not genome1 or not genome2:
            return genome1 or genome2

        max_len = max(len(genome1), len(genome2))
        min_len = min(len(genome1), len(genome2))

        if method == 'single_point':
            # Single crossover point
            point = random.randint(1, min_len - 1)
            offspring = genome1[:point] + genome2[point:]

        elif method == 'multi_point':
            # Two crossover points
            point1 = random.randint(1, min_len // 2)
            point2 = random.randint(min_len // 2, min_len - 1)
            offspring = genome1[:point1] + genome2[point1:point2] + genome1[point2:]

        elif method == 'uniform':
            # Each gene randomly from either parent
            offspring = []
            for i in range(min_len):
                offspring.append(random.choice([genome1[i], genome2[i]]))
            # Add remaining genes from longer parent
            if len(genome1) > len(genome2):
                offspring.extend(genome1[min_len:])
            else:
                offspring.extend(genome2[min_len:])

        else:
            raise ValueError(f"Unknown crossover method: {method}")

        return offspring

    def _mutate(self, genome: List[str]) -> List[str]:
        """
        Apply random mutations to genome

        Mutation types:
        - Point mutation: Change single action
        - Insertion: Add new action
        - Deletion: Remove action
        - Duplication: Copy segment
        """
        mutated = genome.copy()

        # All possible actions (from vocab_v3_binary.json)
        all_actions = [
            'SCAN', 'ATTACK', 'REPLICATE', 'MUTATE', 'EVADE', 'LEARN',
            'REPORT', 'DEFEND', 'REFINE', 'FLAG', 'XOR_PAYLOAD', 'ENCRYPT',
            'COMPRESS', 'HASH_CHECK', 'SYNC', 'SPLIT', 'MERGE', 'STACK_PUSH',
            'STACK_POP', 'TERMINATE', 'PAUSE', 'LOG', 'QUERY', 'ACK', 'NACK',
            'CHECKPOINT', 'VALIDATE', 'BROADCAST', 'LISTEN', 'ROUTE', 'FILTER',
            'TRANSFORM'
        ]

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['point', 'insertion', 'deletion', 'duplication'])

                if mutation_type == 'point':
                    # Point mutation: Replace action
                    mutated[i] = random.choice(all_actions)

                elif mutation_type == 'insertion' and len(mutated) < 20:
                    # Insert new action
                    mutated.insert(i, random.choice(all_actions))

                elif mutation_type == 'deletion' and len(mutated) > 3:
                    # Delete action
                    mutated.pop(i)

                elif mutation_type == 'duplication' and len(mutated) < 15:
                    # Duplicate segment
                    if i + 2 < len(mutated):
                        segment = mutated[i:i+2]
                        mutated[i:i] = segment

        return mutated

    def _inherit_epigenetics(self, parent1: GeneticProfile,
                            parent2: GeneticProfile) -> Dict[str, bool]:
        """
        Inherit epigenetic markers (learned behaviors) from parents
        These are non-genetic traits that can be inherited
        """
        inherited = {}

        # Combine markers from both parents
        all_markers = set(parent1.epigenetic_markers.keys()) | set(parent2.epigenetic_markers.keys())

        for marker in all_markers:
            # Inherit if either parent has it (OR logic)
            # Could also use AND logic for stricter inheritance
            parent1_has = parent1.epigenetic_markers.get(marker, False)
            parent2_has = parent2.epigenetic_markers.get(marker, False)
            inherited[marker] = parent1_has or parent2_has

        return inherited

    def select_parents(self, population_size: int = 2) -> List[str]:
        """
        Select parents for reproduction based on fitness
        Uses tournament selection
        """
        # Filter agents above fitness threshold
        viable_agents = [
            agent_id for agent_id, profile in self.gene_pool.items()
            if profile.fitness_score >= self.fitness_threshold
        ]

        if len(viable_agents) < 2:
            # Not enough fit agents, lower threshold temporarily
            viable_agents = sorted(
                self.gene_pool.keys(),
                key=lambda x: self.gene_pool[x].fitness_score,
                reverse=True
            )[:max(2, len(self.gene_pool) // 2)]

        # Tournament selection
        selected = []
        for _ in range(population_size):
            tournament = random.sample(viable_agents, min(3, len(viable_agents)))
            winner = max(tournament, key=lambda x: self.gene_pool[x].fitness_score)
            selected.append(winner)

        return selected

    def evolve_generation(self, num_offspring: int = 10) -> List[GeneticProfile]:
        """
        Create a new generation through sexual reproduction
        """
        new_generation = []

        for _ in range(num_offspring):
            # Select two parents
            parent1_id, parent2_id = self.select_parents(2)

            # Create offspring through sexual reproduction
            offspring = self.sexual_reproduction(parent1_id, parent2_id)
            new_generation.append(offspring)

        self.generation += 1

        # Record population statistics
        self._record_generation_stats()

        return new_generation

    def _record_generation_stats(self):
        """Record statistics about current generation"""
        if not self.gene_pool:
            return

        stats = {
            'generation': self.generation,
            'population_size': len(self.gene_pool),
            'avg_fitness': np.mean([p.fitness_score for p in self.gene_pool.values()]),
            'max_fitness': max([p.fitness_score for p in self.gene_pool.values()]),
            'avg_genome_length': np.mean([len(p.genome) for p in self.gene_pool.values()]),
            'total_mutations': sum([p.mutations for p in self.gene_pool.values()]),
            'trait_averages': self._calculate_population_traits()
        }

        self.population_history.append(stats)

    def _calculate_population_traits(self) -> Dict[str, float]:
        """Calculate average traits across population"""
        all_traits = defaultdict(list)

        for profile in self.gene_pool.values():
            for trait, value in profile.traits.items():
                all_traits[trait].append(value)

        return {trait: np.mean(values) for trait, values in all_traits.items()}

    def update_fitness(self, agent_id: str, fitness_delta: float):
        """Update agent fitness based on mission results"""
        if agent_id in self.gene_pool:
            profile = self.gene_pool[agent_id]
            profile.fitness_score = max(0.0, min(1.0, profile.fitness_score + fitness_delta))
            profile.age += 1

    def add_epigenetic_marker(self, agent_id: str, marker: str, value: bool = True):
        """Add learned behavior that can be inherited"""
        if agent_id in self.gene_pool:
            self.gene_pool[agent_id].epigenetic_markers[marker] = value

    def get_family_tree(self, agent_id: str, depth: int = 3) -> Dict[str, Any]:
        """Get ancestral lineage of an agent"""
        if agent_id not in self.gene_pool:
            return None

        profile = self.gene_pool[agent_id]
        tree = {
            'agent_id': agent_id,
            'generation': profile.generation,
            'fitness': profile.fitness_score,
            'traits': profile.traits,
            'parents': []
        }

        if depth > 0 and profile.parent_ids:
            for parent_id in profile.parent_ids:
                if parent_id in self.gene_pool:
                    tree['parents'].append(self.get_family_tree(parent_id, depth - 1))

        return tree

    def export_gene_pool(self, filepath: str):
        """Export gene pool to JSON for analysis"""
        export_data = {
            'generation': self.generation,
            'gene_pool': {
                agent_id: asdict(profile)
                for agent_id, profile in self.gene_pool.items()
            },
            'population_history': self.population_history
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Gene pool exported to {filepath}")

    def get_population_report(self) -> str:
        """Generate human-readable population report"""
        if not self.gene_pool:
            return "No agents in gene pool"

        avg_fitness = np.mean([p.fitness_score for p in self.gene_pool.values()])
        max_fitness = max([p.fitness_score for p in self.gene_pool.values()])
        avg_age = np.mean([p.age for p in self.gene_pool.values()])

        # Find most fit agent
        best_agent_id = max(self.gene_pool.keys(),
                           key=lambda x: self.gene_pool[x].fitness_score)
        best_agent = self.gene_pool[best_agent_id]

        report = f"""
╔════════════════════════════════════════════════════════════╗
║           SYNEXS GENETIC POPULATION REPORT                 ║
╚════════════════════════════════════════════════════════════╝

Generation: {self.generation}
Population Size: {len(self.gene_pool)}

FITNESS METRICS:
  Average Fitness: {avg_fitness:.3f}
  Maximum Fitness: {max_fitness:.3f}
  Average Age: {avg_age:.1f} generations

TRAIT DISTRIBUTION (Population Averages):
"""

        trait_avg = self._calculate_population_traits()
        for trait, value in trait_avg.items():
            bar = '█' * int(value * 20)
            report += f"  {trait:15s}: {bar:20s} {value:.3f}\n"

        report += f"""
MOST FIT AGENT:
  ID: {best_agent_id}
  Fitness: {best_agent.fitness_score:.3f}
  Generation: {best_agent.generation}
  Age: {best_agent.age}
  Genome Length: {len(best_agent.genome)}
  Parents: {', '.join(best_agent.parent_ids) if best_agent.parent_ids else 'Genesis'}

  Dominant Traits:
"""
        for trait, value in sorted(best_agent.traits.items(), key=lambda x: x[1], reverse=True)[:3]:
            report += f"    • {trait}: {value:.3f}\n"

        return report


if __name__ == "__main__":
    # Example usage and testing
    print("🧬 Synexs Genetic Recombination System")
    print("=" * 60)

    # Initialize recombinator
    recombinator = GeneticRecombinator(mutation_rate=0.05)

    # Create initial population (Genesis generation)
    print("\n📍 Creating Genesis Generation...")
    genesis_genomes = [
        ['SCAN', 'ATTACK', 'REPLICATE'],
        ['EVADE', 'LEARN', 'DEFEND'],
        ['SCAN', 'ENCRYPT', 'REPORT'],
        ['ATTACK', 'LEARN', 'MUTATE']
    ]

    genesis_agents = []
    for i, genome in enumerate(genesis_genomes):
        agent = recombinator.register_agent(
            agent_id=f"genesis_{i}",
            genome=genome,
            fitness=random.uniform(0.5, 0.9)
        )
        genesis_agents.append(agent.agent_id)
        print(f"  ✓ {agent.agent_id}: {' → '.join(genome)}")

    # Simulate evolution
    print("\n🔬 Evolving Generation 1...")
    gen1 = recombinator.evolve_generation(num_offspring=6)

    for offspring in gen1:
        print(f"\n  Offspring: {offspring.agent_id}")
        print(f"    Parents: {', '.join(offspring.parent_ids)}")
        print(f"    Genome: {' → '.join(offspring.genome)}")
        print(f"    Dominant trait: {max(offspring.traits.items(), key=lambda x: x[1])}")

    # Update fitness based on "mission results"
    print("\n📊 Simulating mission results...")
    for offspring in gen1:
        # Random fitness update
        delta = random.uniform(-0.1, 0.3)
        recombinator.update_fitness(offspring.agent_id, delta)
        print(f"  {offspring.agent_id}: Fitness {offspring.fitness_score:.3f}")

    # Evolve another generation
    print("\n🔬 Evolving Generation 2...")
    gen2 = recombinator.evolve_generation(num_offspring=8)

    # Print population report
    print(recombinator.get_population_report())

    # Show family tree of best agent
    best_id = max(recombinator.gene_pool.keys(),
                 key=lambda x: recombinator.gene_pool[x].fitness_score)
    print(f"\n🌳 Family Tree of Best Agent ({best_id}):")
    print(json.dumps(recombinator.get_family_tree(best_id, depth=2), indent=2))

    # Export gene pool
    recombinator.export_gene_pool('/root/synexs/gene_pool_export.json')

    print("\n✅ Genetic recombination system demonstration complete!")
