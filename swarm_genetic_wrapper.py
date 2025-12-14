#!/usr/bin/env python3
"""
Swarm Genetic Evolution Wrapper
Adds true genetic evolution to agent swarm
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Add synexs to path
sys.path.append('/root/synexs')

# Import genetic system
from synexs_genetic_recombination import GeneticRecombinator, GeneticProfile

# Global genetics instance
_genetics: Optional[GeneticRecombinator] = None
_genetic_log_file = Path('/root/synexs/datasets/genomes/genetic_evolution.json')
_agent_registry: Dict[str, Dict[str, Any]] = {}  # agent_id -> {genome, fitness, stats}


def get_genetics() -> GeneticRecombinator:
    """Get or create global genetics instance"""
    global _genetics

    if _genetics is None:
        _genetics = GeneticRecombinator(mutation_rate=0.05)
        print("🧬 Genetic system initialized for swarm")

        # Load previous genetic history if exists
        _load_genetic_history()

    return _genetics


def _load_genetic_history():
    """Load genetic history from previous sessions"""
    if not _genetic_log_file.exists():
        return

    try:
        with open(_genetic_log_file, 'r') as f:
            history = json.load(f)

        # Restore gene pool
        for agent_data in history.get('gene_pool', []):
            genetics = get_genetics()
            genetics.register_agent(
                agent_id=agent_data['agent_id'],
                genome=agent_data['genome'],
                fitness=agent_data.get('fitness', 0.5)
            )

        print(f"  ✓ Loaded {len(history.get('gene_pool', []))} genomes")

    except Exception as e:
        print(f"  ⚠️ Failed to load genetic history: {e}")


def _save_genetic_history():
    """Save genetic history for future sessions"""
    try:
        genetics = get_genetics()

        history = {
            'timestamp': datetime.now().isoformat(),
            'generation': genetics.generation,
            'gene_pool': []
        }

        # Save all agents in gene pool
        for agent_id, profile in genetics.gene_pool.items():
            history['gene_pool'].append({
                'agent_id': profile.agent_id,
                'generation': profile.generation,
                'genome': profile.genome,
                'fitness': profile.fitness,
                'parent_ids': profile.parent_ids
            })

        _genetic_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_genetic_log_file, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"  ✓ Saved {len(history['gene_pool'])} genomes to genetic history")

    except Exception as e:
        print(f"  ⚠️ Failed to save genetic history: {e}")


def extract_genome_from_agent(agent_file: Path) -> List[str]:
    """
    Extract behavioral genome from agent code
    """
    try:
        content = agent_file.read_text()
        genome = []

        # Map code patterns to genes
        gene_patterns = {
            'SCAN': ['socket', 'nmap', 'port scan', 'reconnaissance'],
            'EVADE': ['obfuscate', 'encode', 'encrypt', 'polymorphic'],
            'ATTACK': ['exploit', 'payload', 'inject', 'overflow'],
            'DEFEND': ['honeypot', 'detect', 'block', 'firewall'],
            'LEARN': ['model', 'train', 'learn', 'adapt', 'feedback'],
            'REPORT': ['log', 'report', 'send', 'communicate'],
            'ADAPT': ['mutate', 'evolve', 'modify', 'self-'],
            'EXFILTRATE': ['exfil', 'download', 'upload', 'transfer'],
            'STEALTH': ['stealth', 'hide', 'covert', 'silent'],
            'REPLICATE': ['replicate', 'copy', 'spawn', 'clone']
        }

        for gene, patterns in gene_patterns.items():
            if any(pattern.lower() in content.lower() for pattern in patterns):
                genome.append(gene)

        # Ensure minimum genome size
        if len(genome) < 3:
            genome.extend(['SCAN', 'LEARN', 'ADAPT'][:3 - len(genome)])

        return genome

    except Exception as e:
        print(f"  ⚠️ Failed to extract genome from {agent_file.name}: {e}")
        return ['SCAN', 'LEARN', 'DEFEND']  # Default genome


def calculate_agent_fitness(agent_id: str, stats: Dict[str, Any]) -> float:
    """
    Calculate agent fitness from performance stats
    """
    fitness = 0.5  # Base fitness

    # Success rate
    if 'success_rate' in stats:
        fitness += stats['success_rate'] * 0.4

    # Survival time (longer is better)
    if 'survival_time' in stats:
        survival_score = min(1.0, stats['survival_time'] / 3600)  # Normalize to 1 hour
        fitness += survival_score * 0.2

    # Stealth (not detected = better)
    if 'detected' in stats:
        stealth_score = 0.0 if stats['detected'] else 1.0
        fitness += stealth_score * 0.2
    else:
        fitness += 0.2  # Assume stealthy if no detection data

    # Innovation (unique patterns)
    if 'unique_actions' in stats:
        innovation_score = min(1.0, stats['unique_actions'] / 10)
        fitness += innovation_score * 0.1

    # Efficiency (low resource usage)
    if 'resource_efficiency' in stats:
        fitness += stats['resource_efficiency'] * 0.1

    # Clamp fitness to [0, 1]
    return max(0.0, min(1.0, fitness))


def register_agent(agent_id: str, agent_file: Path, stats: Dict[str, Any]) -> bool:
    """
    Register an agent in the genetic pool
    """
    try:
        genetics = get_genetics()

        # Extract genome from agent code
        genome = extract_genome_from_agent(agent_file)

        # Calculate fitness
        fitness = calculate_agent_fitness(agent_id, stats)

        # Register with genetic system
        genetics.register_agent(
            agent_id=agent_id,
            genome=genome,
            fitness=fitness
        )

        # Store in registry
        _agent_registry[agent_id] = {
            'genome': genome,
            'fitness': fitness,
            'stats': stats,
            'file': str(agent_file),
            'registered_at': datetime.now().isoformat()
        }

        print(f"  ✓ Registered agent {agent_id} (fitness: {fitness:.2f}, genome: {genome[:3]}...)")

        return True

    except Exception as e:
        print(f"  ❌ Failed to register agent {agent_id}: {e}")
        return False


def evolve_new_generation(num_offspring: int = 5) -> List[GeneticProfile]:
    """
    Evolve new generation of agents
    """
    genetics = get_genetics()

    # Check if we have enough agents to breed
    if len(genetics.gene_pool) < 2:
        print("⚠️ Not enough agents in gene pool for evolution (need >= 2)")
        return []

    print(f"\n🧬 Evolving new generation ({num_offspring} offspring)...")

    # Get population report
    report = genetics.get_population_report()
    print(report)

    # Evolve
    offspring = genetics.evolve_generation(num_offspring=num_offspring)

    # Save genetic history
    _save_genetic_history()

    print(f"\n✅ Evolution complete!")
    print(f"   Generation: {genetics.generation}")
    print(f"   Offspring created: {len(offspring)}")

    for i, child in enumerate(offspring):
        print(f"   [{i+1}] {child.agent_id}: {' → '.join(child.genome[:5])}{'...' if len(child.genome) > 5 else ''}")

    return offspring


def genome_to_agent_code(genome: List[str], agent_id: str) -> str:
    """
    Generate agent code from genome
    """
    code_template = f'''#!/usr/bin/env python3
"""
Genetically Evolved Agent: {agent_id}
Generation: {get_genetics().generation}
Genome: {' → '.join(genome)}

This agent was created through sexual reproduction and mutation.
Its genome encodes its behavioral patterns.
"""

import sys
import time
import random

# Genome: {genome}
GENOME = {genome}

class Agent:
    """Genetically evolved agent"""

    def __init__(self):
        self.agent_id = "{agent_id}"
        self.genome = GENOME
        self.generation = {get_genetics().generation}

    def execute(self):
        """Execute agent behavior based on genome"""
        for gene in self.genome:
            self._express_gene(gene)

    def _express_gene(self, gene: str):
        """Express a single gene (execute behavior)"""
'''

    # Add gene expression methods
    gene_implementations = {
        'SCAN': '''
        if gene == "SCAN":
            # Network scanning behavior
            print(f"[{self.agent_id}] Scanning network...")
            # TODO: Implement scanning logic
''',
        'EVADE': '''
        elif gene == "EVADE":
            # Evasion behavior
            print(f"[{self.agent_id}] Evading detection...")
            # TODO: Implement evasion logic
''',
        'ATTACK': '''
        elif gene == "ATTACK":
            # Attack behavior
            print(f"[{self.agent_id}] Executing attack...")
            # TODO: Implement attack logic
''',
        'LEARN': '''
        elif gene == "LEARN":
            # Learning behavior
            print(f"[{self.agent_id}] Learning from environment...")
            # TODO: Implement learning logic
''',
        'DEFEND': '''
        elif gene == "DEFEND":
            # Defense behavior
            print(f"[{self.agent_id}] Defensive posture...")
            # TODO: Implement defense logic
''',
        'REPORT': '''
        elif gene == "REPORT":
            # Reporting behavior
            print(f"[{self.agent_id}] Reporting status...")
            # TODO: Implement reporting logic
'''
    }

    # Add implementations for genes in genome
    for gene in genome:
        if gene in gene_implementations:
            code_template += gene_implementations[gene]

    code_template += '''

if __name__ == "__main__":
    agent = Agent()
    print(f"Agent {agent.agent_id} activated (Generation {agent.generation})")
    print(f"Genome: {' → '.join(agent.genome)}")
    agent.execute()
'''

    return code_template


def create_offspring_files(offspring: List[GeneticProfile], output_dir: Path) -> List[Path]:
    """
    Create agent files from offspring genomes
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for child in offspring:
        # Generate agent code from genome
        code = genome_to_agent_code(child.genome, child.agent_id)

        # Save to file
        filename = f"agent_gen{child.generation}_{child.agent_id}.py"
        filepath = output_dir / filename

        filepath.write_text(code)
        created_files.append(filepath)

        print(f"  ✓ Created {filename}")

    return created_files


def get_genetic_stats() -> Dict[str, Any]:
    """Get genetic evolution statistics"""
    genetics = get_genetics()

    return {
        'generation': genetics.generation,
        'population_size': len(genetics.gene_pool),
        'total_offspring': genetics.total_offspring,
        'mutation_rate': genetics.mutation_rate,
        'avg_fitness': sum(p.fitness for p in genetics.gene_pool.values()) / max(1, len(genetics.gene_pool)),
        'best_fitness': max((p.fitness for p in genetics.gene_pool.values()), default=0.0)
    }


# Export key functions
__all__ = [
    'register_agent',
    'evolve_new_generation',
    'genome_to_agent_code',
    'create_offspring_files',
    'get_genetic_stats',
    'extract_genome_from_agent',
    'calculate_agent_fitness'
]