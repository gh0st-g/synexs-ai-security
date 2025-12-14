#!/usr/bin/env python3
"""
Synexs Cell Differentiation Engine
Dynamic role specialization based on environmental needs

Biological analogy:
- Stem cells can differentiate into specialized cell types
- Cells change roles based on chemical signals and needs
- Differentiation is guided by gene expression
- Specialized cells are more efficient at specific tasks
"""

import time
import json
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class CellType(Enum):
    """Possible cell specializations"""
    STEM = "stem"                    # Undifferentiated, can become anything
    SCOUT = "scout"                  # Reconnaissance specialist
    ANALYZER = "analyzer"            # Analysis and decision-making
    EXECUTOR = "executor"            # Action execution
    DEFENDER = "defender"            # Protection and defense
    LEARNER = "learner"             # Learning and adaptation
    COMMUNICATOR = "communicator"    # Inter-cell coordination
    REPLICATOR = "replicator"       # Cell division and growth

class DifferentiationSignal(Enum):
    """Environmental signals that trigger differentiation"""
    THREAT_DETECTED = "threat_detected"
    RESOURCES_LOW = "resources_low"
    LEARNING_NEEDED = "learning_needed"
    COORDINATION_REQUIRED = "coordination_required"
    GROWTH_PHASE = "growth_phase"
    DEFENSE_NEEDED = "defense_needed"

@dataclass
class CellProfile:
    """Profile of a specialized cell"""
    cell_id: str
    cell_type: CellType
    specialization_level: float  # 0.0 (stem) to 1.0 (fully specialized)
    capabilities: Set[str]
    efficiency: float  # How good at current role
    plasticity: float  # Ability to change roles (decreases with specialization)
    age: int  # Generations of specialization
    gene_expression: Dict[str, bool]  # Which genes are "on"

@dataclass
class DifferentiationPath:
    """Path from stem cell to specialized cell"""
    origin_type: CellType
    target_type: CellType
    signals_required: List[DifferentiationSignal]
    duration: int  # Cycles to complete
    reversible: bool  # Can go back

class CellDifferentiationEngine:
    """
    Manages cell differentiation and specialization

    Key features:
    - Stem cells differentiate based on environmental needs
    - Cells can dedifferentiate under stress (return to stem)
    - Specialization increases efficiency but reduces flexibility
    - Population maintains balance of cell types
    """

    def __init__(self):
        self.cells: Dict[str, CellProfile] = {}
        self.environment_signals: Set[DifferentiationSignal] = set()

        # Differentiation pathways
        self.pathways = self._initialize_pathways()

        # Population targets (homeostasis)
        self.population_targets = {
            CellType.SCOUT: 0.20,          # 20% scouts
            CellType.ANALYZER: 0.15,       # 15% analyzers
            CellType.EXECUTOR: 0.25,       # 25% executors
            CellType.DEFENDER: 0.15,       # 15% defenders
            CellType.LEARNER: 0.10,        # 10% learners
            CellType.COMMUNICATOR: 0.10,   # 10% communicators
            CellType.REPLICATOR: 0.05      # 5% replicators
        }

        # Cell type capabilities
        self.cell_capabilities = {
            CellType.STEM: {'differentiate', 'divide', 'adapt'},
            CellType.SCOUT: {'scan', 'observe', 'detect', 'report'},
            CellType.ANALYZER: {'analyze', 'decide', 'assess', 'plan'},
            CellType.EXECUTOR: {'attack', 'execute', 'modify', 'deploy'},
            CellType.DEFENDER: {'defend', 'block', 'evade', 'protect'},
            CellType.LEARNER: {'learn', 'remember', 'adapt', 'improve'},
            CellType.COMMUNICATOR: {'broadcast', 'coordinate', 'sync', 'relay'},
            CellType.REPLICATOR: {'replicate', 'mutate', 'spawn', 'clone'}
        }

    def _initialize_pathways(self) -> List[DifferentiationPath]:
        """Initialize differentiation pathways"""
        return [
            DifferentiationPath(
                origin_type=CellType.STEM,
                target_type=CellType.SCOUT,
                signals_required=[DifferentiationSignal.THREAT_DETECTED],
                duration=3,
                reversible=True
            ),
            DifferentiationPath(
                origin_type=CellType.STEM,
                target_type=CellType.ANALYZER,
                signals_required=[DifferentiationSignal.COORDINATION_REQUIRED],
                duration=4,
                reversible=True
            ),
            DifferentiationPath(
                origin_type=CellType.STEM,
                target_type=CellType.EXECUTOR,
                signals_required=[DifferentiationSignal.THREAT_DETECTED],
                duration=3,
                reversible=False
            ),
            DifferentiationPath(
                origin_type=CellType.STEM,
                target_type=CellType.DEFENDER,
                signals_required=[DifferentiationSignal.DEFENSE_NEEDED],
                duration=4,
                reversible=True
            ),
            DifferentiationPath(
                origin_type=CellType.STEM,
                target_type=CellType.LEARNER,
                signals_required=[DifferentiationSignal.LEARNING_NEEDED],
                duration=5,
                reversible=True
            ),
            DifferentiationPath(
                origin_type=CellType.STEM,
                target_type=CellType.COMMUNICATOR,
                signals_required=[DifferentiationSignal.COORDINATION_REQUIRED],
                duration=3,
                reversible=True
            ),
            DifferentiationPath(
                origin_type=CellType.STEM,
                target_type=CellType.REPLICATOR,
                signals_required=[DifferentiationSignal.GROWTH_PHASE],
                duration=4,
                reversible=False
            ),
            # Scout can become Analyzer (lateral differentiation)
            DifferentiationPath(
                origin_type=CellType.SCOUT,
                target_type=CellType.ANALYZER,
                signals_required=[DifferentiationSignal.COORDINATION_REQUIRED],
                duration=2,
                reversible=True
            ),
            # Defender can become Executor under threat
            DifferentiationPath(
                origin_type=CellType.DEFENDER,
                target_type=CellType.EXECUTOR,
                signals_required=[DifferentiationSignal.THREAT_DETECTED],
                duration=2,
                reversible=True
            )
        ]

    def create_stem_cell(self) -> CellProfile:
        """Create a new undifferentiated stem cell"""
        cell = CellProfile(
            cell_id=str(uuid.uuid4())[:8],
            cell_type=CellType.STEM,
            specialization_level=0.0,
            capabilities=self.cell_capabilities[CellType.STEM].copy(),
            efficiency=0.5,  # Moderate efficiency at everything
            plasticity=1.0,  # Maximum flexibility
            age=0,
            gene_expression={}
        )

        self.cells[cell.cell_id] = cell
        return cell

    def differentiate_cell(self, cell_id: str, target_type: CellType,
                          signal: Optional[DifferentiationSignal] = None) -> bool:
        """
        Differentiate a cell into a specialized type

        Returns True if differentiation successful, False otherwise
        """
        if cell_id not in self.cells:
            return False

        cell = self.cells[cell_id]

        # Check if pathway exists
        pathway = self._find_pathway(cell.cell_type, target_type)
        if not pathway:
            print(f"❌ No differentiation pathway from {cell.cell_type.value} to {target_type.value}")
            return False

        # Check if required signals are present
        if signal:
            self.environment_signals.add(signal)

        signals_present = all(
            sig in self.environment_signals
            for sig in pathway.signals_required
        )

        if not signals_present:
            print(f"❌ Required signals not present for differentiation")
            return False

        # Check plasticity
        if cell.plasticity < 0.3:
            print(f"❌ Cell {cell_id} is too specialized to change")
            return False

        # Perform differentiation
        print(f"🧬 Differentiating {cell_id}: {cell.cell_type.value} → {target_type.value}")

        # Update cell properties
        cell.cell_type = target_type
        cell.specialization_level = min(1.0, cell.specialization_level + 0.3)
        cell.capabilities = self.cell_capabilities[target_type].copy()
        cell.efficiency = 0.6 + (cell.specialization_level * 0.4)  # More specialized = more efficient
        cell.plasticity = max(0.1, 1.0 - cell.specialization_level)  # Less plastic as specialize
        cell.age += 1

        # Update gene expression
        cell.gene_expression[f"{target_type.value}_genes"] = True

        return True

    def dedifferentiate_cell(self, cell_id: str) -> bool:
        """
        Dedifferentiate cell back to stem cell (under stress/need)
        Only possible if pathway is reversible
        """
        if cell_id not in self.cells:
            return False

        cell = self.cells[cell_id]

        if cell.cell_type == CellType.STEM:
            return True  # Already stem

        # Check if can dedifferentiate (requires reversible pathway)
        pathway = self._find_pathway(CellType.STEM, cell.cell_type)
        if not pathway or not pathway.reversible:
            print(f"❌ Cell {cell_id} cannot dedifferentiate (irreversible)")
            return False

        # Check if too specialized
        if cell.specialization_level > 0.9:
            print(f"❌ Cell {cell_id} is too specialized to dedifferentiate")
            return False

        print(f"🔄 Dedifferentiating {cell_id}: {cell.cell_type.value} → stem")

        # Revert to stem
        cell.cell_type = CellType.STEM
        cell.specialization_level = max(0.0, cell.specialization_level - 0.4)
        cell.capabilities = self.cell_capabilities[CellType.STEM].copy()
        cell.efficiency = 0.5
        cell.plasticity = min(1.0, cell.plasticity + 0.3)

        # Clear specific gene expression
        cell.gene_expression = {}

        return True

    def _find_pathway(self, origin: CellType, target: CellType) -> Optional[DifferentiationPath]:
        """Find differentiation pathway between cell types"""
        for pathway in self.pathways:
            if pathway.origin_type == origin and pathway.target_type == target:
                return pathway
        return None

    def assess_population_needs(self) -> Dict[CellType, float]:
        """
        Assess what cell types are needed based on population balance
        Returns needed ratio for each type
        """
        # Count current population
        population = {cell_type: 0 for cell_type in CellType}
        for cell in self.cells.values():
            population[cell.cell_type] += 1

        total_cells = len(self.cells)
        if total_cells == 0:
            return {cell_type: 1.0 for cell_type in CellType if cell_type != CellType.STEM}

        # Calculate needs
        needs = {}
        for cell_type, target_ratio in self.population_targets.items():
            current_ratio = population[cell_type] / total_cells
            deficit = target_ratio - current_ratio
            needs[cell_type] = max(0.0, deficit)

        return needs

    def auto_differentiate_population(self) -> int:
        """
        Automatically differentiate cells based on population needs
        Returns number of cells differentiated
        """
        needs = self.assess_population_needs()

        # Find stem cells
        stem_cells = [
            cell_id for cell_id, cell in self.cells.items()
            if cell.cell_type == CellType.STEM
        ]

        if not stem_cells:
            print("⚠️ No stem cells available for differentiation")
            return 0

        # Prioritize needed cell types
        sorted_needs = sorted(needs.items(), key=lambda x: x[1], reverse=True)

        differentiated = 0
        for cell_type, need in sorted_needs:
            if need > 0 and stem_cells:
                # Differentiate one stem cell to this type
                stem_id = stem_cells.pop(0)

                # Determine required signal
                signal = self._get_signal_for_type(cell_type)

                if self.differentiate_cell(stem_id, cell_type, signal):
                    differentiated += 1

        return differentiated

    def _get_signal_for_type(self, cell_type: CellType) -> Optional[DifferentiationSignal]:
        """Get appropriate signal for cell type"""
        signal_map = {
            CellType.SCOUT: DifferentiationSignal.THREAT_DETECTED,
            CellType.ANALYZER: DifferentiationSignal.COORDINATION_REQUIRED,
            CellType.EXECUTOR: DifferentiationSignal.THREAT_DETECTED,
            CellType.DEFENDER: DifferentiationSignal.DEFENSE_NEEDED,
            CellType.LEARNER: DifferentiationSignal.LEARNING_NEEDED,
            CellType.COMMUNICATOR: DifferentiationSignal.COORDINATION_REQUIRED,
            CellType.REPLICATOR: DifferentiationSignal.GROWTH_PHASE
        }
        return signal_map.get(cell_type)

    def emit_signal(self, signal: DifferentiationSignal):
        """Emit environmental signal"""
        self.environment_signals.add(signal)
        print(f"📡 Signal emitted: {signal.value}")

    def clear_signals(self):
        """Clear all environmental signals"""
        self.environment_signals.clear()

    def get_population_report(self) -> str:
        """Generate population report"""
        if not self.cells:
            return "No cells in population"

        # Count by type
        population = {cell_type: 0 for cell_type in CellType}
        for cell in self.cells.values():
            population[cell.cell_type] += 1

        total = len(self.cells)

        # Calculate averages
        avg_specialization = np.mean([c.specialization_level for c in self.cells.values()])
        avg_efficiency = np.mean([c.efficiency for c in self.cells.values()])
        avg_plasticity = np.mean([c.plasticity for c in self.cells.values()])

        report = f"""
╔════════════════════════════════════════════════════════════╗
║           SYNEXS CELL POPULATION REPORT                    ║
╚════════════════════════════════════════════════════════════╝

Total Cells: {total}

CELL TYPE DISTRIBUTION:
"""

        for cell_type in CellType:
            count = population[cell_type]
            ratio = count / total if total > 0 else 0
            target = self.population_targets.get(cell_type, 0)
            bar = '█' * int(ratio * 40)

            status = "✓" if abs(ratio - target) < 0.05 else "⚠"
            report += f"  {status} {cell_type.value:15s}: [{bar:40s}] {ratio:5.1%} (target: {target:5.1%})\n"

        report += f"""
POPULATION METRICS:
  Average Specialization: {avg_specialization:.3f}
  Average Efficiency: {avg_efficiency:.3f}
  Average Plasticity: {avg_plasticity:.3f}

ACTIVE SIGNALS:
"""
        if self.environment_signals:
            for signal in self.environment_signals:
                report += f"  • {signal.value}\n"
        else:
            report += "  (none)\n"

        return report

    def export_population(self, filepath: str):
        """Export cell population data"""
        export_data = {
            'cells': {
                cell_id: asdict(cell)
                for cell_id, cell in self.cells.items()
            },
            'environment_signals': [s.value for s in self.environment_signals],
            'population_targets': {
                ct.value: ratio for ct, ratio in self.population_targets.items()
            }
        }

        # Convert sets to lists for JSON
        for cell_data in export_data['cells'].values():
            if 'capabilities' in cell_data:
                cell_data['capabilities'] = list(cell_data['capabilities'])
            if 'cell_type' in cell_data:
                cell_data['cell_type'] = cell_data['cell_type'].value

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Cell population exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("🧬 Synexs Cell Differentiation Engine")
    print("=" * 60)

    # Initialize engine
    engine = CellDifferentiationEngine()

    # Create initial stem cell population
    print("\n📍 Creating 20 stem cells...")
    for i in range(20):
        engine.create_stem_cell()

    print(engine.get_population_report())

    # Emit signals based on environmental needs
    print("\n📡 Emitting environmental signals...")
    engine.emit_signal(DifferentiationSignal.THREAT_DETECTED)
    engine.emit_signal(DifferentiationSignal.COORDINATION_REQUIRED)
    engine.emit_signal(DifferentiationSignal.LEARNING_NEEDED)

    # Auto-differentiate based on needs
    print("\n🔬 Auto-differentiating cells based on population needs...")
    diff_count = engine.auto_differentiate_population()
    print(f"  Differentiated {diff_count} cells")

    print(engine.get_population_report())

    # Simulate stress - dedifferentiate some cells
    print("\n⚠️ Simulating stress - dedifferentiating cells...")
    scouts = [cid for cid, c in engine.cells.items() if c.cell_type == CellType.SCOUT]
    if scouts:
        engine.dedifferentiate_cell(scouts[0])

    # Add more signals and differentiate again
    print("\n📡 Emitting growth signal...")
    engine.emit_signal(DifferentiationSignal.GROWTH_PHASE)

    diff_count = engine.auto_differentiate_population()
    print(f"  Differentiated {diff_count} more cells")

    # Final report
    print(engine.get_population_report())

    # Export
    engine.export_population('/root/synexs/cell_population.json')

    print("\n✅ Cell differentiation demonstration complete!")
