#!/usr/bin/env python3
"""
Synexs Adaptive Immune System
Biological immune system for digital organisms

Implements two-layer immunity like biological systems:
1. Innate Immunity: Fast, generic responses to known threat patterns
2. Adaptive Immunity: Learns specific threats, creates antibodies, remembers

Key features:
- Antigen recognition (threat fingerprinting)
- Antibody generation (specific defenses)
- Memory B-cells (long-term threat memory)
- Clonal selection (amplify successful responses)
"""

import time
import json
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import numpy as np

@dataclass
class Antigen:
    """
    Foreign threat (like a pathogen in biology)
    Represents attack patterns, honeypots, defensive mechanisms
    """
    antigen_id: str
    threat_type: str  # 'honeypot', 'ids', 'waf', 'rate_limit', 'firewall'
    signature: str  # Unique fingerprint of the threat
    features: Dict[str, Any]  # Characteristics for recognition
    danger_level: float  # 0.0 - 1.0
    first_seen: float
    last_seen: float
    encounter_count: int = 0

@dataclass
class Antibody:
    """
    Specific defense against a threat (like biological antibodies)
    Generated in response to antigens
    """
    antibody_id: str
    target_antigen: str  # Which threat this defends against
    strategy: Dict[str, Any]  # How to counter the threat
    effectiveness: float  # 0.0 - 1.0 (how well it works)
    affinity: float  # 0.0 - 1.0 (how specific to this threat)
    created_at: float
    uses: int = 0
    success_rate: float = 0.5

@dataclass
class MemoryCell:
    """
    Long-term memory of threats (like memory B-cells)
    Enables faster response to previously encountered threats
    """
    memory_id: str
    antigen_signature: str
    antibody_template: Dict[str, Any]
    strength: float  # Memory strength (decays over time)
    recall_speed: float  # How fast to respond on re-encounter
    created_at: float
    last_recalled: float
    recall_count: int = 0

@dataclass
class ImmuneResponse:
    """
    Active immune response to a threat
    Tracks the body's reaction to an infection
    """
    response_id: str
    antigen: Antigen
    antibodies_deployed: List[str]
    start_time: float
    duration: float
    outcome: str  # 'success', 'failure', 'partial'
    inflammation_level: float  # Resource cost of response

class AdaptiveImmuneSystem:
    """
    Adaptive immune system for Synexs digital organism

    Biological analogy:
    - Antigens = Threats (honeypots, IDS, etc.)
    - Antibodies = Specific defenses/strategies
    - Memory cells = Learned responses
    - Immune response = Active defense
    """

    def __init__(self, memory_decay=0.001, affinity_threshold=0.7):
        self.antigens: Dict[str, Antigen] = {}
        self.antibodies: Dict[str, Antibody] = {}
        self.memory_cells: Dict[str, MemoryCell] = {}
        self.active_responses: Dict[str, ImmuneResponse] = {}

        self.memory_decay = memory_decay
        self.affinity_threshold = affinity_threshold

        # Innate immunity: Pre-configured generic responses
        self.innate_responses = self._initialize_innate_immunity()

        # Statistics
        self.total_encounters = 0
        self.successful_responses = 0
        self.failed_responses = 0

    def _initialize_innate_immunity(self) -> Dict[str, Dict[str, Any]]:
        """
        Innate immunity: Generic, fast responses to common threats
        Like the baseline immune system we're born with
        """
        return {
            'honeypot': {
                'action': 'abort_immediately',
                'reasoning': 'Honeypot indicators detected',
                'effectiveness': 0.8,
                'speed': 0.95  # Very fast response
            },
            'rate_limit': {
                'action': 'slow_down',
                'reasoning': 'Rate limiting detected',
                'effectiveness': 0.7,
                'speed': 0.9
            },
            'firewall': {
                'action': 'change_approach',
                'reasoning': 'Firewall blocking',
                'effectiveness': 0.6,
                'speed': 0.85
            },
            'ids': {
                'action': 'evade_signatures',
                'reasoning': 'IDS pattern matching',
                'effectiveness': 0.65,
                'speed': 0.8
            },
            'waf': {
                'action': 'encode_payload',
                'reasoning': 'WAF filtering',
                'effectiveness': 0.6,
                'speed': 0.75
            }
        }

    def recognize_threat(self, threat_data: Dict[str, Any]) -> Optional[Antigen]:
        """
        Antigen recognition: Identify if this is a threat
        Like how the immune system recognizes pathogens
        """
        # Generate unique signature for this threat
        signature = self._generate_signature(threat_data)

        # Check if we've seen this before
        if signature in [a.signature for a in self.antigens.values()]:
            # Known antigen - update
            antigen = self._find_antigen_by_signature(signature)
            antigen.last_seen = time.time()
            antigen.encounter_count += 1
            self.total_encounters += 1
            return antigen

        # New antigen - create entry
        antigen = Antigen(
            antigen_id=str(uuid.uuid4())[:8],
            threat_type=threat_data.get('type', 'unknown'),
            signature=signature,
            features=threat_data.get('features', {}),
            danger_level=self._assess_danger(threat_data),
            first_seen=time.time(),
            last_seen=time.time(),
            encounter_count=1
        )

        self.antigens[antigen.antigen_id] = antigen
        self.total_encounters += 1

        print(f"🦠 New threat recognized: {antigen.threat_type} (danger: {antigen.danger_level:.2f})")
        return antigen

    def _generate_signature(self, threat_data: Dict[str, Any]) -> str:
        """Generate unique fingerprint for threat"""
        # Combine key features into signature
        key_features = [
            str(threat_data.get('type', '')),
            str(threat_data.get('ptr_record', '')),
            str(threat_data.get('timing_pattern', '')),
            str(sorted(threat_data.get('indicators', [])))
        ]

        signature_string = '|'.join(key_features)
        return hashlib.sha256(signature_string.encode()).hexdigest()[:16]

    def _assess_danger(self, threat_data: Dict[str, Any]) -> float:
        """Assess danger level of threat (0.0 - 1.0)"""
        danger = 0.5  # Base danger

        # Increase danger based on threat characteristics
        if threat_data.get('type') == 'honeypot':
            danger += 0.3
        if threat_data.get('has_deception_tech', False):
            danger += 0.2
        if threat_data.get('detection_likelihood', 0) > 0.7:
            danger += 0.2
        if len(threat_data.get('indicators', [])) > 3:
            danger += 0.1

        return min(1.0, danger)

    def _find_antigen_by_signature(self, signature: str) -> Optional[Antigen]:
        """Find antigen by signature"""
        for antigen in self.antigens.values():
            if antigen.signature == signature:
                return antigen
        return None

    def mount_immune_response(self, antigen: Antigen) -> ImmuneResponse:
        """
        Mount immune response to threat

        Process:
        1. Check memory (have we seen this before?)
        2. If yes: Rapid recall response
        3. If no: Generate new antibodies
        4. Deploy antibodies
        5. Monitor effectiveness
        """
        response_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Check for memory recall
        memory = self._check_memory(antigen.signature)

        if memory:
            # Memory recall: Fast, specific response
            print(f"💭 Memory recalled for {antigen.threat_type}")
            antibodies = self._recall_antibodies(memory)
            inflammation = 0.2  # Low resource cost (memory response)
            memory.last_recalled = time.time()
            memory.recall_count += 1
        else:
            # Primary response: Generate new antibodies
            print(f"🔬 Generating new antibodies for {antigen.threat_type}")
            antibodies = self._generate_antibodies(antigen)
            inflammation = 0.8  # High resource cost (new response)

        # Deploy antibodies
        deployed_ids = [ab.antibody_id for ab in antibodies]

        response = ImmuneResponse(
            response_id=response_id,
            antigen=antigen,
            antibodies_deployed=deployed_ids,
            start_time=start_time,
            duration=0.0,  # Will be updated
            outcome='pending',
            inflammation_level=inflammation
        )

        self.active_responses[response_id] = response
        return response

    def _check_memory(self, antigen_signature: str) -> Optional[MemoryCell]:
        """Check if we have memory of this threat"""
        # Decay old memories
        self._decay_memories()

        # Find matching memory
        for memory in self.memory_cells.values():
            if memory.antigen_signature == antigen_signature:
                # Memory strength must be above threshold
                if memory.strength > 0.3:
                    return memory

        return None

    def _decay_memories(self):
        """Decay memory strength over time (like forgetting)"""
        current_time = time.time()

        for memory in list(self.memory_cells.values()):
            # Time since last recall
            time_since_recall = current_time - memory.last_recalled
            days_elapsed = time_since_recall / 86400.0

            # Exponential decay
            memory.strength *= np.exp(-self.memory_decay * days_elapsed)

            # Remove very weak memories
            if memory.strength < 0.1:
                del self.memory_cells[memory.memory_id]

    def _recall_antibodies(self, memory: MemoryCell) -> List[Antibody]:
        """Recall antibodies from memory (fast response)"""
        # Create antibodies from memory template
        antibodies = []

        # Memory provides template for rapid antibody production
        template = memory.antibody_template

        antibody = Antibody(
            antibody_id=str(uuid.uuid4())[:8],
            target_antigen=memory.antigen_signature,
            strategy=template['strategy'],
            effectiveness=template['effectiveness'],
            affinity=0.9,  # High affinity from memory
            created_at=time.time()
        )

        antibodies.append(antibody)
        self.antibodies[antibody.antibody_id] = antibody

        return antibodies

    def _generate_antibodies(self, antigen: Antigen) -> List[Antibody]:
        """
        Generate new antibodies for novel threat
        Like clonal selection in adaptive immunity
        """
        antibodies = []

        # Check innate immunity first
        if antigen.threat_type in self.innate_responses:
            innate = self.innate_responses[antigen.threat_type]
            strategy = {
                'type': 'innate',
                'action': innate['action'],
                'reasoning': innate['reasoning']
            }
            effectiveness = innate['effectiveness']
        else:
            # Novel threat - adaptive response
            strategy = self._devise_strategy(antigen)
            effectiveness = 0.5  # Unknown effectiveness initially

        # Generate primary antibody
        antibody = Antibody(
            antibody_id=str(uuid.uuid4())[:8],
            target_antigen=antigen.antigen_id,
            strategy=strategy,
            effectiveness=effectiveness,
            affinity=self._calculate_affinity(antigen, strategy),
            created_at=time.time()
        )

        antibodies.append(antibody)
        self.antibodies[antibody.antibody_id] = antibody

        # Generate variants (somatic hypermutation in biology)
        for _ in range(2):
            variant_strategy = self._mutate_strategy(strategy)
            variant = Antibody(
                antibody_id=str(uuid.uuid4())[:8],
                target_antigen=antigen.antigen_id,
                strategy=variant_strategy,
                effectiveness=effectiveness * 0.8,  # Slightly lower
                affinity=self._calculate_affinity(antigen, variant_strategy),
                created_at=time.time()
            )
            antibodies.append(variant)
            self.antibodies[variant.antibody_id] = variant

        return antibodies

    def _devise_strategy(self, antigen: Antigen) -> Dict[str, Any]:
        """Devise counter-strategy for threat"""
        strategies = {
            'honeypot': {
                'action': 'abort',
                'tactics': ['validate_ptr', 'check_timing', 'analyze_behavior'],
                'reasoning': 'Honeypot detection and avoidance'
            },
            'rate_limit': {
                'action': 'throttle',
                'tactics': ['reduce_frequency', 'randomize_timing', 'distribute_load'],
                'reasoning': 'Adapt to rate limiting'
            },
            'firewall': {
                'action': 'circumvent',
                'tactics': ['change_ports', 'fragment_packets', 'tunnel'],
                'reasoning': 'Bypass firewall rules'
            },
            'ids': {
                'action': 'evade',
                'tactics': ['polymorphic_payloads', 'encoding', 'obfuscation'],
                'reasoning': 'Evade signature detection'
            }
        }

        # Get strategy for threat type or default
        return strategies.get(
            antigen.threat_type,
            {
                'action': 'analyze_and_adapt',
                'tactics': ['observe', 'test', 'learn'],
                'reasoning': 'Unknown threat - cautious exploration'
            }
        )

    def _mutate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create variant strategy (somatic hypermutation)"""
        mutated = strategy.copy()

        # Randomly modify tactics
        if 'tactics' in mutated:
            tactics = mutated['tactics'].copy()
            if tactics and len(tactics) > 1:
                # Swap order
                import random
                random.shuffle(tactics)
                mutated['tactics'] = tactics

        return mutated

    def _calculate_affinity(self, antigen: Antigen, strategy: Dict[str, Any]) -> float:
        """Calculate how well this strategy matches the threat"""
        # Simple affinity calculation
        # In reality, would be based on strategy effectiveness
        base_affinity = 0.6

        # Bonus for matching threat type
        if strategy.get('type') == 'innate' and antigen.threat_type in self.innate_responses:
            base_affinity += 0.2

        return min(1.0, base_affinity)

    def report_outcome(self, response_id: str, success: bool, details: Dict[str, Any]):
        """
        Report outcome of immune response
        Used to update antibody effectiveness and create memory
        """
        if response_id not in self.active_responses:
            return

        response = self.active_responses[response_id]
        response.duration = time.time() - response.start_time
        response.outcome = 'success' if success else 'failure'

        if success:
            self.successful_responses += 1
        else:
            self.failed_responses += 1

        # Update antibody effectiveness
        for ab_id in response.antibodies_deployed:
            if ab_id in self.antibodies:
                antibody = self.antibodies[ab_id]
                antibody.uses += 1

                # Update success rate
                old_rate = antibody.success_rate
                antibody.success_rate = (
                    (old_rate * (antibody.uses - 1) + (1.0 if success else 0.0)) /
                    antibody.uses
                )

                # Update effectiveness
                if success:
                    antibody.effectiveness = min(1.0, antibody.effectiveness * 1.1)
                else:
                    antibody.effectiveness = max(0.0, antibody.effectiveness * 0.9)

        # Clonal selection: Amplify successful antibodies
        if success:
            self._clonal_selection(response)

        # Create memory for successful responses
        if success and response.antigen.danger_level > 0.6:
            self._create_memory(response)

        # Clean up
        del self.active_responses[response_id]

    def _clonal_selection(self, response: ImmuneResponse):
        """
        Clonal selection: Amplify production of successful antibodies
        High-affinity antibodies are selected and proliferated
        """
        for ab_id in response.antibodies_deployed:
            if ab_id not in self.antibodies:
                continue

            antibody = self.antibodies[ab_id]

            # Only amplify high-affinity, effective antibodies
            if antibody.affinity > self.affinity_threshold and antibody.effectiveness > 0.7:
                # Create clones
                for _ in range(2):
                    clone = Antibody(
                        antibody_id=str(uuid.uuid4())[:8],
                        target_antigen=antibody.target_antigen,
                        strategy=antibody.strategy.copy(),
                        effectiveness=antibody.effectiveness,
                        affinity=antibody.affinity,
                        created_at=time.time()
                    )
                    self.antibodies[clone.antibody_id] = clone

    def _create_memory(self, response: ImmuneResponse):
        """
        Create memory cell for long-term immunity
        Enables rapid response on re-encounter
        """
        # Find best antibody from response
        best_antibody = None
        best_effectiveness = 0.0

        for ab_id in response.antibodies_deployed:
            if ab_id in self.antibodies:
                ab = self.antibodies[ab_id]
                if ab.effectiveness > best_effectiveness:
                    best_effectiveness = ab.effectiveness
                    best_antibody = ab

        if not best_antibody:
            return

        # Create memory cell
        memory = MemoryCell(
            memory_id=str(uuid.uuid4())[:8],
            antigen_signature=response.antigen.signature,
            antibody_template={
                'strategy': best_antibody.strategy,
                'effectiveness': best_antibody.effectiveness
            },
            strength=1.0,  # Full strength initially
            recall_speed=0.95,  # Very fast recall
            created_at=time.time(),
            last_recalled=time.time()
        )

        self.memory_cells[memory.memory_id] = memory
        print(f"💾 Memory created for {response.antigen.threat_type}")

    def get_immune_status(self) -> Dict[str, Any]:
        """Get current immune system status"""
        return {
            'antigens_known': len(self.antigens),
            'antibodies_available': len(self.antibodies),
            'memory_cells': len(self.memory_cells),
            'active_responses': len(self.active_responses),
            'total_encounters': self.total_encounters,
            'success_rate': (
                self.successful_responses / self.total_encounters
                if self.total_encounters > 0 else 0.0
            ),
            'innate_immunity': list(self.innate_responses.keys())
        }

    def export_immune_memory(self, filepath: str):
        """Export immune memory for transfer/backup"""
        export_data = {
            'antigens': {aid: asdict(a) for aid, a in self.antigens.items()},
            'antibodies': {aid: asdict(a) for aid, a in self.antibodies.items()},
            'memory_cells': {mid: asdict(m) for mid, m in self.memory_cells.items()},
            'statistics': {
                'total_encounters': self.total_encounters,
                'successful_responses': self.successful_responses,
                'failed_responses': self.failed_responses
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Immune memory exported to {filepath}")


if __name__ == "__main__":
    # Example usage and testing
    print("🦠 Synexs Adaptive Immune System")
    print("=" * 60)

    # Initialize immune system
    immune = AdaptiveImmuneSystem()

    # Simulate encountering threats
    print("\n📍 Encounter 1: Honeypot")
    threat1 = {
        'type': 'honeypot',
        'ptr_record': 'honeypot.example.com',
        'timing_pattern': 'suspicious',
        'indicators': ['fake_vulns', 'unrealistic_services', 'bait_files'],
        'detection_likelihood': 0.85
    }

    antigen1 = immune.recognize_threat(threat1)
    response1 = immune.mount_immune_response(antigen1)

    # Simulate response outcome
    time.sleep(0.1)
    immune.report_outcome(response1.response_id, success=True, details={})

    # Encounter same threat again (should use memory)
    print("\n📍 Encounter 2: Same honeypot (memory recall)")
    antigen2 = immune.recognize_threat(threat1)  # Same threat
    response2 = immune.mount_immune_response(antigen2)
    immune.report_outcome(response2.response_id, success=True, details={})

    # New threat
    print("\n📍 Encounter 3: Rate limiter")
    threat3 = {
        'type': 'rate_limit',
        'max_requests': 5,
        'window': 10,
        'detection_likelihood': 0.6
    }

    antigen3 = immune.recognize_threat(threat3)
    response3 = immune.mount_immune_response(antigen3)
    immune.report_outcome(response3.response_id, success=True, details={})

    # Status report
    print("\n" + "=" * 60)
    print("IMMUNE SYSTEM STATUS:")
    print("=" * 60)
    status = immune.get_immune_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Export memory
    immune.export_immune_memory('/root/synexs/immune_memory.json')

    print("\n✅ Adaptive immune system demonstration complete!")
