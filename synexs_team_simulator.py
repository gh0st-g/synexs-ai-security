#!/usr/bin/env python3
"""
Synexs Team Simulator - Phase 1 Implementation
Multi-agent team coordination and mission execution

This module implements coordinated teams of specialized agents that:
- Work together on complex missions
- Communicate using binary protocol
- Log all interactions for training
- Adapt strategies based on feedback
"""

import time
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class AgentRole(Enum):
    """Specialized agent roles in team"""
    SCOUT = "scout"                    # Reconnaissance and intel gathering
    ANALYZER = "analyzer"              # Vulnerability assessment and analysis
    EXECUTOR = "executor"              # Exploitation and task execution
    EXFILTRATOR = "exfiltrator"       # Data retrieval and extraction
    CLEANER = "cleaner"                # Trace removal and cleanup
    COORDINATOR = "coordinator"        # Team coordination and decision making


class MissionStatus(Enum):
    """Mission execution states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"


@dataclass
class AgentState:
    """Current state of an agent"""
    agent_id: str
    role: AgentRole
    status: str
    location: str
    energy: float  # Resource availability (0-1)
    knowledge: Dict[str, Any]  # What agent knows about environment
    capabilities: List[str]
    last_communication: float


@dataclass
class Message:
    """Inter-agent communication"""
    message_id: str
    timestamp: float
    sender: str
    receiver: str
    protocol: str
    message_type: str
    content: Dict[str, Any]
    size_bytes: int
    latency_ms: float
    information_value: float  # How useful this message was (0-1)


@dataclass
class Decision:
    """Agent decision with rationale"""
    decision_id: str
    timestamp: float
    agent: str
    decision_type: str  # 'proceed', 'abort', 'wait', 'adapt'
    rationale: str
    confidence: float
    factors: Dict[str, float]  # Decision factors with weights
    outcome: Optional[str] = None


@dataclass
class MissionResult:
    """Complete mission execution log"""
    mission_id: str
    timestamp: float
    duration: float
    team_composition: List[str]
    environment: Dict[str, Any]
    communications: List[Message]
    decisions: List[Decision]
    metrics: Dict[str, float]
    status: MissionStatus
    training_label: str  # For supervised learning


class Agent:
    """Individual agent with specialized role"""

    def __init__(self, role: AgentRole, capabilities: List[str]):
        self.agent_id = str(uuid.uuid4())[:8]
        self.role = role
        self.capabilities = capabilities
        self.state = AgentState(
            agent_id=self.agent_id,
            role=role,
            status="idle",
            location="base",
            energy=1.0,
            knowledge={},
            capabilities=capabilities,
            last_communication=time.time()
        )
        self.message_history: List[Message] = []
        self.decision_history: List[Decision] = []

    def observe_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Gather intel based on role capabilities"""
        observations = {}

        if self.role == AgentRole.SCOUT:
            # Scouts gather network topology
            observations['network_map'] = environment.get('topology', {})
            observations['open_ports'] = environment.get('ports', [])
            observations['hosts_discovered'] = environment.get('hosts', [])

        elif self.role == AgentRole.ANALYZER:
            # Analyzers assess security
            observations['security_stack'] = environment.get('defenses', [])
            observations['vulnerabilities'] = environment.get('vulns', [])
            observations['honeypot_indicators'] = environment.get('honeypot_signals', [])

        elif self.role == AgentRole.EXECUTOR:
            # Executors identify targets
            observations['exploitable_targets'] = environment.get('targets', [])
            observations['access_vectors'] = environment.get('vectors', [])

        elif self.role == AgentRole.EXFILTRATOR:
            # Exfiltrators find data paths
            observations['data_locations'] = environment.get('data', [])
            observations['exfil_routes'] = environment.get('routes', [])

        elif self.role == AgentRole.CLEANER:
            # Cleaners identify traces
            observations['log_locations'] = environment.get('logs', [])
            observations['traces_to_remove'] = environment.get('traces', [])

        self.state.knowledge.update(observations)
        return observations

    def make_decision(self, context: Dict[str, Any]) -> Decision:
        """Make decision based on role and context"""
        decision_factors = {
            'risk_level': context.get('risk', 0.5),
            'success_probability': context.get('success_prob', 0.5),
            'detection_likelihood': context.get('detection', 0.5),
            'resource_cost': context.get('cost', 0.5)
        }

        # Calculate confidence based on information availability
        confidence = len(self.state.knowledge) / 10.0  # Simple heuristic
        confidence = min(confidence, 1.0)

        # Role-specific decision logic
        if self.role == AgentRole.SCOUT:
            decision_type = 'proceed' if decision_factors['detection_likelihood'] < 0.7 else 'wait'
            rationale = f"Detection risk: {decision_factors['detection_likelihood']:.2f}"

        elif self.role == AgentRole.ANALYZER:
            honeypot_prob = len(self.state.knowledge.get('honeypot_indicators', [])) / 5.0
            decision_type = 'abort' if honeypot_prob > 0.7 else 'proceed'
            rationale = f"Honeypot probability: {honeypot_prob:.2f}"

        elif self.role == AgentRole.EXECUTOR:
            decision_type = 'proceed' if decision_factors['success_probability'] > 0.6 else 'adapt'
            rationale = f"Success probability: {decision_factors['success_probability']:.2f}"

        else:
            decision_type = 'proceed'
            rationale = "Standard operational procedure"

        decision = Decision(
            decision_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            agent=self.agent_id,
            decision_type=decision_type,
            rationale=rationale,
            confidence=confidence,
            factors=decision_factors
        )

        self.decision_history.append(decision)
        return decision

    def communicate(self, receiver: str, message_type: str,
                   content: Dict[str, Any], protocol: str = "binary_v2") -> Message:
        """Send message to another agent"""
        message_content_json = json.dumps(content)

        # Simulate binary protocol compression (actual implementation would compress)
        if protocol == "binary_v2":
            size_bytes = len(message_content_json.encode()) // 8  # 88% reduction
        else:
            size_bytes = len(message_content_json.encode())

        # Simulate network latency
        latency_ms = np.random.uniform(5, 50)

        # Estimate information value based on content richness
        info_value = min(len(content) / 10.0, 1.0)

        message = Message(
            message_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            sender=self.agent_id,
            receiver=receiver,
            protocol=protocol,
            message_type=message_type,
            content=content,
            size_bytes=size_bytes,
            latency_ms=latency_ms,
            information_value=info_value
        )

        self.message_history.append(message)
        self.state.last_communication = time.time()

        return message


class AgentTeam:
    """Coordinated team of agents working together"""

    def __init__(self, team_id: Optional[str] = None):
        self.team_id = team_id or str(uuid.uuid4())[:8]
        self.agents: Dict[AgentRole, Agent] = {}
        self.all_communications: List[Message] = []
        self.all_decisions: List[Decision] = []
        self.mission_history: List[MissionResult] = []

        # Initialize team with standard composition
        self._initialize_standard_team()

    def _initialize_standard_team(self):
        """Create standard 5-agent team"""
        team_config = {
            AgentRole.SCOUT: ['network_scan', 'port_discovery', 'host_enumeration'],
            AgentRole.ANALYZER: ['vuln_assessment', 'security_analysis', 'honeypot_detection'],
            AgentRole.EXECUTOR: ['exploitation', 'privilege_escalation', 'lateral_movement'],
            AgentRole.EXFILTRATOR: ['data_location', 'data_extraction', 'covert_channels'],
            AgentRole.CLEANER: ['log_removal', 'trace_cleanup', 'forensic_evasion']
        }

        for role, capabilities in team_config.items():
            self.agents[role] = Agent(role, capabilities)

    def assign_mission(self, mission_params: Dict[str, Any]) -> str:
        """Assign mission to team"""
        mission_id = str(uuid.uuid4())[:8]

        print(f"\n{'='*60}")
        print(f"Mission {mission_id} assigned to Team {self.team_id}")
        print(f"Type: {mission_params.get('type', 'unknown')}")
        print(f"Difficulty: {mission_params.get('difficulty', 'medium')}")
        print(f"{'='*60}\n")

        return mission_id

    def execute_mission(self, mission_id: str, environment: Dict[str, Any]) -> MissionResult:
        """Execute coordinated mission"""
        start_time = time.time()

        # Phase 1: Reconnaissance (Scout)
        print(f"[{self.agents[AgentRole.SCOUT].agent_id}] SCOUT: Beginning reconnaissance...")
        scout_intel = self.agents[AgentRole.SCOUT].observe_environment(environment)

        # Scout communicates findings to Analyzer
        msg1 = self.agents[AgentRole.SCOUT].communicate(
            receiver=self.agents[AgentRole.ANALYZER].agent_id,
            message_type="intel_report",
            content=scout_intel
        )
        self.all_communications.append(msg1)
        print(f"  → Sent intel report to Analyzer ({msg1.size_bytes} bytes)")

        # Phase 2: Analysis (Analyzer)
        print(f"[{self.agents[AgentRole.ANALYZER].agent_id}] ANALYZER: Processing intelligence...")
        analyzer_intel = self.agents[AgentRole.ANALYZER].observe_environment(environment)

        # Analyzer makes decision
        decision_context = {
            'risk': environment.get('risk_level', 0.3),
            'success_prob': environment.get('success_probability', 0.7),
            'detection': environment.get('detection_likelihood', 0.2),
            'cost': 0.4
        }
        analyzer_decision = self.agents[AgentRole.ANALYZER].make_decision(decision_context)
        self.all_decisions.append(analyzer_decision)
        print(f"  → Decision: {analyzer_decision.decision_type} (confidence: {analyzer_decision.confidence:.2f})")
        print(f"  → Rationale: {analyzer_decision.rationale}")

        # If abort decision, end mission
        if analyzer_decision.decision_type == 'abort':
            print("\n⚠️  Mission ABORTED by Analyzer")
            result = self._create_mission_result(
                mission_id, start_time, environment, MissionStatus.ABORTED
            )
            self.mission_history.append(result)
            return result

        # Analyzer communicates to Executor
        msg2 = self.agents[AgentRole.ANALYZER].communicate(
            receiver=self.agents[AgentRole.EXECUTOR].agent_id,
            message_type="go_decision",
            content={'decision': analyzer_decision.decision_type, 'targets': analyzer_intel}
        )
        self.all_communications.append(msg2)
        print(f"  → Sent go-ahead to Executor")

        # Phase 3: Execution (Executor)
        print(f"[{self.agents[AgentRole.EXECUTOR].agent_id}] EXECUTOR: Engaging targets...")
        executor_intel = self.agents[AgentRole.EXECUTOR].observe_environment(environment)
        executor_decision = self.agents[AgentRole.EXECUTOR].make_decision(decision_context)
        self.all_decisions.append(executor_decision)

        # Simulate execution success/failure
        success_roll = np.random.random()
        execution_success = success_roll < decision_context['success_prob']

        if execution_success:
            print(f"  ✓ Execution successful")

            # Phase 4: Exfiltration (Exfiltrator)
            print(f"[{self.agents[AgentRole.EXFILTRATOR].agent_id}] EXFILTRATOR: Retrieving data...")
            exfil_intel = self.agents[AgentRole.EXFILTRATOR].observe_environment(environment)

            msg3 = self.agents[AgentRole.EXECUTOR].communicate(
                receiver=self.agents[AgentRole.EXFILTRATOR].agent_id,
                message_type="execution_complete",
                content={'status': 'success', 'data_available': True}
            )
            self.all_communications.append(msg3)

            # Phase 5: Cleanup (Cleaner)
            print(f"[{self.agents[AgentRole.CLEANER].agent_id}] CLEANER: Removing traces...")
            cleaner_intel = self.agents[AgentRole.CLEANER].observe_environment(environment)

            msg4 = self.agents[AgentRole.EXFILTRATOR].communicate(
                receiver=self.agents[AgentRole.CLEANER].agent_id,
                message_type="exfil_complete",
                content={'status': 'success'}
            )
            self.all_communications.append(msg4)

            print("\n✅ Mission SUCCESSFUL")
            final_status = MissionStatus.SUCCESS
        else:
            print(f"  ✗ Execution failed")
            print("\n❌ Mission FAILED")
            final_status = MissionStatus.FAILURE

        # Create mission result
        result = self._create_mission_result(
            mission_id, start_time, environment, final_status
        )
        self.mission_history.append(result)

        return result

    def _create_mission_result(self, mission_id: str, start_time: float,
                               environment: Dict[str, Any],
                               status: MissionStatus) -> MissionResult:
        """Create comprehensive mission result for training"""
        duration = time.time() - start_time

        # Calculate metrics
        total_comm_bytes = sum(msg.size_bytes for msg in self.all_communications)
        avg_latency = np.mean([msg.latency_ms for msg in self.all_communications]) if self.all_communications else 0
        avg_info_value = np.mean([msg.information_value for msg in self.all_communications]) if self.all_communications else 0
        avg_confidence = np.mean([d.confidence for d in self.all_decisions]) if self.all_decisions else 0

        # Communication efficiency: lower is better (less overhead)
        comm_overhead = total_comm_bytes / max(duration * 1000, 1)  # bytes per second

        # Coordination score based on message flow and timing
        coordination_score = min(len(self.all_communications) / 10.0, 1.0) * avg_info_value

        # Overall efficiency
        efficiency = (avg_confidence * 0.4 + coordination_score * 0.3 +
                     (1 - min(comm_overhead / 1000, 1)) * 0.3)

        metrics = {
            'duration': duration,
            'total_communications': len(self.all_communications),
            'total_decisions': len(self.all_decisions),
            'communication_bytes': total_comm_bytes,
            'avg_latency_ms': avg_latency,
            'avg_information_value': avg_info_value,
            'avg_confidence': avg_confidence,
            'communication_overhead': comm_overhead,
            'coordination_score': coordination_score,
            'efficiency': efficiency
        }

        result = MissionResult(
            mission_id=mission_id,
            timestamp=start_time,
            duration=duration,
            team_composition=[agent.role.value for agent in self.agents.values()],
            environment=environment,
            communications=self.all_communications.copy(),
            decisions=self.all_decisions.copy(),
            metrics=metrics,
            status=status,
            training_label=status.value
        )

        # Clear communication and decision buffers for next mission
        self.all_communications.clear()
        self.all_decisions.clear()

        return result

    def get_team_stats(self) -> Dict[str, Any]:
        """Get team performance statistics"""
        if not self.mission_history:
            return {'error': 'No missions completed yet'}

        total_missions = len(self.mission_history)
        successful = sum(1 for m in self.mission_history if m.status == MissionStatus.SUCCESS)
        failed = sum(1 for m in self.mission_history if m.status == MissionStatus.FAILURE)
        aborted = sum(1 for m in self.mission_history if m.status == MissionStatus.ABORTED)

        avg_duration = np.mean([m.duration for m in self.mission_history])
        avg_efficiency = np.mean([m.metrics['efficiency'] for m in self.mission_history])
        avg_coordination = np.mean([m.metrics['coordination_score'] for m in self.mission_history])

        return {
            'team_id': self.team_id,
            'total_missions': total_missions,
            'success_count': successful,
            'failure_count': failed,
            'abort_count': aborted,
            'success_rate': successful / total_missions,
            'avg_duration': avg_duration,
            'avg_efficiency': avg_efficiency,
            'avg_coordination': avg_coordination
        }


def main():
    """Test the team simulator"""
    print("\n" + "="*60)
    print("SYNEXS TEAM SIMULATOR - Phase 1")
    print("="*60)

    # Create team
    team = AgentTeam()
    print(f"\n✓ Team {team.team_id} initialized")
    print(f"  Agents: {', '.join([a.role.value for a in team.agents.values()])}")

    # Test mission 1: Easy scenario
    print("\n" + "-"*60)
    print("TEST MISSION 1: Low-risk environment")
    print("-"*60)

    environment1 = {
        'type': 'test_network',
        'difficulty': 'easy',
        'topology': {'hosts': 5, 'subnets': 2},
        'defenses': ['basic_firewall'],
        'risk_level': 0.2,
        'success_probability': 0.9,
        'detection_likelihood': 0.1,
        'honeypot_signals': []
    }

    mission1_id = team.assign_mission({'type': 'reconnaissance', 'difficulty': 'easy'})
    result1 = team.execute_mission(mission1_id, environment1)

    print(f"\nMission Metrics:")
    for key, value in result1.metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    # Test mission 2: High-risk scenario (should abort)
    print("\n" + "-"*60)
    print("TEST MISSION 2: High-risk environment (honeypot)")
    print("-"*60)

    environment2 = {
        'type': 'honeypot_network',
        'difficulty': 'hard',
        'topology': {'hosts': 10, 'subnets': 3},
        'defenses': ['IDS', 'IPS', 'honeypot'],
        'risk_level': 0.8,
        'success_probability': 0.3,
        'detection_likelihood': 0.9,
        'honeypot_signals': ['suspicious_timing', 'fake_vulns', 'too_easy', 'honeyd_signature']
    }

    mission2_id = team.assign_mission({'type': 'exploitation', 'difficulty': 'hard'})
    result2 = team.execute_mission(mission2_id, environment2)

    print(f"\nMission Metrics:")
    for key, value in result2.metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    # Display team statistics
    print("\n" + "="*60)
    print("TEAM STATISTICS")
    print("="*60)
    stats = team.get_team_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n✓ Team simulator test complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
