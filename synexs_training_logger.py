#!/usr/bin/env python3
"""
Synexs Training Logger - GPU Training Pipeline
Formats team simulation data for PyTorch GPU training

This module:
- Captures all mission execution data
- Formats for GPU-optimized training
- Generates training batches in real-time
- Integrates with PyTorch DataLoader
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import asdict
from datetime import datetime
import joblib

from synexs_team_simulator import MissionResult, Message, Decision, MissionStatus


class TrainingDataLogger:
    """Logs mission data in format optimized for GPU training"""

    def __init__(self, output_dir: str = "./training_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Separate directories for different data types
        self.missions_dir = self.output_dir / "missions"
        self.batches_dir = self.output_dir / "batches"
        self.models_dir = self.output_dir / "models"

        self.missions_dir.mkdir(exist_ok=True)
        self.batches_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # Training data buffer
        self.mission_buffer: List[MissionResult] = []
        self.batch_size = 32
        self.sequence_length = 50  # Max messages per mission

        # Statistics
        self.total_logged = 0
        self.total_batches = 0

        print(f"✓ Training logger initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Batch size: {self.batch_size}")

    def log_mission(self, mission_result: MissionResult) -> str:
        """Log individual mission result"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mission_{mission_result.mission_id}_{timestamp}.jsonl"
        filepath = self.missions_dir / filename

        # Convert to JSON-serializable format
        mission_data = {
            'mission_id': mission_result.mission_id,
            'timestamp': mission_result.timestamp,
            'duration': mission_result.duration,
            'team_composition': mission_result.team_composition,
            'environment': mission_result.environment,
            'communications': [asdict(msg) for msg in mission_result.communications],
            'decisions': [asdict(dec) for dec in mission_result.decisions],
            'metrics': mission_result.metrics,
            'status': mission_result.status.value,
            'training_label': mission_result.training_label
        }

        # Write JSONL format (one mission per line)
        with open(filepath, 'w') as f:
            f.write(json.dumps(mission_data) + '\n')

        self.total_logged += 1

        # Add to buffer for batch processing
        self.mission_buffer.append(mission_result)

        print(f"✓ Mission {mission_result.mission_id} logged to {filename}")

        # Check if we have enough for a batch
        if len(self.mission_buffer) >= self.batch_size:
            self._create_training_batch()

        return str(filepath)

    def _create_training_batch(self) -> str:
        """Create GPU-optimized training batch"""
        if len(self.mission_buffer) < self.batch_size:
            return ""

        print(f"\n{'='*50}")
        print(f"Creating training batch {self.total_batches + 1}")
        print(f"{'='*50}")

        # Take batch_size missions from buffer
        batch_missions = self.mission_buffer[:self.batch_size]
        self.mission_buffer = self.mission_buffer[self.batch_size:]

        # Convert to training tensors
        features, labels, metadata = self._missions_to_tensors(batch_missions)

        # Save as PyTorch tensor file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"batch_{self.total_batches:06d}_{timestamp}.pt"
        batch_filepath = self.batches_dir / batch_filename

        torch.save({
            'features': features,
            'labels': labels,
            'metadata': metadata,
            'batch_id': self.total_batches,
            'timestamp': timestamp,
            'batch_size': self.batch_size
        }, batch_filepath)

        self.total_batches += 1

        print(f"✓ Batch saved: {batch_filename}")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Device ready: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*50}\n")

        return str(batch_filepath)

    def _missions_to_tensors(self, missions: List[MissionResult]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Convert missions to GPU-optimized tensors

        Returns:
            features: [batch_size, sequence_length, feature_dim]
            labels: [batch_size, num_classes]
            metadata: List of mission metadata dicts
        """
        batch_features = []
        batch_labels = []
        batch_metadata = []

        for mission in missions:
            # Extract features from mission
            features = self._extract_features(mission)
            batch_features.append(features)

            # Create label (mission success/failure/abort)
            label = self._create_label(mission)
            batch_labels.append(label)

            # Store metadata for analysis
            metadata = {
                'mission_id': mission.mission_id,
                'duration': mission.duration,
                'status': mission.status.value,
                'efficiency': mission.metrics['efficiency']
            }
            batch_metadata.append(metadata)

        # Convert to tensors
        features_tensor = torch.stack(batch_features)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

        return features_tensor, labels_tensor, batch_metadata

    def _extract_features(self, mission: MissionResult) -> torch.Tensor:
        """
        Extract feature vector from mission

        Feature dimensions:
        - Communication sequence embeddings
        - Decision features
        - Environment features
        - Timing features
        - Performance metrics
        """
        # Communication features (sequence of messages)
        comm_features = self._encode_communications(mission.communications)

        # Decision features
        decision_features = self._encode_decisions(mission.decisions)

        # Environment features
        env_features = self._encode_environment(mission.environment)

        # Temporal features
        temporal_features = torch.tensor([
            mission.duration,
            len(mission.communications),
            len(mission.decisions),
            mission.metrics.get('avg_latency_ms', 0),
            mission.metrics.get('communication_overhead', 0)
        ], dtype=torch.float32)

        # Performance metrics
        metric_features = torch.tensor([
            mission.metrics.get('efficiency', 0),
            mission.metrics.get('coordination_score', 0),
            mission.metrics.get('avg_confidence', 0),
            mission.metrics.get('avg_information_value', 0)
        ], dtype=torch.float32)

        # Combine all features
        # [sequence_length, feature_dim]
        combined_features = torch.cat([
            comm_features,
            decision_features,
            env_features.unsqueeze(0).expand(self.sequence_length, -1),
            temporal_features.unsqueeze(0).expand(self.sequence_length, -1),
            metric_features.unsqueeze(0).expand(self.sequence_length, -1)
        ], dim=1)

        return combined_features

    def _encode_communications(self, messages: List[Message]) -> torch.Tensor:
        """
        Encode message sequence as tensor

        Features per message:
        - Message size
        - Latency
        - Information value
        - Protocol efficiency
        """
        # Pad or truncate to sequence_length
        encoded = torch.zeros((self.sequence_length, 4), dtype=torch.float32)

        for i, msg in enumerate(messages[:self.sequence_length]):
            encoded[i] = torch.tensor([
                msg.size_bytes / 1000.0,  # Normalize
                msg.latency_ms / 100.0,
                msg.information_value,
                1.0 if msg.protocol == 'binary_v2' else 0.0
            ])

        return encoded

    def _encode_decisions(self, decisions: List[Decision]) -> torch.Tensor:
        """
        Encode decision sequence as tensor

        Features per decision:
        - Decision type (one-hot)
        - Confidence
        - Risk factors
        """
        # Pad or truncate to sequence_length
        encoded = torch.zeros((self.sequence_length, 6), dtype=torch.float32)

        decision_type_map = {
            'proceed': 0,
            'abort': 1,
            'wait': 2,
            'adapt': 3
        }

        for i, decision in enumerate(decisions[:self.sequence_length]):
            decision_type_idx = decision_type_map.get(decision.decision_type, 0)

            # One-hot encode decision type
            one_hot = torch.zeros(4)
            one_hot[decision_type_idx] = 1.0

            # Combine with confidence and risk
            encoded[i] = torch.cat([
                one_hot,
                torch.tensor([decision.confidence]),
                torch.tensor([decision.factors.get('risk_level', 0.5)])
            ])

        return encoded

    def _encode_environment(self, environment: Dict[str, Any]) -> torch.Tensor:
        """
        Encode environment features as tensor

        Features:
        - Risk level
        - Success probability
        - Detection likelihood
        - Honeypot indicators count
        - Defense complexity
        """
        features = torch.tensor([
            environment.get('risk_level', 0.5),
            environment.get('success_probability', 0.5),
            environment.get('detection_likelihood', 0.5),
            len(environment.get('honeypot_signals', [])) / 10.0,  # Normalize
            len(environment.get('defenses', [])) / 5.0
        ], dtype=torch.float32)

        return features

    def _create_label(self, mission: MissionResult) -> int:
        """
        Create classification label

        Labels:
        0 - SUCCESS
        1 - FAILURE
        2 - ABORTED
        """
        label_map = {
            MissionStatus.SUCCESS: 0,
            MissionStatus.FAILURE: 1,
            MissionStatus.ABORTED: 2
        }
        return label_map.get(mission.status, 1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'total_missions_logged': self.total_logged,
            'total_batches_created': self.total_batches,
            'missions_in_buffer': len(self.mission_buffer),
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'output_directory': str(self.output_dir),
            'cuda_available': torch.cuda.is_available()
        }

    def export_for_training(self) -> str:
        """Export all batches for GPU training"""
        # Flush any remaining missions in buffer
        if len(self.mission_buffer) > 0:
            print(f"\nFlushing {len(self.mission_buffer)} remaining missions...")
            # Pad to batch_size if needed
            while len(self.mission_buffer) < self.batch_size and len(self.mission_buffer) > 0:
                # Duplicate last mission to fill batch
                self.mission_buffer.append(self.mission_buffer[-1])
            self._create_training_batch()

        # Create index file for training
        index_file = self.batches_dir / "training_index.json"
        batch_files = sorted(self.batches_dir.glob("batch_*.pt"))

        index_data = {
            'total_batches': len(batch_files),
            'batch_files': [str(f.name) for f in batch_files],
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'created': datetime.now().isoformat(),
            'cuda_available': torch.cuda.is_available()
        }

        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)

        print(f"\n✓ Training data exported")
        print(f"  Total batches: {len(batch_files)}")
        print(f"  Index file: {index_file}")

        return str(index_file)


class GPUTrainingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for GPU training"""

    def __init__(self, batch_dir: str):
        self.batch_dir = Path(batch_dir)
        self.batch_files = sorted(self.batch_dir.glob("batch_*.pt"))

        if not self.batch_files:
            raise ValueError(f"No batch files found in {batch_dir}")

        print(f"✓ Dataset initialized with {len(self.batch_files)} batches")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        batch_data = torch.load(self.batch_files[idx])
        return batch_data['features'], batch_data['labels']


def main():
    """Test training logger"""
    from synexs_team_simulator import AgentTeam

    print("\n" + "="*60)
    print("SYNEXS TRAINING LOGGER - Test")
    print("="*60)

    # Initialize logger
    logger = TrainingDataLogger(output_dir="./training_logs_test")

    # Create team
    team = AgentTeam()

    # Run multiple missions to test batch creation
    print("\nRunning 5 test missions...")

    environments = [
        {
            'type': 'test_network',
            'risk_level': 0.2,
            'success_probability': 0.9,
            'detection_likelihood': 0.1,
            'honeypot_signals': [],
            'defenses': ['firewall']
        },
        {
            'type': 'medium_network',
            'risk_level': 0.5,
            'success_probability': 0.6,
            'detection_likelihood': 0.4,
            'honeypot_signals': ['timing'],
            'defenses': ['firewall', 'IDS']
        },
        {
            'type': 'honeypot',
            'risk_level': 0.9,
            'success_probability': 0.2,
            'detection_likelihood': 0.9,
            'honeypot_signals': ['timing', 'fake_vulns', 'too_easy'],
            'defenses': ['firewall', 'IDS', 'IPS', 'honeypot']
        },
        {
            'type': 'test_network',
            'risk_level': 0.3,
            'success_probability': 0.8,
            'detection_likelihood': 0.2,
            'honeypot_signals': [],
            'defenses': ['basic_firewall']
        },
        {
            'type': 'secure_network',
            'risk_level': 0.6,
            'success_probability': 0.5,
            'detection_likelihood': 0.5,
            'honeypot_signals': ['timing'],
            'defenses': ['firewall', 'IDS', 'WAF']
        }
    ]

    for i, env in enumerate(environments):
        print(f"\n--- Mission {i+1}/5 ---")
        mission_id = team.assign_mission({'type': 'test', 'number': i+1})
        result = team.execute_mission(mission_id, env)
        logger.log_mission(result)

    # Show statistics
    print("\n" + "="*60)
    print("LOGGER STATISTICS")
    print("="*60)
    stats = logger.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export for training
    print("\n" + "="*60)
    index_file = logger.export_for_training()

    print("\n✓ Training logger test complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
