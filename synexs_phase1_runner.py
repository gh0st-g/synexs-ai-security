#!/usr/bin/env python3
"""
Synexs Phase 1 - Complete Integration
Runs team simulations, logs data, and prepares for GPU training

This is the main entry point for Phase 1 development
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

from synexs_team_simulator import AgentTeam, MissionStatus
from synexs_training_logger import TrainingDataLogger


class MissionGenerator:
    """Generates diverse mission scenarios for training"""

    def __init__(self):
        self.mission_count = 0

        # Pre-defined environment templates
        self.environment_templates = {
            'easy': {
                'risk_level': (0.1, 0.3),
                'success_probability': (0.8, 0.95),
                'detection_likelihood': (0.05, 0.2),
                'honeypot_signals': 0,
                'defenses': ['basic_firewall']
            },
            'medium': {
                'risk_level': (0.3, 0.6),
                'success_probability': (0.5, 0.75),
                'detection_likelihood': (0.3, 0.5),
                'honeypot_signals': (0, 2),
                'defenses': ['firewall', 'IDS']
            },
            'hard': {
                'risk_level': (0.6, 0.85),
                'success_probability': (0.2, 0.5),
                'detection_likelihood': (0.5, 0.8),
                'honeypot_signals': (1, 3),
                'defenses': ['firewall', 'IDS', 'IPS', 'WAF']
            },
            'honeypot': {
                'risk_level': (0.85, 0.95),
                'success_probability': (0.05, 0.25),
                'detection_likelihood': (0.8, 0.95),
                'honeypot_signals': (3, 5),
                'defenses': ['firewall', 'IDS', 'IPS', 'honeypot', 'deception_tech']
            }
        }

        self.signal_types = [
            'suspicious_timing',
            'fake_vulnerabilities',
            'too_easy_access',
            'honeyd_signature',
            'unrealistic_services',
            'bait_files',
            'fake_credentials'
        ]

    def generate(self, difficulty: str = None) -> Dict[str, Any]:
        """Generate random environment based on difficulty"""
        if difficulty is None:
            # Random difficulty weighted toward medium
            difficulty = np.random.choice(
                ['easy', 'medium', 'hard', 'honeypot'],
                p=[0.2, 0.5, 0.2, 0.1]
            )

        template = self.environment_templates[difficulty]

        # Generate values from ranges
        if isinstance(template['risk_level'], tuple):
            risk_level = np.random.uniform(*template['risk_level'])
        else:
            risk_level = template['risk_level']

        if isinstance(template['success_probability'], tuple):
            success_prob = np.random.uniform(*template['success_probability'])
        else:
            success_prob = template['success_probability']

        if isinstance(template['detection_likelihood'], tuple):
            detection = np.random.uniform(*template['detection_likelihood'])
        else:
            detection = template['detection_likelihood']

        # Generate honeypot signals
        if isinstance(template['honeypot_signals'], tuple):
            num_signals = np.random.randint(*template['honeypot_signals'])
        else:
            num_signals = template['honeypot_signals']

        honeypot_signals = np.random.choice(
            self.signal_types,
            size=min(num_signals, len(self.signal_types)),
            replace=False
        ).tolist()

        # Create environment
        environment = {
            'mission_number': self.mission_count,
            'type': f'{difficulty}_network',
            'difficulty': difficulty,
            'topology': {
                'hosts': np.random.randint(3, 20),
                'subnets': np.random.randint(1, 5)
            },
            'risk_level': risk_level,
            'success_probability': success_prob,
            'detection_likelihood': detection,
            'honeypot_signals': honeypot_signals,
            'defenses': template['defenses'].copy()
        }

        self.mission_count += 1
        return environment


def run_training_session(num_missions: int = 100,
                         output_dir: str = "./training_logs",
                         team_id: str = None):
    """
    Run complete training session

    Args:
        num_missions: Number of missions to run
        output_dir: Where to save training data
        team_id: Optional team identifier
    """
    print("\n" + "="*70)
    print("SYNEXS PHASE 1 - TEAM TRAINING SESSION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Missions to run: {num_missions}")
    print(f"  Output directory: {output_dir}")
    print(f"  Team ID: {team_id or 'auto-generated'}")
    print("="*70 + "\n")

    # Initialize components
    team = AgentTeam(team_id=team_id)
    logger = TrainingDataLogger(output_dir=output_dir)
    generator = MissionGenerator()

    # Statistics tracking
    start_time = time.time()
    success_count = 0
    failure_count = 0
    abort_count = 0

    # Run missions
    for i in range(num_missions):
        print(f"\n{'─'*70}")
        print(f"Mission {i+1}/{num_missions}")
        print(f"{'─'*70}")

        # Generate environment
        environment = generator.generate()
        print(f"Difficulty: {environment['difficulty']}")
        print(f"Risk: {environment['risk_level']:.2f} | "
              f"Success Prob: {environment['success_probability']:.2f} | "
              f"Detection: {environment['detection_likelihood']:.2f}")

        # Run mission
        mission_id = team.assign_mission({
            'type': environment['type'],
            'number': i + 1
        })
        result = team.execute_mission(mission_id, environment)

        # Log for training
        logger.log_mission(result)

        # Update statistics
        if result.status == MissionStatus.SUCCESS:
            success_count += 1
        elif result.status == MissionStatus.FAILURE:
            failure_count += 1
        else:
            abort_count += 1

        # Progress update every 10 missions
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (num_missions - i - 1) / rate

            print(f"\n{'='*70}")
            print(f"Progress: {i+1}/{num_missions} missions ({(i+1)/num_missions*100:.1f}%)")
            print(f"  Success: {success_count} | Failure: {failure_count} | Abort: {abort_count}")
            print(f"  Rate: {rate:.2f} missions/sec")
            print(f"  Est. time remaining: {remaining:.1f}s")
            print(f"{'='*70}")

    # Final statistics
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("TRAINING SESSION COMPLETE")
    print("="*70)
    print(f"\nSession Statistics:")
    print(f"  Total missions: {num_missions}")
    print(f"  Successful: {success_count} ({success_count/num_missions*100:.1f}%)")
    print(f"  Failed: {failure_count} ({failure_count/num_missions*100:.1f}%)")
    print(f"  Aborted: {abort_count} ({abort_count/num_missions*100:.1f}%)")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time per mission: {total_time/num_missions:.2f}s")

    # Team statistics
    print(f"\nTeam Performance:")
    team_stats = team.get_team_stats()
    for key, value in team_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Logger statistics
    print(f"\nTraining Data:")
    logger_stats = logger.get_statistics()
    for key, value in logger_stats.items():
        print(f"  {key}: {value}")

    # Export training data
    print(f"\n{'─'*70}")
    print("Exporting training data...")
    index_file = logger.export_for_training()
    print(f"✓ Training data ready for GPU training")
    print(f"  Index file: {index_file}")
    print(f"{'─'*70}")

    print("\n✓ Phase 1 training session complete!")
    print("="*70 + "\n")

    print("Next steps:")
    print("  1. Review training data in:", output_dir)
    print("  2. Start GPU training with:")
    print(f"     python synexs_gpu_trainer.py {Path(output_dir) / 'batches'}")
    print()

    return {
        'team': team,
        'logger': logger,
        'generator': generator,
        'stats': {
            'total_missions': num_missions,
            'success_count': success_count,
            'failure_count': failure_count,
            'abort_count': abort_count,
            'total_time': total_time
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Synexs Phase 1 - Team Training Runner"
    )
    parser.add_argument(
        '--missions',
        type=int,
        default=100,
        help='Number of missions to run (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./training_logs',
        help='Output directory for training data (default: ./training_logs)'
    )
    parser.add_argument(
        '--team-id',
        type=str,
        default=None,
        help='Team identifier (default: auto-generated)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with 10 missions'
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick:
        print("\n⚡ Quick test mode: Running 10 missions")
        args.missions = 10

    # Run training session
    results = run_training_session(
        num_missions=args.missions,
        output_dir=args.output,
        team_id=args.team_id
    )


if __name__ == "__main__":
    main()
