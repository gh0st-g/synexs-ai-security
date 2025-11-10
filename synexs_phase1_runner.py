#!/usr/bin/env python3
"""
Synexs Phase 1 - Complete Integration
Runs team simulations, logs data, and prepares for GPU training

This is the main entry point for Phase 1 development

OPTIMIZATIONS:
- Robust error handling with file logging
- Adaptive checkpoint intervals for large runs
- Progress persistence to JSON for external monitoring
- Memory-efficient buffer management
- Graceful shutdown with signal handlers
- Recovery from all error conditions
"""

import argparse
import time
import signal
import sys
import json
import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from synexs_team_simulator import AgentTeam, MissionStatus
from synexs_training_logger import TrainingDataLogger

# Global flag for graceful shutdown
shutdown_requested = False
logger_instance = None


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup comprehensive logging to both file and console"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create logger
    logger = logging.getLogger('synexs_training')
    logger.setLevel(logging.DEBUG)

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized: {log_file}")
    return logger


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
    print(f"\n\n⚠️  {signal_name} received. Finishing current mission and saving progress...")
    if logger_instance:
        logger_instance.warning(f"{signal_name} received - graceful shutdown initiated")
    shutdown_requested = True


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


def save_checkpoint(output_dir: str, mission_num: int, stats: Dict[str, Any],
                    team_stats: Dict[str, Any], logger_stats: Dict[str, Any]):
    """Save progress checkpoint with atomic write"""
    checkpoint_file = Path(output_dir) / "checkpoint.json"
    checkpoint_temp = Path(output_dir) / "checkpoint.json.tmp"

    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'missions_completed': mission_num,
        'stats': stats,
        'team_stats': team_stats,
        'logger_stats': logger_stats
    }

    try:
        # Write to temp file first (atomic operation)
        with open(checkpoint_temp, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Move temp to final location (atomic on POSIX)
        checkpoint_temp.replace(checkpoint_file)
    except Exception as e:
        if logger_instance:
            logger_instance.error(f"Failed to save checkpoint: {e}")
        raise


def save_progress_file(output_dir: str, mission_num: int, total_missions: int,
                       stats: Dict[str, Any], start_time: float):
    """Save progress data for external monitoring (progress.sh)"""
    progress_file = Path(output_dir) / "progress.json"
    progress_temp = Path(output_dir) / "progress.json.tmp"

    elapsed = time.time() - start_time
    rate = (mission_num / elapsed) if elapsed > 0 else 0
    remaining = ((total_missions - mission_num) / rate) if rate > 0 else 0

    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'mission_current': mission_num,
        'mission_total': total_missions,
        'progress_percent': (mission_num / total_missions * 100) if total_missions > 0 else 0,
        'elapsed_seconds': elapsed,
        'rate_missions_per_sec': rate,
        'eta_seconds': remaining,
        'stats': stats,
        'status': 'running'
    }

    try:
        with open(progress_temp, 'w') as f:
            json.dump(progress_data, f, indent=2)
        progress_temp.replace(progress_file)
    except Exception as e:
        if logger_instance:
            logger_instance.error(f"Failed to save progress file: {e}")


def calculate_checkpoint_interval(num_missions: int) -> int:
    """Calculate optimal checkpoint interval based on total missions"""
    if num_missions <= 100:
        return 10  # Every 10 missions for small runs
    elif num_missions <= 1000:
        return 50  # Every 50 missions for medium runs
    elif num_missions <= 10000:
        return 100  # Every 100 missions for large runs
    else:
        return 500  # Every 500 missions for very large runs (100K)


def load_checkpoint(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load progress checkpoint if exists"""
    checkpoint_file = Path(output_dir) / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                if logger_instance:
                    logger_instance.info(f"Checkpoint loaded: {data.get('missions_completed', 0)} missions")
                return data
        except json.JSONDecodeError as e:
            msg = f"Checkpoint file corrupted: {e}. Starting fresh."
            print(f"⚠️  Warning: {msg}")
            if logger_instance:
                logger_instance.error(msg)
        except Exception as e:
            msg = f"Could not load checkpoint: {e}"
            print(f"⚠️  Warning: {msg}")
            if logger_instance:
                logger_instance.error(msg)
    return None


def run_training_session(num_missions: int = 100,
                         output_dir: str = "./training_logs",
                         team_id: str = None,
                         resume: bool = True):
    """
    Run complete training session with checkpoint/resume support

    Args:
        num_missions: Number of missions to run
        output_dir: Where to save training data
        team_id: Optional team identifier
        resume: Whether to resume from checkpoint if exists
    """
    global shutdown_requested, logger_instance

    # Setup logging first
    logger = setup_logging(output_dir)
    logger_instance = logger

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("="*70)
    logger.info("SYNEXS PHASE 1 - TEAM TRAINING SESSION")
    logger.info("="*70)

    print("\n" + "="*70)
    print("SYNEXS PHASE 1 - TEAM TRAINING SESSION")
    print("="*70)

    # Calculate adaptive checkpoint interval
    checkpoint_interval = calculate_checkpoint_interval(num_missions)
    logger.info(f"Checkpoint interval: every {checkpoint_interval} missions")

    # Check for checkpoint
    start_mission = 0
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            start_mission = checkpoint['missions_completed']
            print(f"\n📂 Checkpoint found! Resuming from mission {start_mission + 1}")
            logger.info(f"Resuming from checkpoint at mission {start_mission + 1}")

    print(f"\nConfiguration:")
    print(f"  Missions to run: {num_missions}")
    print(f"  Starting from: Mission {start_mission + 1}")
    print(f"  Output directory: {output_dir}")
    print(f"  Team ID: {team_id or 'auto-generated'}")
    print(f"  Checkpoint interval: {checkpoint_interval} missions")
    print("="*70 + "\n")

    logger.info(f"Configuration: missions={num_missions}, start={start_mission+1}, output={output_dir}")

    # Initialize components
    try:
        logger.info("Initializing team, data logger, and mission generator...")
        team = AgentTeam(team_id=team_id)
        data_logger = TrainingDataLogger(output_dir=output_dir)
        generator = MissionGenerator()
        logger.info("All components initialized successfully")
    except Exception as e:
        error_msg = f"Error initializing components: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Statistics tracking
    start_time = time.time()
    if checkpoint and 'stats' in checkpoint:
        success_count = checkpoint['stats'].get('success_count', 0)
        failure_count = checkpoint['stats'].get('failure_count', 0)
        abort_count = checkpoint['stats'].get('abort_count', 0)
    else:
        success_count = 0
        failure_count = 0
        abort_count = 0

    # Advance generator to correct position
    generator.mission_count = start_mission
    logger.info(f"Starting mission loop from mission {start_mission + 1}")

    # Run missions
    missions_completed = start_mission
    for i in range(start_mission, num_missions):
        # Check for shutdown request
        if shutdown_requested:
            logger.warning("Shutdown requested - breaking mission loop")
            print("\n\n🛑 Graceful shutdown initiated...")
            break

        print(f"\n{'─'*70}")
        print(f"Mission {i+1}/{num_missions}")
        print(f"{'─'*70}")

        try:
            # Generate environment
            logger.debug(f"Mission {i+1}: Generating environment")
            environment = generator.generate()
            print(f"Difficulty: {environment['difficulty']}")
            print(f"Risk: {environment['risk_level']:.2f} | "
                  f"Success Prob: {environment['success_probability']:.2f} | "
                  f"Detection: {environment['detection_likelihood']:.2f}")

            # Run mission
            logger.debug(f"Mission {i+1}: Assigning to team")
            mission_id = team.assign_mission({
                'type': environment['type'],
                'number': i + 1
            })

            logger.debug(f"Mission {i+1}: Executing mission {mission_id}")
            result = team.execute_mission(mission_id, environment)

            # Log for training
            logger.debug(f"Mission {i+1}: Logging results")
            data_logger.log_mission(result)

            # Update statistics
            if result.status == MissionStatus.SUCCESS:
                success_count += 1
                logger.debug(f"Mission {i+1}: SUCCESS")
            elif result.status == MissionStatus.FAILURE:
                failure_count += 1
                logger.debug(f"Mission {i+1}: FAILURE")
            else:
                abort_count += 1
                logger.debug(f"Mission {i+1}: ABORTED")

            missions_completed = i + 1

        except KeyboardInterrupt:
            # Let signal handler deal with this
            raise
        except Exception as e:
            error_msg = f"Error during mission {i+1}: {e}"
            print(f"❌ {error_msg}")
            print(f"   Continuing with next mission...")
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            failure_count += 1
            missions_completed = i + 1
            continue

        # Progress update and checkpoint at adaptive intervals
        if (i + 1) % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1 - start_mission) / elapsed if elapsed > 0 else 0
            remaining = (num_missions - i - 1) / rate if rate > 0 else 0

            print(f"\n{'='*70}")
            print(f"Progress: {i+1}/{num_missions} missions ({(i+1)/num_missions*100:.1f}%)")
            print(f"  Success: {success_count} | Failure: {failure_count} | Abort: {abort_count}")
            print(f"  Rate: {rate:.2f} missions/sec")
            print(f"  Est. time remaining: {remaining:.1f}s ({remaining/60:.1f}m)")
            print(f"{'='*70}")

            logger.info(f"Progress: {i+1}/{num_missions} ({(i+1)/num_missions*100:.1f}%) - "
                       f"Rate: {rate:.2f} m/s - ETA: {remaining/60:.1f}m")

            # Save checkpoint
            try:
                current_stats = {
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'abort_count': abort_count,
                    'elapsed_time': elapsed
                }
                save_checkpoint(output_dir, i + 1, current_stats,
                              team.get_team_stats(), data_logger.get_statistics())
                print(f"💾 Checkpoint saved at mission {i+1}")
                logger.debug(f"Checkpoint saved at mission {i+1}")
            except Exception as e:
                error_msg = f"Could not save checkpoint: {e}"
                print(f"⚠️  Warning: {error_msg}")
                logger.error(error_msg)
                logger.error(traceback.format_exc())

        # Save progress file more frequently for monitoring
        if (i + 1) % max(10, checkpoint_interval // 5) == 0:
            try:
                current_stats = {
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'abort_count': abort_count
                }
                save_progress_file(output_dir, i + 1, num_missions, current_stats, start_time)
            except Exception as e:
                logger.error(f"Could not save progress file: {e}")

        # Memory management for long runs
        if (i + 1) % max(100, checkpoint_interval * 2) == 0:
            logger.debug(f"Running garbage collection at mission {i+1}")
            gc.collect()  # Force garbage collection

    # Final statistics
    total_time = time.time() - start_time
    # missions_completed already set correctly in loop

    print("\n" + "="*70)
    if shutdown_requested:
        print("TRAINING SESSION INTERRUPTED (SAVED)")
        logger.warning("Training session interrupted by user/signal")
    else:
        print("TRAINING SESSION COMPLETE")
        logger.info("Training session completed successfully")
    print("="*70)

    print(f"\nSession Statistics:")
    print(f"  Missions planned: {num_missions}")
    print(f"  Missions completed: {missions_completed}")
    if missions_completed > 0:
        print(f"  Successful: {success_count} ({success_count/missions_completed*100:.1f}%)")
        print(f"  Failed: {failure_count} ({failure_count/missions_completed*100:.1f}%)")
        print(f"  Aborted: {abort_count} ({abort_count/missions_completed*100:.1f}%)")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    if missions_completed > 0:
        print(f"  Avg time per mission: {total_time/missions_completed:.2f}s")

    logger.info(f"Session complete: {missions_completed}/{num_missions} missions, "
               f"Success: {success_count}, Failed: {failure_count}, Aborted: {abort_count}")

    # Team statistics
    try:
        print(f"\nTeam Performance:")
        team_stats = team.get_team_stats()
        for key, value in team_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        logger.debug(f"Team stats: {team_stats}")
    except Exception as e:
        error_msg = f"Could not get team stats: {e}"
        print(f"⚠️  {error_msg}")
        logger.error(error_msg)

    # Logger statistics
    try:
        print(f"\nTraining Data:")
        logger_stats = data_logger.get_statistics()
        for key, value in logger_stats.items():
            print(f"  {key}: {value}")
        logger.debug(f"Logger stats: {logger_stats}")
    except Exception as e:
        error_msg = f"Could not get logger stats: {e}"
        print(f"⚠️  {error_msg}")
        logger.error(error_msg)

    # Export training data
    try:
        print(f"\n{'─'*70}")
        print("Exporting training data...")
        logger.info("Exporting training data for GPU training")
        index_file = data_logger.export_for_training()
        print(f"✓ Training data ready for GPU training")
        print(f"  Index file: {index_file}")
        print(f"{'─'*70}")
        logger.info(f"Training data exported: {index_file}")
    except Exception as e:
        error_msg = f"Error exporting training data: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())

    # Save final checkpoint
    try:
        final_stats = {
            'success_count': success_count,
            'failure_count': failure_count,
            'abort_count': abort_count,
            'elapsed_time': total_time,
            'completed': not shutdown_requested
        }
        save_checkpoint(output_dir, missions_completed, final_stats,
                       team.get_team_stats(), data_logger.get_statistics())
        print(f"\n💾 Final checkpoint saved")
        logger.info("Final checkpoint saved")
    except Exception as e:
        error_msg = f"Could not save final checkpoint: {e}"
        print(f"⚠️  Warning: {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())

    # Save final progress file
    try:
        final_stats_dict = {
            'success_count': success_count,
            'failure_count': failure_count,
            'abort_count': abort_count
        }
        save_progress_file(output_dir, missions_completed, num_missions, final_stats_dict, start_time)
        # Mark as completed
        progress_file = Path(output_dir) / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                data = json.load(f)
            data['status'] = 'completed' if not shutdown_requested else 'interrupted'
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save final progress file: {e}")

    if shutdown_requested:
        print("\n⚠️  Training interrupted. To resume, run the same command again.")
        print(f"   Progress saved at mission {missions_completed}")
    else:
        print("\n✓ Phase 1 training session complete!")

    print("="*70 + "\n")

    print("Next steps:")
    print("  1. Review training data in:", output_dir)
    print("  2. Start GPU training with:")
    print(f"     python synexs_gpu_trainer.py {Path(output_dir) / 'batches'}")
    if missions_completed < num_missions:
        print(f"  3. Resume training: python synexs_phase1_runner.py --missions {num_missions} --output {output_dir}")
    print()

    return {
        'team': team,
        'logger': logger,
        'generator': generator,
        'stats': {
            'total_missions': missions_completed,
            'success_count': success_count,
            'failure_count': failure_count,
            'abort_count': abort_count,
            'total_time': total_time,
            'interrupted': shutdown_requested
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Synexs Phase 1 - Team Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:          python3 synexs_phase1_runner.py --quick
  100 missions:        python3 synexs_phase1_runner.py --missions 100
  1000 missions:       python3 synexs_phase1_runner.py --missions 1000
  Resume from crash:   python3 synexs_phase1_runner.py --missions 1000
  Fresh start:         python3 synexs_phase1_runner.py --missions 1000 --no-resume
        """
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
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore existing checkpoint'
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick:
        print("\n⚡ Quick test mode: Running 10 missions")
        args.missions = 10

    # Run training session
    try:
        results = run_training_session(
            num_missions=args.missions,
            output_dir=args.output,
            team_id=args.team_id,
            resume=not args.no_resume
        )
        return 0
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
