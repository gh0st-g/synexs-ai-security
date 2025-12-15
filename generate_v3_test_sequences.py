#!/usr/bin/env python3
"""
Generate V3 Binary Protocol Test Sequences
Creates test data for validating the complete data flow pipeline
"""

import json
import random
import time
from pathlib import Path

# V3 Protocol Actions (from vocab_v3_binary.json)
V3_ACTIONS = [
    "SCAN", "ATTACK", "REPLICATE", "MUTATE", "EVADE",
    "LEARN", "REPORT", "DEFEND", "REFINE", "FLAG",
    "XOR_PAYLOAD", "ENCRYPT", "COMPRESS", "HASH_CHECK", "SYNC",
    "SPLIT", "MERGE", "STACK_PUSH", "STACK_POP", "TERMINATE",
    "PAUSE", "LOG", "QUERY", "ACK", "NACK",
    "CHECKPOINT", "VALIDATE", "BROADCAST", "LISTEN", "ROUTE",
    "FILTER", "TRANSFORM"
]

def generate_v3_sequences(num_sequences=50, min_length=3, max_length=8):
    """
    Generate V3 protocol sequences

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of sequences with metadata
    """
    sequences = []

    for i in range(num_sequences):
        # Random sequence length
        length = random.randint(min_length, max_length)

        # Generate random action sequence
        actions = random.choices(V3_ACTIONS, k=length)

        # Create sequence string (space-separated for compatibility with existing cells)
        sequence_str = " ".join(actions)

        sequences.append({
            "sequence": sequence_str,
            "original": sequence_str,
            "refined": sequence_str,
            "purpose": "V3 binary protocol test",
            "protocol": "v3",
            "actions": actions,
            "timestamp": int(time.time())
        })

    return sequences

def save_to_refined(sequences, filename=None):
    """Save sequences to datasets/refined/ directory"""
    refined_dir = Path("datasets/refined")
    refined_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"refined_v3_test_{int(time.time())}.json"

    filepath = refined_dir / filename

    with open(filepath, 'w') as f:
        json.dump(sequences, f, indent=2)

    return filepath

def save_to_generated(sequences, filename=None):
    """Save sequences to datasets/generated/ directory"""
    generated_dir = Path("datasets/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"generated_v3_test_{int(time.time())}.json"

    filepath = generated_dir / filename

    # Generated format uses "sequences" wrapper
    data = {"sequences": sequences}

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath

def main():
    print("=" * 60)
    print("V3 Binary Protocol - Test Sequence Generator")
    print("=" * 60)

    # Generate test sequences
    num_sequences = 100
    sequences = generate_v3_sequences(num_sequences)

    print(f"\n✅ Generated {len(sequences)} V3 protocol sequences")
    print(f"   Sequence lengths: 3-8 actions")
    print(f"   Protocol: V3 Binary ({len(V3_ACTIONS)} total actions)")

    # Show samples
    print(f"\n📋 Sample sequences:")
    for i, seq in enumerate(sequences[:5], 1):
        print(f"   {i}. {seq['sequence'][:60]}...")
        print(f"      Actions: {len(seq['actions'])}")

    # Save to both directories for testing
    refined_path = save_to_refined(sequences)
    generated_path = save_to_generated(sequences)

    print(f"\n💾 Saved test data:")
    print(f"   Refined: {refined_path}")
    print(f"   Generated: {generated_path}")

    # Statistics
    action_counts = {}
    total_actions = 0
    for seq in sequences:
        total_actions += len(seq['actions'])
        for action in seq['actions']:
            action_counts[action] = action_counts.get(action, 0) + 1

    print(f"\n📊 Statistics:")
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Total actions: {total_actions}")
    print(f"   Avg actions/sequence: {total_actions/len(sequences):.1f}")
    print(f"   Unique actions used: {len(action_counts)}/{len(V3_ACTIONS)}")

    print(f"\n   Top 10 actions:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"      {action}: {count}")

    print(f"\n✅ Ready for testing! Run cell_006.py to classify these sequences.")

if __name__ == "__main__":
    main()
