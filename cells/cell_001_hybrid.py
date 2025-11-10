#!/usr/bin/env python3
"""
cell_001_hybrid.py - Multi-Protocol Symbolic Sequence Generator
Supports V1 (Greek), V2 (Symbols), V3 (Binary) protocols
Backward compatible with 1-week hybrid mode
"""

import os
import sys
import json
import time
import random

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binary_protocol import (
    HybridProtocol,
    encode_base64,
    BINARY_ACTIONS
)

# ==================== Configuration ====================

# Protocol selection (can be changed via environment)
PROTOCOL_MODE = os.environ.get("SYNEXS_PROTOCOL", "v3-hybrid")  # v1, v2, v3, v3-hybrid
BATCH_SIZE = 50
OUTPUT_DIR = "datasets/generated"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== Vocabularies ====================

# V1: Greek words (legacy)
VOCAB_V1 = ["SIGMA", "OMEGA", "THETA", "DELTA", "ZETA", "ALPHA"]

# V2: Core actions (for symbols)
VOCAB_V2 = ["SCAN", "ATTACK", "REPLICATE", "MUTATE", "EVADE", "LEARN", "REPORT", "DEFEND", "REFINE", "FLAG"]

# V3: All binary actions
VOCAB_V3 = list(BINARY_ACTIONS.values())

# ==================== Generator Functions ====================

def generate_sequence_v1(length: int = 8) -> str:
    """Generate V1 (Greek words) sequence"""
    tokens = random.choices(VOCAB_V1, k=length)
    return " ".join(tokens)

def generate_sequence_v2(length: int = 8) -> str:
    """Generate V2 (Symbolic) sequence"""
    from protocol_v2_proposal import encode_sequence
    actions = random.choices(VOCAB_V2, k=length)
    return encode_sequence(actions)

def generate_sequence_v3(length: int = 8, format: str = "base64") -> dict:
    """Generate V3 (Binary) sequence"""
    actions = random.choices(VOCAB_V3, k=length)

    if format == "base64":
        encoded = encode_base64(actions)
    elif format == "hex":
        from binary_protocol import encode_hex
        encoded = encode_hex(actions)
    else:
        from binary_protocol import encode_binary
        encoded = encode_binary(actions).hex()

    return {
        "sequence": encoded,
        "format": format,
        "actions": actions,  # Ground truth
        "protocol": "v3"
    }

def generate_hybrid_batch(batch_size: int = 50) -> dict:
    """Generate batch with multiple protocols for comparison"""

    sequences_v1 = []
    sequences_v2 = []
    sequences_v3 = []

    for _ in range(batch_size):
        length = random.randint(4, 8)

        # V1: Greek words
        v1_seq = generate_sequence_v1(length)
        sequences_v1.append({
            "sequence": v1_seq,
            "protocol": "v1",
            "purpose": "Legacy compatibility"
        })

        # V2: Symbols
        v2_seq = generate_sequence_v2(length)
        sequences_v2.append({
            "sequence": v2_seq,
            "protocol": "v2",
            "purpose": "Human-readable symbolic"
        })

        # V3: Binary (base64)
        v3_data = generate_sequence_v3(length, format="base64")
        sequences_v3.append({
            "sequence": v3_data["sequence"],
            "protocol": "v3",
            "format": "base64",
            "purpose": "Ultra-compact binary",
            "actions": v3_data["actions"]
        })

    return {
        "v1": sequences_v1,
        "v2": sequences_v2,
        "v3": sequences_v3,
        "metadata": {
            "batch_size": batch_size,
            "timestamp": int(time.time()),
            "mode": "hybrid",
            "protocols": ["v1", "v2", "v3"]
        }
    }

# ==================== Main Function ====================

def main():
    """Main generator - supports all protocol modes"""

    print(f"🧠 [cell_001_hybrid] Protocol mode: {PROTOCOL_MODE}")
    print(f"📦 [cell_001_hybrid] Batch size: {BATCH_SIZE}")

    timestamp = int(time.time())

    if PROTOCOL_MODE == "v1":
        # Pure V1 (backward compatible)
        sequences = [
            {"sequence": generate_sequence_v1(), "protocol": "v1"}
            for _ in range(BATCH_SIZE)
        ]
        filename = f"{OUTPUT_DIR}/generated_v1_{timestamp}.json"

    elif PROTOCOL_MODE == "v2":
        # Pure V2 (symbolic)
        sequences = [
            {"sequence": generate_sequence_v2(), "protocol": "v2"}
            for _ in range(BATCH_SIZE)
        ]
        filename = f"{OUTPUT_DIR}/generated_v2_{timestamp}.json"

    elif PROTOCOL_MODE == "v3":
        # Pure V3 (binary)
        sequences = [
            generate_sequence_v3()
            for _ in range(BATCH_SIZE)
        ]
        filename = f"{OUTPUT_DIR}/generated_v3_{timestamp}.json"

    elif PROTOCOL_MODE == "v3-hybrid":
        # Hybrid mode - all protocols (1-week transition)
        batch = generate_hybrid_batch(BATCH_SIZE)
        filename = f"{OUTPUT_DIR}/generated_hybrid_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(batch, f, indent=2)

        # Statistics
        v1_total = sum(len(s["sequence"].encode()) for s in batch["v1"])
        v2_total = sum(len(s["sequence"].encode()) for s in batch["v2"])
        v3_total = sum(len(s["sequence"].encode()) for s in batch["v3"])

        print(f"✅ [cell_001_hybrid] Generated {BATCH_SIZE} sequences per protocol")
        print(f"   V1 total: {v1_total} bytes")
        print(f"   V2 total: {v2_total} bytes ({((v1_total-v2_total)/v1_total)*100:.1f}% reduction)")
        print(f"   V3 total: {v3_total} bytes ({((v1_total-v3_total)/v1_total)*100:.1f}% reduction)")
        print(f"→  Saved to: {filename}")
        return

    else:
        raise ValueError(f"Unknown protocol mode: {PROTOCOL_MODE}")

    # Save single-protocol output
    with open(filename, "w") as f:
        json.dump({"sequences": sequences}, f, indent=2)

    print(f"✅ [cell_001_hybrid] Generated {len(sequences)} sequences ({PROTOCOL_MODE})")
    print(f"→  Saved to: {filename}")

# ==================== Entry Point ====================

if __name__ == "__main__":
    main()
