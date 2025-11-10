#!/usr/bin/env python3
"""
Synexs Protocol V2 - AI-Optimized Communication
Replaces verbose Greek words with compact symbolic tokens
"""

import json
import random
import logging

# ==================== Protocol Definition ====================

# Core actions (10 primary operations)
CORE_ACTIONS = {
    "△": "SCAN",         # Detect honeypot patterns
    "□": "ATTACK",       # Execute payload
    "◆": "REPLICATE",    # Spawn new agent
    "◇": "MUTATE",       # Change signature
    "○": "EVADE",        # PTR detected, abort
    "●": "LEARN",        # Agent killed, update AI
    "◉": "REPORT",       # Send kill data
    "◎": "DEFEND",       # Localhost only mode
    "⬡": "REFINE",       # Optimize sequence
    "⬢": "FLAG",         # Anomaly detected
}

# Extended operations (16 additional)
EXTENDED_OPS = {
    "⊕": "XOR_PAYLOAD",   # XOR encoding
    "⊗": "ENCRYPT",       # Encryption
    "⊙": "COMPRESS",      # Compress data
    "⊚": "HASH_CHECK",    # Verify integrity
    "⊛": "SYNC",          # Synchronize swarm
    "⊜": "SPLIT",         # Divide task
    "⊝": "MERGE",         # Combine results
    "⊞": "STACK_PUSH",    # Push to stack
    "⊟": "STACK_POP",     # Pop from stack
    "⊠": "TERMINATE",     # Kill agent
    "⊡": "PAUSE",         # Sleep/wait
    "⊢": "LOG",           # Write log
    "⊣": "QUERY",         # Request info
    "⊤": "ACK",           # Acknowledge
    "⊥": "NACK",          # Negative ack
    "⊦": "CHECKPOINT",    # Save state
}

# Combine all tokens
PROTOCOL_V2 = {**CORE_ACTIONS, **EXTENDED_OPS}
REVERSE_PROTOCOL = {v: k for k, v in PROTOCOL_V2.items()}

# ==================== Encoding/Decoding ====================

def encode_action(action: str) -> str:
    """Convert human action to symbolic token"""
    return REVERSE_PROTOCOL.get(action, "?")

def decode_action(token: str) -> str:
    """Convert symbolic token to human action"""
    return PROTOCOL_V2.get(token, "UNKNOWN")

def encode_sequence(actions: list) -> str:
    """Convert action list to symbolic sequence"""
    return "".join([encode_action(a) for a in actions])

def decode_sequence(symbolic: str) -> list:
    """Convert symbolic sequence to action list"""
    return [decode_action(c) for c in symbolic]

# ==================== Efficiency Comparison ====================

def compare_protocols():
    """Compare old vs new protocol efficiency"""
    try:
        # Example message
        actions = ["REPLICATE", "MUTATE", "SCAN", "ATTACK", "EVADE"]

        # Old protocol (Greek words)
        old_tokens = ["SIGMA", "OMEGA", "THETA", "DELTA", "ZETA"]
        old_message = " ".join(old_tokens)
        old_size = len(old_message.encode('utf-8'))

        # New protocol (symbols)
        new_message = encode_sequence(actions)
        new_size = len(new_message.encode('utf-8'))

        print("=" * 60)
        print("📊 Protocol Efficiency Comparison")
        print("=" * 60)
        print(f"\n🔴 Old Protocol (Greek Words):")
        print(f"   Message: {old_message}")
        print(f"   Size: {old_size} bytes")
        print(f"   Tokens: {len(old_tokens)}")
        print(f"   Avg per token: {old_size / len(old_tokens):.1f} bytes")

        print(f"\n🟢 New Protocol (Symbols):")
        print(f"   Message: {new_message}")
        print(f"   Size: {new_size} bytes")
        print(f"   Tokens: {len(actions)}")
        print(f"   Avg per token: {new_size / len(actions):.1f} bytes")

        print(f"\n💡 Improvement:")
        reduction = ((old_size - new_size) / old_size) * 100
        print(f"   Size reduction: {reduction:.1f}%")
        print(f"   Bandwidth savings: {old_size - new_size} bytes per message")
        print(f"   Speedup: {old_size / new_size:.2f}x faster transmission")
        print("=" * 60)

        # Decode to verify
        print(f"\n🔍 Decoded actions: {decode_sequence(new_message)}")
    except Exception as e:
        logging.error(f"Error in compare_protocols: {e}")
        raise

# ==================== Training Data Generator ====================

def generate_training_data(num_samples: int = 100):
    """Generate training data using symbolic protocol"""
    try:
        training_data = []
        actions = list(PROTOCOL_V2.keys())
        meanings = list(PROTOCOL_V2.values())

        for _ in range(num_samples):
            # Random sequence length (3-8 tokens)
            seq_len = random.randint(3, 8)
            sequence = "".join(random.choices(actions, k=seq_len))

            # Decode to actions
            decoded = decode_sequence(sequence)

            # Generate natural language description
            if "REPLICATE" in decoded and "MUTATE" in decoded:
                description = "Spawn new agent with mutated signature. Swarm evolution."
            elif "SCAN" in decoded and "EVADE" in decoded:
                description = "Honeypot detected. Abort mission and evade."
            elif "ATTACK" in decoded and "REPORT" in decoded:
                description = "Execute payload and report results to swarm."
            elif "LEARN" in decoded:
                description = f"Agent terminated. Learn from failure: {', '.join(decoded[:3])}"
            else:
                description = f"Execute sequence: {' → '.join(decoded[:3])}"

            training_data.append({
                "instruction": f"What does {sequence} mean?",
                "input": "",
                "output": description
            })

        return training_data
    except Exception as e:
        logging.error(f"Error in generate_training_data: {e}")
        raise

# ==================== Export to New Vocab ====================

def export_vocab_v2():
    """Export Protocol V2 as vocab.json compatible format"""
    try:
        vocab_v2 = {}

        # Add control tokens
        vocab_v2["<PAD>"] = 0
        vocab_v2["<START>"] = 1
        vocab_v2["<END>"] = 2
        vocab_v2["<UNK>"] = 3

        # Add symbolic tokens
        idx = 4
        for symbol, action in PROTOCOL_V2.items():
            vocab_v2[symbol] = idx
            idx += 1

        return vocab_v2
    except Exception as e:
        logging.error(f"Error in export_vocab_v2: {e}")
        raise

# ==================== Main ====================

def main():
    try:
        print("\n🧠 Synexs Protocol V2 - AI-Optimized Communication\n")

        # Show comparison
        compare_protocols()

        # Generate sample training data
        print("\n📚 Sample Training Data (5 examples):\n")
        training_samples = generate_training_data(5)
        for i, sample in enumerate(training_samples, 1):
            print(f"{i}. Instruction: {sample['instruction']}")
            print(f"   Output: {sample['output']}\n")

        # Export vocab
        vocab_v2 = export_vocab_v2()
        print(f"📖 Protocol V2 Vocabulary: {len(vocab_v2)} tokens")
        print(f"   Core actions: {len(CORE_ACTIONS)}")
        print(f"   Extended ops: {len(EXTENDED_OPS)}")
        print(f"   Total capacity: {len(vocab_v2)} unique symbols")

        # Save to file
        with open("vocab_v2.json", "w") as f:
            json.dump(vocab_v2, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Exported to: vocab_v2.json")

        # Save training data
        with open("training_symbolic_v2.jsonl", "w") as f:
            for sample in generate_training_data(50):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"✅ Generated 50 training samples: training_symbolic_v2.jsonl")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s')
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        raise