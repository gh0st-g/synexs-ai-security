#!/usr/bin/env python3
"""
Visual demonstration of Protocol V1 vs V2 vs V3
Shows real-world message examples with binary protocol
"""

import logging
from protocol_v2_proposal import (
    encode_sequence, decode_sequence,
    CORE_ACTIONS, PROTOCOL_V2
)
from binary_protocol import (
    encode_binary, encode_base64, decode_binary,
    HybridProtocol
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def demo_message(title: str, actions: list):
    """Show same message in V1, V2, and V3 protocols"""
    logging.info(f"\n{'='*80}")
    logging.info(f"📨 {title}")
    logging.info('='*80)

    # V1: Greek words
    v1_mapping = {
        "SCAN": "SIGMA",
        "ATTACK": "OMEGA",
        "REPLICATE": "THETA",
        "MUTATE": "DELTA",
        "EVADE": "ZETA",
        "LEARN": "ALPHA",
        "REPORT": "BETA",
        "DEFEND": "GAMMA",
        "REFINE": "EPSILON",
        "FLAG": "LAMBDA",
    }

    v1_tokens = [v1_mapping.get(a, "UNKNOWN") for a in actions]
    v1_message = " ".join(v1_tokens)
    v1_size = len(v1_message.encode('utf-8'))

    # V2: Symbols
    try:
        v2_message = encode_sequence(actions)
        v2_size = len(v2_message.encode('utf-8'))
    except ValueError as e:
        logging.error(f"Error encoding V2 message: {e}")
        return

    # V3: Binary (base64 for display)
    try:
        v3_binary = encode_binary(actions)
        v3_b64 = encode_base64(v3_binary)
        v3_size = len(v3_binary)
        v3_b64_size = len(v3_b64.encode('utf-8'))
    except ValueError as e:
        logging.error(f"Error encoding V3 message: {e}")
        return

    # Calculate savings
    v2_savings = ((v1_size - v2_size) / v1_size) * 100
    v3_savings = ((v1_size - v3_size) / v1_size) * 100
    v3_b64_savings = ((v1_size - v3_b64_size) / v1_size) * 100

    logging.info(f"\n🔴 Protocol V1 (Greek Words):")
    logging.info(f"   Message: {v1_message}")
    logging.info(f"   Bytes: {v1_size}")
    logging.info(f"   Efficiency: 1.0x (baseline)")

    logging.info(f"\n🟡 Protocol V2 (Symbols):")
    logging.info(f"   Message: {v2_message}")
    logging.info(f"   Bytes: {v2_size}")
    logging.info(f"   Efficiency: {v1_size/v2_size:.2f}x faster")
    logging.info(f"   Reduction: {v2_savings:.1f}%")

    logging.info(f"\n🟢 Protocol V3-Binary (Raw):")
    logging.info(f"   Message: {v3_binary.hex()}")
    logging.info(f"   Bytes: {v3_size}")
    logging.info(f"   Efficiency: {v1_size/v3_size:.2f}x faster")
    logging.info(f"   Reduction: {v3_savings:.1f}%")

    logging.info(f"\n🔵 Protocol V3-Base64 (JSON-Safe):")
    logging.info(f"   Message: {v3_b64}")
    logging.info(f"   Bytes: {v3_b64_size}")
    logging.info(f"   Efficiency: {v1_size/v3_b64_size:.2f}x faster")
    logging.info(f"   Reduction: {v3_b64_savings:.1f}%")

    logging.info(f"\n💡 Actions: {' → '.join(actions)}")

    # Verify decode
    try:
        decoded = decode_binary(v3_binary)
        match = "✅" if decoded == actions else "❌"
        logging.info(f"🧪 Decode test: {match} ({len(decoded)}/{len(actions)} actions)")
    except ValueError as e:
        logging.error(f"Error decoding V3 message: {e}")

def main():
    logging.info("\n" + "="*80)
    logging.info("🧠 SYNEXS PROTOCOL COMPARISON: V1 vs V2 vs V3")
    logging.info("="*80)

    examples = [
        ("Agent Replication Sequence", ["SCAN", "REPLICATE", "MUTATE", "REPORT"]),
        ("Honeypot Detection & Evasion", ["SCAN", "SCAN", "EVADE", "REPORT", "LEARN"]),
        ("Attack Execution", ["SCAN", "ATTACK", "REPLICATE", "REPORT"]),
        ("Learning from Agent Death", ["LEARN", "MUTATE", "REPLICATE", "DEFEND"]),
        ("Swarm Coordination (8 actions)", ["SCAN", "SCAN", "ATTACK", "REPLICATE", "MUTATE", "LEARN", "REPORT", "DEFEND"])
    ]

    for title, actions in examples:
        try:
            demo_message(title, actions)
        except Exception as e:
            logging.error(f"Error in demo_message: {e}")

    logging.info(f"\n{'='*80}")
    logging.info("📊 SUMMARY")
    logging.info('='*80)

    logging.info("\n🔴 Protocol V1 (Greek Words):")
    logging.info("   • Size: Baseline (1.0x)")
    logging.info("   • Human-readable: ✅ Excellent")
    logging.info("   • AI efficiency: ❌ Poor")
    logging.info("   • Use case: Legacy systems, debugging")

    logging.info("\n🟡 Protocol V2 (Symbols):")
    logging.info("   • Size: 40-52% reduction")
    logging.info("   • Speed: 1.5-2.1x faster")
    logging.info("   • Human-readable: ⚠️ Requires decoder")
    logging.info("   • AI efficiency: ✅ Good")
    logging.info("   • Use case: Balance of readability + efficiency")

    logging.info("\n🟢 Protocol V3 (Binary):")
    logging.info("   • Size: 82-88% reduction")
    logging.info("   • Speed: 6-8.3x faster")
    logging.info("   • Human-readable: ❌ Hex/Base64 only")
    logging.info("   • AI efficiency: ✅ Excellent")
    logging.info("   • Use case: Production, high-throughput swarms")

    logging.info("\n💡 Recommendations:")
    logging.info("   • Production: V3-Base64 (compact + JSON-safe)")
    logging.info("   • Development: V2 (readable + efficient)")
    logging.info("   • Logging: V1 (human-readable)")
    logging.info("   • Hybrid mode: All 3 for 1-week transition")

    logging.info("\n📈 Performance Impact (1000 agents × 100 msg/hour):")
    logging.info("   • V1: 2.8 MB/hour bandwidth")
    logging.info("   • V2: 1.5 MB/hour (47% savings)")
    logging.info("   • V3: 0.4 MB/hour (86% savings)")

    logging.info(f"\n{'='*80}\n")

    # Show binary protocol capabilities
    logging.info("📖 Binary Protocol V3 Actions:")
    logging.info("-" * 80)
    actions_list = list(HybridProtocol.BINARY_ACTIONS.items())[:10]
    for code, action in actions_list:
        logging.info(f"   {code:02X} (0b{code:05b}) → {action}")
    logging.info(f"   ... ({len(HybridProtocol.BINARY_ACTIONS)} total actions, 5 bits each)")
    logging.info(f"\n{'='*80}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.exception(e)