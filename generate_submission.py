#!/usr/bin/env python3
"""
Generate submission.jsonl — 30 test pair compositions.

Usage:
    1. Set GEMINI_API_KEY environment variable
    2. python generate_submission.py

Reads test_pairs.json from the expanded dataset and produces submission.jsonl
"""

import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add parent dir
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

DATASET_DIR = Path(__file__).parent / "dataset" / "expanded"
OUTPUT_FILE = Path(__file__).parent / "submission.jsonl"


def load_dataset():
    """Load the expanded dataset."""
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    # Categories
    cat_dir = DATASET_DIR / "categories"
    for f in cat_dir.glob("*.json"):
        data = json.load(open(f))
        categories[data["slug"]] = data

    # Merchants
    m_dir = DATASET_DIR / "merchants"
    for f in m_dir.glob("*.json"):
        data = json.load(open(f))
        merchants[data["merchant_id"]] = data

    # Customers
    c_dir = DATASET_DIR / "customers"
    for f in c_dir.glob("*.json"):
        data = json.load(open(f))
        customers[data["customer_id"]] = data

    # Triggers
    t_dir = DATASET_DIR / "triggers"
    for f in t_dir.glob("*.json"):
        data = json.load(open(f))
        triggers[data["id"]] = data

    # Test pairs
    with open(DATASET_DIR / "test_pairs.json") as f:
        test_pairs = json.load(f)["pairs"]

    return categories, merchants, customers, triggers, test_pairs


def main():
    from bot import compose_message, get_category_for_merchant

    print("Loading dataset...")
    categories, merchants, customers, triggers, test_pairs = load_dataset()
    print(f"  {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers, {len(test_pairs)} test pairs")

    # Populate the bot's context store
    from bot import contexts
    for slug, cat in categories.items():
        contexts[("category", slug)] = {"version": 1, "payload": cat}
    for mid, m in merchants.items():
        contexts[("merchant", mid)] = {"version": 1, "payload": m}
    for cid, c in customers.items():
        contexts[("customer", cid)] = {"version": 1, "payload": c}
    for tid, t in triggers.items():
        contexts[("trigger", tid)] = {"version": 1, "payload": t}

    print(f"\nGenerating {len(test_pairs)} submissions...\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, pair in enumerate(test_pairs):
            test_id = pair["test_id"]
            trigger_id = pair["trigger_id"]
            merchant_id = pair["merchant_id"]
            customer_id = pair.get("customer_id")

            trigger = triggers.get(trigger_id, {})
            merchant = merchants.get(merchant_id, {})
            category = categories.get(merchant.get("category_slug", ""), {})
            customer = customers.get(customer_id) if customer_id else None

            print(f"  [{test_id}] {trigger.get('kind', '?')} → {merchant.get('identity', {}).get('name', '?')}", end="")

            try:
                result = compose_message(category, merchant, trigger, customer)
                line = {
                    "test_id": test_id,
                    "body": result["body"],
                    "cta": result.get("cta", "open_ended"),
                    "send_as": result.get("send_as", "vera"),
                    "suppression_key": result.get("suppression_key", ""),
                    "rationale": result.get("rationale", ""),
                }
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
                print(f" ✓ ({len(result['body'])} chars)")
            except Exception as e:
                print(f" ✗ ERROR: {e}")
                line = {
                    "test_id": test_id,
                    "body": f"Hi, update regarding your business profile. Reply for details.",
                    "cta": "open_ended",
                    "send_as": "vera",
                    "suppression_key": trigger.get("suppression_key", ""),
                    "rationale": "Fallback due to composition error",
                }
                out.write(json.dumps(line, ensure_ascii=False) + "\n")

            # Rate limit for API
            time.sleep(1)

    print(f"\n✓ Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
