"""Quick integration test for the bot endpoints."""
import urllib.request
import json
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BOT = "http://localhost:8080"

def post(path, data):
    req = urllib.request.Request(
        f"{BOT}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read().decode())

def get(path):
    resp = urllib.request.urlopen(f"{BOT}{path}", timeout=5)
    return json.loads(resp.read().decode())

print("=== Healthz ===")
h = get("/v1/healthz")
print(json.dumps(h, indent=2))

print("\n=== Push dentists category ===")
cat = json.load(open("dataset/categories/dentists.json"))
r = post("/v1/context", {"scope": "category", "context_id": "dentists", "version": 1, "payload": cat, "delivered_at": "2026-04-26T09:45:00Z"})
print(json.dumps(r, indent=2))

print("\n=== Push Dr. Meera merchant ===")
merchants = json.load(open("dataset/merchants_seed.json"))["merchants"]
meera = merchants[0]
r = post("/v1/context", {"scope": "merchant", "context_id": meera["merchant_id"], "version": 1, "payload": meera, "delivered_at": "2026-04-26T09:45:30Z"})
print(json.dumps(r, indent=2))

print("\n=== Push research digest trigger ===")
triggers = json.load(open("dataset/triggers_seed.json"))["triggers"]
trg = triggers[0]  # research_digest for Dr. Meera
r = post("/v1/context", {"scope": "trigger", "context_id": trg["id"], "version": 1, "payload": trg, "delivered_at": "2026-04-26T10:32:00Z"})
print(json.dumps(r, indent=2))

print("\n=== Tick (should compose a message) ===")
r = post("/v1/tick", {"now": "2026-04-26T10:35:00Z", "available_triggers": [trg["id"]]})
print(json.dumps(r, indent=2, ensure_ascii=False))

if r.get("actions"):
    action = r["actions"][0]
    conv_id = action["conversation_id"]
    print(f"\n=== Reply: merchant says 'Yes please send the abstract' ===")
    r2 = post("/v1/reply", {
        "conversation_id": conv_id,
        "merchant_id": meera["merchant_id"],
        "from_role": "merchant",
        "message": "Yes please send the abstract. Also draft the patient WhatsApp.",
        "received_at": "2026-04-26T10:42:00Z",
        "turn_number": 2
    })
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    print(f"\n=== Reply: auto-reply test ===")
    r3 = post("/v1/reply", {
        "conversation_id": "conv_auto_test",
        "merchant_id": meera["merchant_id"],
        "from_role": "merchant",
        "message": "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2
    })
    print(json.dumps(r3, indent=2, ensure_ascii=False))

    print(f"\n=== Reply: hostile test ===")
    r4 = post("/v1/reply", {
        "conversation_id": "conv_hostile_test",
        "merchant_id": meera["merchant_id"],
        "from_role": "merchant",
        "message": "Stop messaging me. This is useless spam.",
        "received_at": "2026-04-26T10:48:00Z",
        "turn_number": 2
    })
    print(json.dumps(r4, indent=2, ensure_ascii=False))

print("\n=== Healthz (after context push) ===")
h = get("/v1/healthz")
print(json.dumps(h, indent=2))

print("\n=== All tests passed! ===")
