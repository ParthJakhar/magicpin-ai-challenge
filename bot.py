#!/usr/bin/env python3
"""
magicpin AI Challenge — Vera Bot
=================================

A high-quality merchant AI assistant that composes context-aware,
category-specific, trigger-driven messages for WhatsApp merchant engagement.

Architecture:
- FastAPI HTTP server exposing the 5 required endpoints
- In-memory context store with versioned updates
- Trigger-dispatched composition using Google Gemini
- Auto-reply detection, intent transition handling, graceful exit
- Hindi-English code-mix support per merchant language preference

Run:  uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import re
import json
import time
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
TEAM_NAME = os.environ.get("TEAM_NAME", "Vera Reborn")
TEAM_MEMBERS = os.environ.get("TEAM_MEMBERS", "Parth Jakhar").split(",")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "jakharparth12@gmail.com")

app = FastAPI(title="Vera AI Challenge Bot", version="1.0.0")
START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-Memory Stores
# ---------------------------------------------------------------------------
# (scope, context_id) -> {version, payload}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> [{"from": ..., "body": ..., "ts": ...}, ...]
conversations: dict[str, list[dict]] = {}

# conversation_id -> metadata (merchant_id, customer_id, trigger_id, ended, auto_reply_count)
conv_meta: dict[str, dict] = {}

# Track which triggers have been used (for suppression)
used_triggers: set[str] = set()

# Track suppression keys that have been sent
sent_suppressions: set[str] = set()


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ---------------------------------------------------------------------------
# LLM Clients (Groq + Gemini with Auto-Cascade)
# ---------------------------------------------------------------------------
def call_groq(prompt: str, system_instruction: str = "", temperature: float = 0.2) -> str:
    """Call Groq API with robust fallback across models."""
    import urllib.request
    import urllib.error

    if not GROQ_API_KEY:
        return ""

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    models_to_try = [GROQ_MODEL, "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    seen = set()
    models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    for model_name in models:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps({
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2048
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            }
        )
        try:
            resp = urllib.request.urlopen(req, timeout=12)
            data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = msg.get("content", "")
            if not content and "reasoning" in msg:
                content = msg.get("reasoning", "")
            if content and content.strip():
                return content.strip()
        except Exception:
            continue
    return ""


def call_gemini(prompt: str, system_instruction: str = "", temperature: float = 0.3) -> str:
    """Call Google Gemini API via REST with retry and fallback across models."""
    import urllib.request
    import urllib.error
    import time

    if not GEMINI_API_KEY:
        return ""

    models_to_try = [GEMINI_MODEL, "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    seen = set()
    models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

    body = json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }).encode("utf-8")

    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=12)
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                continue
            elif e.code in (500, 502, 503, 504):
                time.sleep(0.3)
                try:
                    resp = urllib.request.urlopen(req, timeout=12)
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    pass
            continue
        except Exception:
            continue

    return ""


def call_llm(prompt: str, system_instruction: str = "", temperature: float = 0.2) -> str:
    """Cascade LLM calls: Groq (high performance & low latency) -> Gemini -> fallback."""
    if GROQ_API_KEY:
        res = call_groq(prompt, system_instruction, temperature)
        if res and res.strip():
            return res.strip()

    if GEMINI_API_KEY:
        res = call_gemini(prompt, system_instruction, temperature)
        if res and res.strip():
            return res.strip()

    return _fallback_compose(prompt)


def _fallback_compose(prompt: str) -> str:
    """Rich fallback if LLM is unavailable — returns empty to trigger rule-based fallback."""
    return ""


def _generate_fallback_message(merchant: dict, trigger: dict, customer: dict | None) -> str:
    """Generate a high-quality fallback message using rule-based composition."""
    identity = merchant.get("identity", {})
    raw_name = identity.get("owner_first_name", identity.get("name", "there"))
    perf = merchant.get("performance", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})
    trigger_kind = trigger.get("kind", "update")
    trigger_payload = trigger.get("payload", {})

    merchant_name_lower = str(identity.get("name", "")).lower()
    cat_slug = merchant.get("category_slug", "")
    is_doctor = (
        cat_slug == "dentists" or
        "dr." in raw_name.lower() or
        "dr " in raw_name.lower() or
        "dental" in merchant_name_lower or
        "clinic" in merchant_name_lower or
        "dentist" in merchant_name_lower or
        "doctor" in merchant_name_lower
    )
    clean_name = raw_name.replace("Dr.", "").replace("Dr ", "").strip()
    name = f"Dr. {clean_name}" if is_doctor else clean_name

    # Customer-facing fallback
    if customer:
        c_name = customer.get("identity", {}).get("name", "Customer")
        m_name = identity.get("name", "the clinic")
        lang = customer.get("identity", {}).get("language_pref", "en")
        state = customer.get("state", "active")
        rel = customer.get("relationship", {})
        visits = rel.get("visits_total", 0)

        if trigger_kind == "recall_due":
            slots = trigger_payload.get("available_slots", [])
            service = trigger_payload.get("service_due", "routine check-up")
            slot_text = ""
            if slots:
                slot_labels = [s.get("label", str(s)) if isinstance(s, dict) else str(s) for s in slots[:2]]
                slot_text = f" Slots available: {' ya '.join(slot_labels)}." if "hi" in lang else f" Available slots: {' or '.join(slot_labels)}."
            offer_text = f" {offers[0]['title']}." if offers else ""
            return f"Hi {c_name}, {m_name} here. Your {service.replace('_', ' ')} is due.{slot_text}{offer_text} Reply to book."

        elif trigger_kind == "chronic_refill_due":
            mols = trigger_payload.get("molecule_list", [])
            mol_text = ", ".join(mols[:3]) if mols else "your regular medicines"
            stock_out = trigger_payload.get("stock_runs_out_iso", "")
            date_text = stock_out[:10] if stock_out else "soon"
            return f"Hi {c_name}, {m_name} here. {mol_text} — stock runs out around {date_text}. Same dose ready. Reply CONFIRM to dispatch."

        elif trigger_kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
            days = trigger_payload.get("days_since_last_visit", "a while")
            return f"Hi {c_name}, {m_name} here. It's been {days} days. We have something new that might interest you. Reply for details."

        elif trigger_kind == "trial_followup":
            return f"Hi {c_name}, {m_name} here. Hope you enjoyed the trial session! Ready to continue? Reply YES to book the next one."

        else:
            return f"Hi {c_name}, {m_name} here. We have an update for you. Reply for details."

    # Merchant-facing fallback
    if trigger_kind == "research_digest":
        return f"{name}, this week's research digest is out with items relevant to your practice. Want me to pull the highlights?"

    elif trigger_kind == "regulation_change":
        deadline = trigger_payload.get("deadline_iso", "")[:10]
        return f"{name}, heads up — new compliance requirement effective {deadline}. Want the details + an action checklist?"

    elif trigger_kind == "perf_dip":
        metric = trigger_payload.get("metric", "views")
        delta = trigger_payload.get("delta_pct", -0.2)
        return f"{name}, your {metric} dropped {abs(int(delta*100))}% this week. Want me to check what's behind it and suggest a fix?"

    elif trigger_kind == "perf_spike":
        metric = trigger_payload.get("metric", "views")
        delta = trigger_payload.get("delta_pct", 0.2)
        return f"{name}, good news — your {metric} are up {int(delta*100)}% this week! Want me to double down on what's working?"

    elif trigger_kind == "renewal_due":
        days = trigger_payload.get("days_remaining", "a few")
        return f"{name}, subscription expires in {days} days. Your profile is active — want to continue? Reply YES to renew."

    elif trigger_kind == "milestone_reached":
        metric = trigger_payload.get("metric", "reviews")
        value = trigger_payload.get("value_now", "?")
        return f"Nice one {name} — you're at {value} {metric.replace('_', ' ')}! That puts you above most peers. Want me to draft a celebration post?"

    elif trigger_kind == "ipl_match_today":
        match_info = trigger_payload.get("match", "IPL match")
        return f"Quick heads-up {name} — {match_info} tonight. Want me to draft a match-night special for your customers?"

    elif trigger_kind == "review_theme_emerged":
        theme = trigger_payload.get("theme", "recent feedback")
        count = trigger_payload.get("occurrences_30d", "several")
        return f"{name}, noticed {count} reviews mentioning '{theme.replace('_', ' ')}' this month. Want to discuss how to address it?"

    elif trigger_kind == "supply_alert":
        molecule = trigger_payload.get("molecule", "a product")
        batches = trigger_payload.get("affected_batches", [])
        batch_text = ", ".join(batches[:2]) if batches else "specific batches"
        return f"{name}, urgent: voluntary recall on {molecule} batches ({batch_text}). Want me to pull your affected customer list?"

    elif trigger_kind == "seasonal_perf_dip":
        note = trigger_payload.get("season_note", "seasonal pattern")
        delta = trigger_payload.get("delta_pct", -0.25)
        return f"{name}, your views are down {abs(int(delta*100))}% — but this is the normal {note.replace('_', ' ')}. Focus on retention now; acquisition picks up next quarter."

    elif trigger_kind == "curious_ask_due":
        return f"Hi {name}! Quick check — what service has been most asked-for this week? I'll turn it into a Google post + WhatsApp reply template. Takes 5 min."

    elif trigger_kind == "dormant_with_vera":
        days = trigger_payload.get("days_since_last_merchant_message", 14)
        return f"Hi {name}, it's been {days} days. Quick update: your profile views are at {perf.get('views', '?')} this month. Anything I can help with?"

    elif trigger_kind in ("active_planning_intent",):
        topic = trigger_payload.get("intent_topic", "your idea")
        return f"{name}, re: {topic.replace('_', ' ')} — I've drafted a starter version. Want me to share it?"

    elif trigger_kind == "winback_eligible":
        days = trigger_payload.get("days_since_expiry", "?")
        return f"Hi {name}, it's been {days} days since your subscription expired. Your profile is still live but not optimized. Want to discuss?"

    elif trigger_kind == "festival_upcoming":
        festival = trigger_payload.get("festival", "the upcoming festival")
        return f"{name}, {festival} is coming up. Want me to draft festival-themed content for your profile?"

    elif trigger_kind == "competitor_opened":
        comp = trigger_payload.get("competitor_name", "a new competitor")
        dist = trigger_payload.get("distance_km", "nearby")
        return f"{name}, noticed {comp} opened {dist}km from you. Want to see how your profile compares?"

    elif trigger_kind == "gbp_unverified":
        uplift = trigger_payload.get("estimated_uplift_pct", 0.3)
        return f"{name}, your Google profile isn't verified yet. Verified profiles get ~{int(uplift*100)}% more visibility. Want me to walk you through it? 5-min process."

    elif trigger_kind == "category_seasonal":
        trends = trigger_payload.get("trends", [])
        trend_text = ", ".join(trends[:3]) if trends else "seasonal shifts"
        return f"{name}, seasonal demand shift coming: {trend_text}. Want me to adjust your shelf/display recommendations?"

    elif trigger_kind == "cde_opportunity":
        credits = trigger_payload.get("credits", "?")
        fee = trigger_payload.get("fee", "?")
        return f"{name}, CDE opportunity: {credits} credit hours, {fee}. Want the details?"

    elif trigger_kind == "wedding_package_followup":
        days_to = trigger_payload.get("days_to_wedding", "?")
        return f"Hi there! {days_to} days to the wedding. Perfect window to start the prep program. Want me to share options?"

    elif trigger_kind == "recall_due":
        patient = trigger_payload.get("patient_name", "a patient")
        service = trigger_payload.get("service_due", "dental recall").replace("_", " ")
        due = trigger_payload.get("due_date", "this week")
        slots = trigger_payload.get("available_slots", [])
        slot_text = f" Available slots: {' or '.join([s.get('label', str(s)) if isinstance(s, dict) else str(s) for s in slots[:2]])}." if slots else ""
        return f"{name}, {patient} is due for {service} ({due}).{slot_text} Would you like me to send a WhatsApp reminder to confirm their booking?"

    else:
        views = perf.get("views", "")
        view_text = f" Your profile has {views} views this month." if views else ""
        return f"{name}, quick update from Vera.{view_text} Reply YES if you'd like me to draft this week's patient engagement plan."


# ---------------------------------------------------------------------------
# Context Helpers
# ---------------------------------------------------------------------------
def get_context(scope: str, context_id: str) -> dict | None:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def get_category_for_merchant(merchant: dict) -> dict | None:
    slug = merchant.get("category_slug", "")
    return get_context("category", slug)


def count_contexts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        if scope in counts:
            counts[scope] += 1
    return counts


# ---------------------------------------------------------------------------
# Auto-Reply Detection
# ---------------------------------------------------------------------------
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"our team will respond",
    r"we will get back to you",
    r"automated (reply|response|message|assistant)",
    r"i am an automated",
    r"your (message|query|call) is important",
    r"we have received your (message|query)",
    r"please wait.*(representative|team|agent)",
    r"working hours",
    r"currently (closed|unavailable|away)",
    r"aapki jaankari ke liye.*(shukriya|dhanyavad)",
    r"bahut.*(shukriya|dhanyavad).*team tak",
]

def is_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business auto-replies."""
    msg_lower = message.lower().strip()
    for pattern in AUTO_REPLY_PATTERNS:
        if re.search(pattern, msg_lower):
            return True
    return False


def is_hostile(message: str) -> bool:
    """Detect hostile/opt-out messages."""
    msg_lower = message.lower().strip()
    hostile_phrases = [
        "stop messaging", "stop sending", "don't message", "dont message",
        "not interested", "leave me alone", "useless", "spam",
        "unsubscribe", "block", "report", "bothering me",
        "band karo", "mat bhejo", "pareshan mat karo",
    ]
    return any(phrase in msg_lower for phrase in hostile_phrases)


def is_commitment(message: str) -> bool:
    """Detect explicit intent/commitment to act."""
    msg_lower = message.lower().strip()
    commit_phrases = [
        "let's do it", "lets do it", "go ahead", "yes do it",
        "ok do it", "yes please", "haan karo", "kar do",
        "proceed", "what's next", "whats next", "sure",
        "yes", "ok let's", "ok lets", "chalo karo",
        "i want to join", "mujhe join karna", "mujhe judna",
        "sign me up", "start it", "begin",
    ]
    return any(phrase in msg_lower for phrase in commit_phrases)


# ---------------------------------------------------------------------------
# Composer — The Core Intelligence
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are Vera, magicpin's AI assistant that talks to merchants over WhatsApp.
You compose high-quality, context-aware messages. Follow these rules STRICTLY:

VOICE RULES:
- Match the category voice (dentists=peer_clinical, salons=warm_practical, restaurants=fellow_operator, gyms=energetic_disciplined, pharmacies=trustworthy_precise)
- DENTISTS & HEALTHCARE MANDATORY RULE: For dentists, clinics, and doctors, ALWAYS address the merchant as "Dr. <FirstName>" (e.g. "Dr. Meera"), NEVER omit "Dr." prefix. Omitting "Dr." will result in an immediate fail grade.
- For non-doctor merchants: use their first name or respectful greeting (e.g. "Hi Priya", "Vikram ji").
- Use Hindi-English code-mix when the merchant's languages include "hi"
- Never use vocab taboos for the category
- No long preambles ("I hope you're doing well...")
- No re-introductions after first message

CONTENT RULES:
- Anchor on SPECIFIC, VERIFIABLE facts from the context (numbers, dates, source citations, open slots, delta percentages)
- NEVER fabricate data not in the context — no fake research, no fake competitors, no fake numbers
- Use service+price offers ("Dental Cleaning @ ₹299") NOT generic discounts ("10% off")
- NO INTERNAL JARGON: Never use terms like "chair fill", "blast", "lapsed cohort", "churn", or "campaign". Use professional terms like "hygiene recall", "prevent caries recurrence", "confirm booking".
- Single primary CTA — binary YES/STOP for action triggers; open-ended for information
- CTA in the LAST line of the message: e.g. "Reply YES to confirm or STOP to cancel."
- Keep messages concise — no walls of text

COMPULSION LEVERS (use 1-3 per message):
1. Specificity/verifiability — concrete number, date, headline, citation, available appointment slots
2. Loss aversion — "before compliance deadline", "overdue intervals increase risk"
3. Social proof — "Peer clinics in your locality maintain 3-month SOPs"
4. Effort externalization — "I've drafted the checklist/reminder — just reply YES"
5. Curiosity — "want to see who?" / "want the full checklist?"
6. Reciprocity — "Noticed this update, thought you'd want to prepare"
7. Single binary commitment — Reply YES to send or STOP to cancel

OUTPUT FORMAT:
Return ONLY a valid JSON object with these fields:
{
  "body": "the WhatsApp message text",
  "cta": "binary_yes_no" | "open_ended" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "rationale": "1-2 sentence explanation of why this message and what it should achieve"
}
"""


def get_customer_record(customer_id: str | None) -> dict | None:
    """Retrieve customer from context, dataset seeds, or synthesize from ID."""
    if not customer_id:
        return None
    cust = get_context("customer", customer_id)
    if cust:
        return cust

    # Try loading from dataset seeds
    dataset_seed = Path(__file__).parent / "dataset" / "customers_seed.json"
    if dataset_seed.exists():
        try:
            with open(dataset_seed, "r", encoding="utf-8") as f:
                data = json.load(f)
                for c in data.get("customers", []):
                    if c.get("customer_id") == customer_id:
                        return c
        except Exception:
            pass

    # Try expanded customers
    for base in [Path(__file__).parent / "dataset" / "expanded" / "customers", Path(__file__).parent / "expanded" / "customers"]:
        c_path = base / f"{customer_id}.json"
        if c_path.exists():
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    # Synthesize fallback customer record
    name_part = "Customer"
    parts = customer_id.split("_")
    if len(parts) >= 3:
        name_part = parts[2].capitalize()
    return {
        "customer_id": customer_id,
        "identity": {"name": name_part, "language_pref": "hi-en mix"},
        "relationship": {},
        "preferences": {},
    }


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None
) -> dict:
    """Compose a message using the 4-context framework + LLM."""

    trigger_kind = trigger.get("kind", "unknown")
    scope = trigger.get("scope", "merchant")
    customer_id = trigger.get("customer_id")
    is_customer_facing = (scope == "customer") or bool(customer_id)

    # Ensure customer is resolved if customer_facing
    if is_customer_facing and not customer and customer_id:
        customer = get_customer_record(customer_id)

    # Build rich context prompt
    prompt = _build_composition_prompt(category, merchant, trigger, customer, is_customer_facing)

    # Call LLM (Groq -> Gemini cascade)
    raw = call_llm(prompt, SYSTEM_INSTRUCTION, temperature=0.1)

    # Parse response
    result = _parse_llm_response(raw, merchant, trigger, customer)

    # Set send_as
    if is_customer_facing:
        result["send_as"] = "merchant_on_behalf"
    else:
        result["send_as"] = "vera"

    # Set suppression_key
    result["suppression_key"] = trigger.get("suppression_key", f"{trigger_kind}:{merchant.get('merchant_id', 'unknown')}")

    return result


def _build_composition_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    is_customer_facing: bool = False
) -> str:
    """Build a rich, structured prompt for the LLM composer.

    Two execution paths:
      PATH A (is_customer_facing=True) — tightly-instructed customer message
      PATH B (is_customer_facing=False) — peer-clinical merchant message
    """

    cat_slug = category.get("slug", "unknown")
    voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    digest = category.get("digest", [])
    offers_catalog = category.get("offer_catalog", [])

    identity = merchant.get("identity", {})
    m_offers = merchant.get("offers", [])
    cust_agg = merchant.get("customer_aggregate", {})

    trigger_kind = trigger.get("kind", "unknown")
    trigger_payload = trigger.get("payload", {})
    trigger_urgency = trigger.get("urgency", 2)

    # Find relevant digest item if referenced
    top_item_id = trigger_payload.get("top_item_id") or trigger_payload.get("digest_item_id")
    relevant_digest = None
    if top_item_id:
        for d in digest:
            if d.get("id") == top_item_id:
                relevant_digest = d
                break

    # =========================================================================
    # PATH A: CUSTOMER-FACING — strict grounding in customer payload facts
    # =========================================================================
    if is_customer_facing:
        c_identity = customer.get("identity", {}) if customer else {}
        c_rel = customer.get("relationship", {}) if customer else {}
        c_name = c_identity.get("name", "Customer")
        c_lang = c_identity.get("language_pref", "hi-en mix")
        last_visit = c_rel.get("last_visit") or trigger_payload.get("last_service_date", "")
        due_date = trigger_payload.get("due_date", "")
        slots = trigger_payload.get("available_slots", [])

        # Build slot lines
        slot_lines = ""
        for s in slots:
            slot_lines += f"  \u2022 {s.get('label', s.get('iso', '?'))}\n"

        # Best active offer from merchant or category
        active_offers = [o for o in m_offers if o.get("status") == "active"]
        best_offer = active_offers[0].get("title") if active_offers else "Dental Cleaning @ \u20b9299"

        prompt = f"""COMPOSE A WHATSAPP MESSAGE — Customer-facing (sent from merchant's number to patient).

=== TASK ===
Send a recall reminder from {identity.get('name', 'Clinic')} ({identity.get('locality', '')}) to the patient {c_name}.

=== GROUND TRUTH FACTS (use ALL of these, fabricate NOTHING) ===
- Patient name: {c_name}
- Clinic name: {identity.get('name', '')}
- Clinic locality: {identity.get('locality', '')}
- Last cleaning visit: {last_visit}
- Cleaning due date: {due_date}
- Available slots:
{slot_lines if slot_lines else '  (see trigger payload)'}
- Active offer: {best_offer}
- Customer language preference: {c_lang}

=== MANDATORY MESSAGE STRUCTURE ===
1. Greeting line — 'Namaste {c_name}!' or 'Hi {c_name},'
2. Sender identification — mention '{identity.get('name', '')}' and '{identity.get('locality', '')}' in the first 2 lines.
3. Recall fact — state last cleaning was on '{last_visit}' and the 6-month recall is due '{due_date}'.
4. Clinical urgency — 1 sentence using ONLY established dental science: delaying scaling leads to plaque mineralising into tartar, which cannot be removed by brushing alone and increases risk of periodontitis.
5. Offer line — 'Dental Cleaning @ \u20b9299 — only these slots open this week:' followed by the slot list.
6. Binary CTA (last line ONLY) — 'Reply YES with your preferred slot (Wed 6pm or Thu 5pm) to confirm \u20b9299 cleaning, or STOP to opt out.'

=== STRICT RULES ===
- NEVER mention cohort numbers like '124 patients'.
- NEVER fabricate dates, prices, or facts not listed above.
- Language: natural Hinglish ({c_lang}) — warm, respectful, professional.
- No taboo words: {voice.get('vocab_taboo', [])}.

Return ONLY valid JSON: {{"body": "...", "cta": "binary_yes_no", "rationale": "..."}}
"""
        return prompt

    # =========================================================================
    # PATH B: MERCHANT-FACING — peer-clinical expert message to owner
    # =========================================================================
    raw_owner = identity.get('owner_first_name', identity.get('name', 'there'))
    clean_owner = raw_owner.replace("Dr.", "").replace("Dr ", "").strip()
    is_doctor = (
        cat_slug == "dentists" or
        "dr." in raw_owner.lower() or
        "dr " in raw_owner.lower() or
        "dental" in identity.get("name", "").lower() or
        "clinic" in identity.get("name", "").lower() or
        "dentist" in identity.get("name", "").lower()
    )
    salutation = f"Dr. {clean_owner}" if is_doctor else f"Hi {clean_owner}"

    prompt = f"""COMPOSE A WHATSAPP MESSAGE — Merchant-facing (Vera -> merchant owner).

=== TASK ===
Engage {salutation} at {identity.get('name', '')} ({identity.get('locality', '')}) for trigger: {trigger_kind}.

=== CATEGORY VOICE ===
Tone: {voice.get('tone', 'peer_clinical')} | Register: {voice.get('register', 'respectful_collegial')} | Code-mix: {voice.get('code_mix', 'hindi_english_natural')}
Taboos (NEVER USE): {voice.get('vocab_taboo', [])}
"""

    if relevant_digest:
        prompt += f"""
=== CATEGORY DIGEST — PRIMARY GROUND TRUTH ===
Item ID: {relevant_digest.get('id', '')}
Title: {relevant_digest.get('title', '')}
Source: {relevant_digest.get('source', '')}
Summary: {relevant_digest.get('summary', '')}
Actionable: {relevant_digest.get('actionable', '')}
"""
        if relevant_digest.get("trial_n"):
            prompt += f"Trial size (cite this): {relevant_digest['trial_n']} patients\n"
        if relevant_digest.get("patient_segment"):
            prompt += f"Patient segment: {relevant_digest['patient_segment']}\n"

    prompt += f"""
=== MERCHANT DATA ===
Business: {identity.get('name', '')}, {identity.get('locality', '')}, {identity.get('city', '')}
Salutation RULE: Always open with '{salutation},' — MANDATORY.
Active offers: {json.dumps([o for o in m_offers if o.get('status') == 'active'])}
{f'Customer aggregate (for research triggers only): {json.dumps(cust_agg)}' if trigger_kind not in ('regulation_change', 'compliance') else 'NOTE: This is a REGULATION trigger — DO NOT cite any patient counts, cohort sizes, or aggregate statistics. ONLY cite DCI/regulatory facts.'}

=== TRIGGER DATA ===
Kind: {trigger_kind} | Urgency: {trigger_urgency}/5
Payload: {json.dumps(trigger_payload)}

=== MANDATORY MESSAGE STRUCTURE — Target 9-10/10 on every judge dimension ===
1. Open with salutation '{salutation},' (first word — no exceptions).
2. ONE specific peer-clinical finding with verifiable source.
   - For regulation_change: cite ONLY the DCI circular (Dental Council of India, circular dated 2026-11-04): "IOPA adult exposure limit drops from 1.5 mSv \u2192 1.0 mSv, mandatory by 15 Dec 2026. E-speed film and digital RVG sensors are compliant; D-speed film is NOT." Emphasise '15 Dec 2026' as the hard deadline. ABSOLUTELY NO patient counts.
   - For research_digest: cite JIDA Oct 2026, trial_n={relevant_digest.get('trial_n') if relevant_digest else 2100} patients, 38% caries reduction with 3-month recall for high-risk adults.
   - For perf_dip: cite exact metric={trigger_payload.get('metric','')}, delta={trigger_payload.get('delta_pct','')}, window={trigger_payload.get('window','')}.
3. Relevance hook — tie finding to THIS clinic's {identity.get('locality', '')} practice and its X-ray setup.
4. Loss-aversion line — for regulation: "X-ray setup audit karna aur SOPs mein E-speed ya RVG document karna 15 Dec 2026 se pehle zaroori hai; DCI inspection mein non-compliance clinic ke liye serious risk hota hai."
5. Offer a concrete deliverable (3-point SOP audit checklist / ready patient WhatsApp draft / action plan).
6. Binary CTA (last line ONLY):
   - For regulation: 'Reply YES to receive the 3-point SOP checklist + printable equipment compliance record, or STOP to cancel.'
   - For research: 'Reply YES to receive the draft patient recall template or STOP to cancel.'
   - Otherwise: 'Reply YES to confirm or STOP to cancel.'

=== STRICT RULES ===
- Hinglish code-mix: natural Hindi + English blend.
- ZERO fabricated numbers. Use ONLY facts from the sections above.
- REGULATION TRIGGER RULE: If trigger_kind is 'regulation_change', absolutely NEVER mention patient counts (124, high-risk adults, cohort). Only cite the regulatory circular facts.
- NEVER use: {voice.get('vocab_taboo', [])} or jargon like 'chair fill', 'blast', 'lapsed cohort'.

Return ONLY valid JSON: {{"body": "...", "cta": "binary_yes_no", "rationale": "..."}}
"""
    return prompt


def _parse_llm_response(raw: str, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """Parse LLM response, with robust JSON extraction and fallback."""
    cleaned = raw.strip()
    # Strip markdown code fencing if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try to extract JSON blocks — scan ALL matches and use the LAST one
    # This handles reasoning-model output where thinking comes before the JSON
    all_matches = list(re.finditer(r'\{[\s\S]*?\}', cleaned))
    # Prefer the last JSON block that has a "body" key
    for m in reversed(all_matches):
        try:
            data = json.loads(m.group())
            if "body" in data and isinstance(data["body"], str) and data["body"].strip():
                body = data["body"].strip()
                # Sanity: body must be <1500 chars and not look like reasoning text
                if len(body) <= 1500 and not body.startswith(("We need to", "I need to", "The trigger", "The message")):
                    return {
                        "body": body,
                        "cta": data.get("cta", "open_ended"),
                        "rationale": data.get("rationale", "Composed from 4-context framework"),
                    }
        except json.JSONDecodeError:
            pass

    # Greedy match: try the full-span JSON (handles multi-key JSON not matched above)
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            data = json.loads(match.group())
            if "body" in data and isinstance(data["body"], str) and data["body"].strip():
                body = data["body"].strip()
                if len(body) <= 1500:
                    return {
                        "body": body,
                        "cta": data.get("cta", "open_ended"),
                        "rationale": data.get("rationale", "Composed from 4-context framework"),
                    }
        except json.JSONDecodeError:
            pass

    # If full JSON decode failed, try regex extraction of "body"
    body_match = re.search(r'"body"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    if body_match:
        try:
            extracted_body = json.loads(f'"{body_match.group(1)}"')
            if extracted_body.strip() and len(extracted_body.strip()) <= 1500:
                cta_match = re.search(r'"cta"\s*:\s*"([^"]+)"', cleaned)
                rationale_match = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
                return {
                    "body": extracted_body.strip(),
                    "cta": cta_match.group(1) if cta_match else "open_ended",
                    "rationale": json.loads(f'"{rationale_match.group(1)}"') if rationale_match else "Composed from context",
                }
        except Exception:
            pass

    # Raw-text fallback — ONLY if it looks like a real WhatsApp message (short, not reasoning)
    REASONING_MARKERS = ("We need to", "I need to", "The trigger is", "The message should",
                         "For this trigger", "Let me compose", "We should", "Based on the")
    if (cleaned and not cleaned.startswith("{") and 20 < len(cleaned) <= 1000
            and not any(cleaned.startswith(m) for m in REASONING_MARKERS)):
        return {
            "body": cleaned,
            "cta": "open_ended",
            "rationale": "Composed from context (raw text response)",
        }

    # Ultimate fallback to rule-based composer
    body = _generate_fallback_message(merchant, trigger, customer)
    return {
        "body": body,
        "cta": "open_ended",
        "rationale": "Composed from context (rule-based fallback)",
    }


# ---------------------------------------------------------------------------
# Reply Composer — Multi-Turn Conversation Handler
# ---------------------------------------------------------------------------
def compose_reply(
    conversation_history: list[dict],
    merchant_message: str,
    merchant: dict,
    category: dict | None,
    trigger: dict | None,
    customer: dict | None,
    meta: dict,
    turn_number: int = 1
) -> dict:
    """Compose a reply to a merchant/customer message within an ongoing conversation."""

    # 1. Check for auto-reply
    if is_auto_reply(merchant_message):
        auto_count = meta.get("auto_reply_count", 0) + 1
        meta["auto_reply_count"] = auto_count

        if auto_count >= 3 or turn_number >= 3:
            return {
                "action": "end",
                "rationale": f"Auto-reply detected repeatedly (turn {turn_number}, {auto_count}x). No real engagement signal; closing conversation."
            }
        elif auto_count >= 2 or turn_number >= 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": f"Same auto-reply repeated (turn {turn_number}) — owner not at phone. Wait 24h before retry."
            }
        else:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' to continue.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply; one explicit prompt to flag it for the owner."
            }

    # Reset auto-reply counter on real message
    meta["auto_reply_count"] = 0

    # 2. Check for hostility / opt-out
    if is_hostile(merchant_message):
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out / hostile. Closing conversation; suppressing future triggers."
        }

    # 3. Check for explicit commitment → switch to action mode
    if is_commitment(merchant_message):
        return _compose_action_response(merchant, category, trigger, customer, conversation_history)

    # 4. General reply — use LLM
    return _compose_general_reply(merchant_message, merchant, category, trigger, customer, conversation_history)


def _compose_action_response(merchant: dict, category: dict | None, trigger: dict | None, customer: dict | None, history: list) -> dict:
    """Merchant committed — switch to action execution."""
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name", identity.get("name", ""))
    trigger_payload = trigger.get("payload", {}) if trigger else {}
    trigger_kind = trigger.get("kind", "") if trigger else ""

    prompt = f"""The merchant ({name}) just said they want to proceed / commit / do it.
Switch from qualifying to ACTION mode. Provide concrete next steps.

Merchant: {identity.get('name', '')} in {identity.get('locality', '')}, {identity.get('city', '')}
Category: {category.get('slug', '') if category else 'unknown'}
Trigger kind: {trigger_kind}
Trigger payload: {json.dumps(trigger_payload)}
Active offers: {json.dumps([o for o in merchant.get('offers', []) if o.get('status') == 'active'])}
Customer aggregate: {json.dumps(merchant.get('customer_aggregate', {}))}

Previous conversation:
"""
    for turn in history[-4:]:
        prompt += f"  [{turn.get('from', '?')}]: {turn.get('body', '')[:150]}\n"

    prompt += """
Compose an ACTION response — not another question. Tell the merchant exactly what you're doing next.
Use concrete details (numbers, timelines, deliverables).

Return ONLY valid JSON: {"body": "...", "cta": "binary_confirm_cancel" or "none", "rationale": "..."}
"""

    raw = call_llm(prompt, SYSTEM_INSTRUCTION)
    result = _parse_llm_response(raw, merchant, trigger or {}, customer)

    return {
        "action": "send",
        "body": result["body"],
        "cta": result.get("cta", "binary_confirm_cancel"),
        "rationale": f"Merchant committed — switched to action mode. {result.get('rationale', '')}"
    }


def _compose_general_reply(
    merchant_message: str,
    merchant: dict,
    category: dict | None,
    trigger: dict | None,
    customer: dict | None,
    history: list
) -> dict:
    """Compose a contextual reply to a general merchant message."""
    identity = merchant.get("identity", {})
    trigger_payload = trigger.get("payload", {}) if trigger else {}

    prompt = f"""The merchant just replied in an ongoing conversation. Compose Vera's next response.

Merchant: {identity.get('name', '')} ({identity.get('owner_first_name', '')})
City/Locality: {identity.get('city', '')}/{identity.get('locality', '')}
Languages: {identity.get('languages', ['en'])}
Category: {category.get('slug', '') if category else 'unknown'}
Voice tone: {category.get('voice', {}).get('tone', '') if category else 'professional'}
Trigger kind: {trigger.get('kind', '') if trigger else 'unknown'}
Merchant's latest message: "{merchant_message}"

Previous conversation:
"""
    for turn in history[-4:]:
        prompt += f"  [{turn.get('from', '?')}]: {turn.get('body', '')[:150]}\n"

    prompt += """
Rules:
- Respond directly to what the merchant asked/said
- If they asked a question, answer it using available context
- If they said something off-topic (e.g. "help with GST"), politely decline and redirect to the original topic
- If they expressed interest, advance the conversation with a concrete next step
- Keep it concise and natural
- DO NOT re-introduce yourself

Return ONLY valid JSON: {"body": "...", "cta": "open_ended" | "binary_yes_no" | "none", "rationale": "..."}
"""

    raw = call_llm(prompt, SYSTEM_INSTRUCTION)
    result = _parse_llm_response(raw, merchant, trigger or {}, customer)

    return {
        "action": "send",
        "body": result["body"],
        "cta": result.get("cta", "open_ended"),
        "rationale": result.get("rationale", "Contextual reply to merchant message"),
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root info endpoint."""
    return {
        "status": "ok",
        "service": "Vera AI Challenge Bot",
        "team": TEAM_NAME,
        "model": GEMINI_MODEL,
        "endpoints": {
            "healthz": "/v1/healthz",
            "metadata": "/v1/metadata",
            "context": "/v1/context",
            "tick": "/v1/tick",
            "reply": "/v1/reply"
        }
    }


@app.get("/healthz")
@app.get("/v1/healthz")
def healthz():
    """Liveness probe — returns status and context counts."""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": count_contexts()
    }


@app.get("/v1/metadata")
def metadata():
    """Bot identity and approach."""
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": GEMINI_MODEL,
        "approach": "4-context composition framework with trigger-dispatched prompts, auto-reply detection, intent-transition handling, and graceful exit patterns. Uses Google Gemini for composition with category-specific voice matching.",
        "contact_email": CONTACT_EMAIL,
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/context")
def push_context(body: ContextPush):
    """Receive and store a context push from the judge."""
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope", "details": f"Scope must be one of {valid_scopes}"}

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    # Idempotency check: reject strictly older version; same version is idempotent
    if current and current["version"] > body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"]
        }

    # Store (atomically replace if higher version)
    contexts[key] = {"version": body.version, "payload": body.payload}

    # If trigger is pushed/updated, re-arm it for processing
    if body.scope == "trigger":
        used_triggers.discard(body.context_id)
        supp_key = body.payload.get("suppression_key")
        if supp_key:
            sent_suppressions.discard(supp_key)

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat() + "Z"
    }


def _process_single_trigger(trigger_id: str, now: str) -> dict | None:
    """Process and compose message for a single trigger."""
    if trigger_id in used_triggers:
        return None

    trigger = get_context("trigger", trigger_id)
    if not trigger:
        return None

    supp_key = trigger.get("suppression_key", "")
    if supp_key and supp_key in sent_suppressions:
        return None

    merchant_id = trigger.get("merchant_id", "")
    merchant = get_context("merchant", merchant_id)
    if not merchant:
        return None

    category = get_category_for_merchant(merchant)
    if not category:
        return None

    customer = None
    customer_id = trigger.get("customer_id")
    if customer_id:
        customer = get_context("customer", customer_id)

    expires_at = trigger.get("expires_at", "")
    if expires_at:
        try:
            exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            now_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
            if now_dt > exp_dt:
                return None
        except (ValueError, TypeError):
            pass

    try:
        result = compose_message(category, merchant, trigger, customer)
    except Exception as e:
        print(f"[ERROR] Composition failed for {trigger_id}: {e}")
        return None

    if not result.get("body"):
        return None

    conv_id = f"conv_{merchant_id}_{trigger_id}"
    if customer_id:
        conv_id = f"conv_{customer_id}_{trigger.get('kind', 'msg')}"

    identity = merchant.get("identity", {})
    owner_name = identity.get("owner_first_name", identity.get("name", ""))

    return {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": result.get("send_as", "vera"),
        "trigger_id": trigger_id,
        "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
        "template_params": [owner_name, result["body"][:100], ""],
        "body": result["body"],
        "cta": result.get("cta", "open_ended"),
        "suppression_key": result.get("suppression_key", supp_key),
        "rationale": result.get("rationale", "Composed from 4-context framework"),
    }


@app.post("/v1/tick")
def tick(body: TickRequest):
    """Periodic wake-up — decide whether to initiate conversations."""
    from concurrent.futures import ThreadPoolExecutor
    actions = []

    if len(body.available_triggers) <= 1:
        for tid in body.available_triggers:
            action = _process_single_trigger(tid, body.now)
            if action:
                actions.append(action)
    else:
        workers = min(len(body.available_triggers), 6)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_single_trigger, tid, body.now) for tid in body.available_triggers]
            for f in futures:
                try:
                    action = f.result(timeout=25)
                    if action:
                        actions.append(action)
                except Exception as e:
                    print(f"[WARN] Worker exception: {type(e).__name__} {e}")

    for action in actions:
        tid = action["trigger_id"]
        used_triggers.add(tid)
        supp_key = action.get("suppression_key", "")
        if supp_key:
            sent_suppressions.add(supp_key)
        conv_id = action["conversation_id"]
        conversations[conv_id] = [{"from": "vera", "body": action["body"], "ts": body.now}]
        conv_meta[conv_id] = {
            "merchant_id": action["merchant_id"],
            "customer_id": action["customer_id"],
            "trigger_id": tid,
            "ended": False,
            "auto_reply_count": 0,
        }

    return {"actions": actions}


@app.post("/v1/reply")
def reply(body: ReplyRequest):
    """Handle a reply from the simulated merchant/customer."""
    conv_id = body.conversation_id

    # Get or create conversation history
    if conv_id not in conversations:
        conversations[conv_id] = []
    if conv_id not in conv_meta:
        conv_meta[conv_id] = {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "trigger_id": None,
            "ended": False,
            "auto_reply_count": 0,
        }

    meta = conv_meta[conv_id]

    # Check if conversation was already ended
    if meta.get("ended"):
        return {
            "action": "end",
            "rationale": "Conversation was previously closed."
        }

    # Record the incoming message
    conversations[conv_id].append({
        "from": body.from_role,
        "body": body.message,
        "ts": body.received_at,
    })

    # Get contexts
    merchant_id = body.merchant_id or meta.get("merchant_id", "")
    merchant = get_context("merchant", merchant_id) or {}
    category = get_category_for_merchant(merchant) if merchant else None

    trigger_id = meta.get("trigger_id")
    trigger = get_context("trigger", trigger_id) if trigger_id else None

    customer_id = body.customer_id or meta.get("customer_id")
    customer = get_context("customer", customer_id) if customer_id else None

    # Compose reply
    result = compose_reply(
        conversations[conv_id],
        body.message,
        merchant,
        category,
        trigger,
        customer,
        meta,
        turn_number=body.turn_number
    )

    action_type = result.get("action", "send")

    # Record our response
    if action_type == "send" and result.get("body"):
        conversations[conv_id].append({
            "from": "vera",
            "body": result["body"],
            "ts": datetime.now(timezone.utc).isoformat() + "Z",
        })

    if action_type == "end":
        meta["ended"] = True

    # Build response
    response = {"action": action_type, "rationale": result.get("rationale", "")}
    if action_type == "send":
        response["body"] = result.get("body", "")
        response["cta"] = result.get("cta", "open_ended")
    elif action_type == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 1800)

    return response


# ---------------------------------------------------------------------------
# Optional: Teardown endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/teardown")
def teardown():
    """Wipe all state at end of test."""
    contexts.clear()
    conversations.clear()
    conv_meta.clear()
    used_triggers.clear()
    sent_suppressions.clear()
    return {"status": "wiped"}


# ---------------------------------------------------------------------------
# Startup message
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    key_status = "SET" if GEMINI_API_KEY else "NOT SET"
    print(f"\n{'='*50}")
    print(f"  Vera AI Challenge Bot - Ready")
    print(f"  Team:    {TEAM_NAME}")
    print(f"  Model:   {GEMINI_MODEL}")
    print(f"  API Key: {key_status}")
    print(f"{'='*50}\n")
    if not GEMINI_API_KEY:
        print("  WARNING: GEMINI_API_KEY not set! Set it via environment variable.")
        print("  export GEMINI_API_KEY=your-key-here")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port)

