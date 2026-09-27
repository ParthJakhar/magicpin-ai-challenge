"""
conversation_handlers.py — Multi-turn conversation handler for Vera.

This module demonstrates advanced multi-turn handling including:
- Auto-reply detection with escalating back-off strategy
- Intent transition detection (qualifying → action mode)
- Graceful exit on hostility or opt-out
- Language switching detection mid-conversation
- Off-topic redirection
- Conversation cadence planning within 24h session window
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    turns: list = field(default_factory=list)
    auto_reply_count: int = 0
    phase: Literal["qualifying", "action", "follow_up", "closing"] = "qualifying"
    merchant_sentiment: Literal["unknown", "engaged", "neutral", "resistant", "hostile"] = "unknown"
    ended: bool = False
    language_detected: Optional[str] = None
    topics_covered: list = field(default_factory=list)


def respond(state: ConversationState, merchant_message: str) -> dict:
    """
    Given the conversation so far + the merchant's latest message, produce the reply.

    This is the optional multi-turn handler that demonstrates advanced conversation
    flow management.

    Returns:
        dict with keys: action, body (if send), cta, rationale, wait_seconds (if wait)
    """
    from bot import (
        is_auto_reply, is_hostile, is_commitment,
        compose_reply, get_context, get_category_for_merchant,
        conversations, conv_meta
    )

    # Record the turn
    state.turns.append({"from": "merchant", "body": merchant_message})

    # Get contexts
    merchant = get_context("merchant", state.merchant_id) or {}
    category = get_category_for_merchant(merchant) if merchant else None
    trigger = get_context("trigger", state.trigger_id) if state.trigger_id else None
    customer = get_context("customer", state.customer_id) if state.customer_id else None

    # ---- Phase 1: Auto-reply detection ----
    if is_auto_reply(merchant_message):
        state.auto_reply_count += 1

        if state.auto_reply_count >= 3:
            state.ended = True
            state.phase = "closing"
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {state.auto_reply_count}x. No engagement; closing."
            }
        elif state.auto_reply_count >= 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": f"Auto-reply {state.auto_reply_count}x — owner unavailable. 24h back-off."
            }
        else:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' to continue.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply; one explicit prompt for the owner."
            }

    # Reset on real message
    state.auto_reply_count = 0

    # ---- Phase 2: Hostility check ----
    if is_hostile(merchant_message):
        state.ended = True
        state.merchant_sentiment = "hostile"
        state.phase = "closing"

        return {
            "action": "send",
            "body": "Apologies — I won't message again. If anything changes, you can always restart with 'Hi Vera'. 🙏",
            "cta": "none",
            "rationale": "Merchant frustration detected; one-line apology + exit."
        }

    # ---- Phase 3: Intent transition ----
    if is_commitment(merchant_message) and state.phase == "qualifying":
        state.phase = "action"
        state.merchant_sentiment = "engaged"

        # Use the bot's action composer
        meta = conv_meta.get(state.conversation_id, {
            "merchant_id": state.merchant_id,
            "customer_id": state.customer_id,
            "trigger_id": state.trigger_id,
            "ended": False,
            "auto_reply_count": 0,
        })

        result = compose_reply(
            state.turns, merchant_message,
            merchant, category, trigger, customer, meta
        )
        return result

    # ---- Phase 4: Detect language ----
    _detect_language(state, merchant_message)

    # ---- Phase 5: General reply ----
    state.merchant_sentiment = "engaged" if len(merchant_message) > 20 else "neutral"

    meta = conv_meta.get(state.conversation_id, {
        "merchant_id": state.merchant_id,
        "customer_id": state.customer_id,
        "trigger_id": state.trigger_id,
        "ended": False,
        "auto_reply_count": 0,
    })

    result = compose_reply(
        state.turns, merchant_message,
        merchant, category, trigger, customer, meta
    )

    # Record our turn
    if result.get("action") == "send" and result.get("body"):
        state.turns.append({"from": "vera", "body": result["body"]})

    return result


def _detect_language(state: ConversationState, message: str):
    """Detect language from the merchant's message for code-mix alignment."""
    hindi_indicators = ["hai", "karo", "kya", "haan", "nahi", "acha", "theek",
                        "bhai", "ji", "hoon", "chahiye", "karenge", "batao"]
    msg_lower = message.lower()

    hindi_count = sum(1 for w in hindi_indicators if w in msg_lower.split())
    total_words = len(msg_lower.split())

    if total_words > 0:
        ratio = hindi_count / total_words
        if ratio > 0.3:
            state.language_detected = "hi"
        elif ratio > 0.1:
            state.language_detected = "hi-en mix"
        else:
            state.language_detected = "en"
