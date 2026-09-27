# Vera AI Challenge — Submission

## Team

**Team Name:** Vera Reborn  
**Member:** Parth Jakhar (jakharparth12@gmail.com)  
**Model:** Groq (`openai/gpt-oss-120b` / `qwen/qwen3.8-27b`) with automatic cascade to Google Gemini Flash (`gemini-3.5-flash` / `gemini-3.6-flash` / `gemini-2.5-flash`)  

## Approach

### Architecture

The bot is a **FastAPI HTTP server** implementing all 5 required endpoints (`/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`). It uses the **4-context composition framework** defined in the challenge brief:

```
compose(CategoryContext, MerchantContext, TriggerContext, CustomerContext?) → ComposedMessage
```

### Composition Strategy

**Trigger-dispatched prompting**: Instead of a single generic prompt, the composer builds a rich, structured prompt that includes:

1. **Category context** — voice rules, peer stats, relevant digest items, seasonal beats, trends
2. **Merchant context** — identity, performance, offers, signals, conversation history, customer aggregate
3. **Trigger context** — kind, payload, urgency, suppression key
4. **Customer context** (when present) — identity, relationship, preferences, consent scope

The prompt references specific data from each context layer, ensuring the LLM grounds its output in verifiable facts rather than hallucinating.

**Digest item resolution**: When a trigger references a digest item (e.g., `top_item_id: "d_2026W17_jida_fluoride"`), the composer looks up the full digest entry from the CategoryContext and includes the title, source, trial size, patient segment, and actionable recommendation in the prompt.

### Multi-Turn Handling

The bot maintains full conversation state per `conversation_id`:

1. **Auto-reply detection** — Pattern matching against 12+ common WhatsApp Business auto-reply phrases (English + Hindi). Escalating strategy: first auto-reply → one nudge, second → 24h wait, third → end.
2. **Intent transition** — When the merchant signals commitment ("let's do it", "go ahead", "haan karo"), the bot switches from qualifying mode to action mode immediately.
3. **Hostility handling** — Detects opt-out/hostile messages and exits gracefully with a single polite message.
4. **Off-topic redirection** — Uses LLM to politely decline out-of-scope asks and redirect to the original trigger topic.

### Context Versioning

The `/v1/context` endpoint is idempotent on `(scope, context_id, version)`. Higher versions atomically replace prior versions. The bot uses the latest version whenever composing.

### Suppression

Trigger suppression keys are tracked to prevent duplicate sends. Once a message is sent for a suppression key, no further messages are composed for the same key.

## Tradeoffs

1. **LLM latency vs quality**: We chose Gemini Flash for the best balance of speed (<5s per composition) and quality. This keeps well within the 30s timeout.
2. **In-memory state**: All context is stored in-memory (Python dicts). Production would use Redis, but for the test window this is reliable and fast.
3. **Temperature 0.2**: Low temperature for consistency across runs, with just enough variation to avoid repetition.
4. **Prompt length**: We include substantial context in the prompt (~2KB per composition) to give the LLM the best chance at specificity. This costs tokens but ensures grounding.

## What Additional Context Would Have Helped

1. **Real merchant conversation transcripts** — More examples of how merchants actually reply (in different languages, with different intents) would help tune the reply composer.
2. **Offer catalog availability by time** — Real-time slot availability for customer-facing booking messages.
3. **Merchant response rate history** — Knowing which merchants are likely to respond would help prioritize trigger processing.
4. **A/B test results** from production Vera — What message patterns have historically driven the highest reply rates per category.

## Files

| File | Purpose |
|---|---|
| `bot.py` | Main bot — FastAPI server with all 5 endpoints + LLM composer |
| `conversation_handlers.py` | Optional multi-turn handler demonstrating advanced flow management |
| `generate_submission.py` | Script to generate `submission.jsonl` from the test pairs |
| `README.md` | This file |

## Configuration & Running

Configuration is loaded from environment variables or `.env` file (see `.env.example`).

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 2. Start the bot server
uvicorn bot:app --host 0.0.0.0 --port 8080

# 3. Test with integration test suite
python test_bot.py

# 4. Test with the judge simulator
python judge_simulator.py

# 5. Generate submission.jsonl
python generate_submission.py
```
