import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent))

from bot import _build_composition_prompt, call_llm, SYSTEM_INSTRUCTION, _parse_llm_response
from judge_simulator import DatasetLoader, LLMScorer, create_provider

dataset = DatasetLoader(Path(__file__).parent / "dataset")
dataset.load()

provider = create_provider()
scorer = LLMScorer(provider, dataset)

cat_dentists = dataset.categories["dentists"]
m_001 = dataset.merchants["m_001_drmeera_dentist_delhi"]
trg_002 = dataset.triggers["trg_002_compliance_dci_radiograph"]
trg_003 = dataset.triggers["trg_003_recall_due_priya"]
cust_priya = dataset.customers["c_001_priya_for_m001"]

print("--- Testing Message 1 (Regulation) Generation ---")
prompt1 = _build_composition_prompt(cat_dentists, m_001, trg_002, None, is_customer_facing=False)
raw1 = call_llm(prompt1, SYSTEM_INSTRUCTION, temperature=0.1)
res1 = _parse_llm_response(raw1, m_001, trg_002, None)
action1 = {
    "body": res1["body"],
    "cta": res1["cta"],
    "send_as": "vera",
    "trigger_id": "trg_002_compliance_dci_radiograph",
    "merchant_id": "m_001_drmeera_dentist_delhi"
}
print(f"Generated Body:\n{res1['body']}\n")
score1 = scorer.score(action1, cat_dentists, m_001, trg_002, None)
print(f"Score 1: {score1.total}/50")
print(f"Specificity: {score1.specificity}, Category: {score1.category_fit}, Merchant: {score1.merchant_fit}, Decision: {score1.decision_quality}, Engagement: {score1.engagement_compulsion}")
print(f"Hint: {score1.hint}\n")

print("--- Testing Message 2 (Customer Priya Recall) Generation ---")
prompt2 = _build_composition_prompt(cat_dentists, m_001, trg_003, cust_priya, is_customer_facing=True)
raw2 = call_llm(prompt2, SYSTEM_INSTRUCTION, temperature=0.1)
res2 = _parse_llm_response(raw2, m_001, trg_003, cust_priya)
action2 = {
    "body": res2["body"],
    "cta": res2["cta"],
    "send_as": "merchant_on_behalf",
    "trigger_id": "trg_003_recall_due_priya",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "customer_id": "c_001_priya_for_m001"
}
print(f"Generated Body:\n{res2['body']}\n")
score2 = scorer.score(action2, cat_dentists, m_001, trg_003, cust_priya)
print(f"Score 2: {score2.total}/50")
print(f"Specificity: {score2.specificity}, Category: {score2.category_fit}, Merchant: {score2.merchant_fit}, Decision: {score2.decision_quality}, Engagement: {score2.engagement_compulsion}")
print(f"Hint: {score2.hint}")
