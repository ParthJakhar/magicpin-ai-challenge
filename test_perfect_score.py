import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent))

from judge_simulator import DatasetLoader, LLMScorer, create_provider

dataset = DatasetLoader(Path(__file__).parent / "dataset")
if not dataset.load():
    print("Dataset failed")
    sys.exit(1)

provider = create_provider()
scorer = LLMScorer(provider, dataset)

# Test Message 1: trg_002 (Regulation change - DCI)
trg_002 = dataset.triggers["trg_002_compliance_dci_radiograph"]
m_001 = dataset.merchants["m_001_drmeera_dentist_delhi"]
cat_dentists = dataset.categories["dentists"]

msg1_body = (
    "Dr. Meera, Dental Council of India circular ke mutabiq 15 Dec 2026 se adult IOPA exposure dose limit "
    "1.5 mSv se ghatakar 1.0 mSv kar di gayi hai. E-speed film aur digital RVG sensors pass hote hain, par "
    "D-speed film disqualify ho jayegi. Dr. Meera's Dental Clinic (Lajpat Nagar) ke X-ray setup ko Dec 15 se "
    "pehle audit karke SOPs mein E-speed ya RVG document karna zaroori hai, taaki DCI inspection penalty "
    "ka risk na rahe. Maine aapke clinic ke liye 3-point X-ray SOP audit checklist draft kar di hai.\n\n"
    "Reply YES to receive the SOP checklist or STOP to cancel."
)

action1 = {
    "body": msg1_body,
    "cta": "binary_yes_no",
    "send_as": "vera",
    "trigger_id": "trg_002_compliance_dci_radiograph",
    "merchant_id": "m_001_drmeera_dentist_delhi"
}

print("=== SCORING MESSAGE 1 (DCI Regulation) ===")
score1 = scorer.score(action1, cat_dentists, m_001, trg_002, None)
print(f"Specificity: {score1.specificity}/10 - {score1.specificity_reason}")
print(f"Category Fit: {score1.category_fit}/10 - {score1.category_fit_reason}")
print(f"Merchant Fit: {score1.merchant_fit}/10 - {score1.merchant_fit_reason}")
print(f"Decision Quality: {score1.decision_quality}/10 - {score1.decision_quality_reason}")
print(f"Engagement: {score1.engagement_compulsion}/10 - {score1.engagement_reason}")
print(f"Total: {score1.total}/50")
print(f"Hint: {score1.hint}")

# Test Message 2: trg_003 (Recall due - Priya)
trg_003 = dataset.triggers["trg_003_recall_due_priya"]
cust_priya = dataset.customers["c_001_priya_for_m001"]

msg2_body = (
    "Namaste Priya! Dr. Meera's Dental Clinic, Lajpat Nagar se recall reminder hai. "
    "Aapka last cleaning visit 12 May 2026 ko hua tha aur 6-month preventive cleaning 12 Nov 2026 ko due hai. "
    "Routine scaling delay karne se plaque buildup aur enamel demineralization ka risk rehta hai. "
    "Humare active clinic offer 'Dental Cleaning @ ₹299' ke sath limited slots hold kiye gaye hain:\n"
    "• Wed 5 Nov at 6:00 PM\n"
    "• Thu 6 Nov at 5:00 PM\n\n"
    "Reply YES with your preferred slot (Wed 6pm or Thu 5pm) to confirm ₹299 cleaning, or STOP to opt out."
)

action2 = {
    "body": msg2_body,
    "cta": "binary_yes_no",
    "send_as": "merchant_on_behalf",
    "trigger_id": "trg_003_recall_due_priya",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "customer_id": "c_001_priya_for_m001"
}

print("\n=== SCORING MESSAGE 2 (Customer Priya Recall) ===")
score2 = scorer.score(action2, cat_dentists, m_001, trg_003, cust_priya)
print(f"Specificity: {score2.specificity}/10 - {score2.specificity_reason}")
print(f"Category Fit: {score2.category_fit}/10 - {score2.category_fit_reason}")
print(f"Merchant Fit: {score2.merchant_fit}/10 - {score2.merchant_fit_reason}")
print(f"Decision Quality: {score2.decision_quality}/10 - {score2.decision_quality_reason}")
print(f"Engagement: {score2.engagement_compulsion}/10 - {score2.engagement_reason}")
print(f"Total: {score2.total}/50")
print(f"Hint: {score2.hint}")
