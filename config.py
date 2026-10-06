
"""Application configuration and defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

APP_TITLE = "Banking AML, Fraud & Risk Analytics"
APP_ICON = "🏦"
DEFAULT_RISK_LEVELS = {
    "Low": (0, 25),
    "Medium": (25, 50),
    "High": (50, 75),
    "Critical": (75, 100),
}

CANONICAL_FIELDS = {
    "bank_account": "Bank Account",
    "transaction_date": "Entry Date",
    "amount": "Def Amount",
    "debit_account": "Debit Id",
    "credit_account": "Credit Id",
    "customer_id": "Doc Customer",
    "contract_customer": "Contract Customer",
    "document_number": "Document Num",
    "currency": "Currency Id",
    "branch": "Branch",
    "note": "Note",
    "doc_type": "Doc Type",
    "doc_state": "Doc State",
    "cash_flag": "Doc Cash",
    "user": "User",
    "debit_name": "Debit Acc Name",
    "credit_name": "Credit Acc Name",
}
CANONICAL_FIELDS_2 = {
    "doc_type": "Տիպ",
    "doc_state": "Կարգավիճակ",
    "transaction_date": "Մուտքի ամս.",
    "amount": "Գումար(ըստ համակարգի)",
    "currency": "Արժույթ",
    "customer_id": "Վճարող",
    "note": "Նշումներ",
    "cash_register": "Դրամարկղ",
    "debit_account": "Դեբետ հաշիվ",
    "credit_account": "Կրեդիտ հաշիվ",
    "branch": "Մասնաճյուղ",
    "cash_flag": "Փաստ. համար",
    "user": "Օգտագործող",
}

DEFAULT_RULES: Dict[str, Dict[str, Any]] = {
    "large_amount": {"enabled": True, "weight": 20, "threshold": 25000000, "description": "Transaction amount exceeds threshold."},
    "round_amount": {"enabled": True, "weight": 10, "threshold": 100000, "description": "Amount is a large round number."},
    "cash_transaction": {"enabled": True, "weight": 14, "threshold": None, "description": "Cash-related transaction."},
    "night_activity": {"enabled": True, "weight": 10, "threshold": None, "description": "Transaction occurred at night."},
    "weekend_activity": {"enabled": True, "weight": 5, "threshold": None, "description": "Transaction occurred on weekend."},
    "high_frequency_sender": {"enabled": True, "weight": 14, "threshold": 20, "description": "Sender has unusually high transaction count."},
    "many_counterparties": {"enabled": True, "weight": 15, "threshold": 15, "description": "Sender interacts with many counterparties."},
    "multiple_currencies": {"enabled": True, "weight": 12, "threshold": 2, "description": "Customer/account uses multiple currencies."},
}

@dataclass
class AppState:
    """Container for runtime mapping and configuration."""
    column_mapping: Dict[str, str] = field(default_factory=dict)
    risk_rules: Dict[str, Dict[str, Any]] = field(default_factory=lambda: DEFAULT_RULES.copy())
