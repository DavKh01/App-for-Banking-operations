
"""Application configuration and defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

APP_TITLE = "Banking AML, Fraud & Risk Analytics"
APP_ICON = "🏦"
DEFAULT_RISK_LEVELS = {
    "Low": (0, 24),
    "Medium": (25, 49),
    "High": (50, 74),
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
    "blacklist_flag": "With Black List Ignor",
    "entry_date": "Entry Date",
    "user": "User",
    "debit_balance": "Debit Acc EOD Bal",
    "credit_balance": "Credit Acc EOD Bal",
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
    "doc_customer": "Վճարող",
    "note": "Նշումներ",
    "cash_register": "Դրամարկղ",
    "debit_account": "Դեբետ հաշիվ",
    "credit_account": "Կրեդիտ հաշիվ",
    "branch": "Մասնաճյուղ",
    "cash_flag": "Փաստ. համար",
    "entry_date": "Մուտքի ամս.",
    "user": "Օգտագործող",
    "blacklist_flag": "Սև ցուցակի անտեսմամբ",
}

DEFAULT_RULES: Dict[str, Dict[str, Any]] = {
    "large_amount": {"enabled": True, "weight": 18, "threshold": 25000000, "description": "Transaction amount exceeds threshold."},
    "round_amount": {"enabled": True, "weight": 8, "threshold": 100000, "description": "Amount is a large round number."},
    "cash_transaction": {"enabled": True, "weight": 12, "threshold": None, "description": "Cash-related transaction."},
    "night_activity": {"enabled": True, "weight": 8, "threshold": None, "description": "Transaction occurred at night."},
    "weekend_activity": {"enabled": True, "weight": 6, "threshold": None, "description": "Transaction occurred on weekend."},
    "duplicate_document": {"enabled": True, "weight": 10, "threshold": None, "description": "Repeated document number."},
    "high_frequency_sender": {"enabled": True, "weight": 14, "threshold": 20, "description": "Sender has unusually high transaction count."},
    "many_counterparties": {"enabled": True, "weight": 14, "threshold": 15, "description": "Sender interacts with many counterparties."},
    "multiple_currencies": {"enabled": True, "weight": 10, "threshold": 2, "description": "Customer/account uses multiple currencies."},
    "balance_anomaly": {"enabled": True, "weight": 12, "threshold": 0.9, "description": "Amount unusually large versus available EOD balance."},
}

@dataclass
class AppState:
    """Container for runtime mapping and configuration."""
    column_mapping: Dict[str, str] = field(default_factory=dict)
    risk_rules: Dict[str, Dict[str, Any]] = field(default_factory=lambda: DEFAULT_RULES.copy())
