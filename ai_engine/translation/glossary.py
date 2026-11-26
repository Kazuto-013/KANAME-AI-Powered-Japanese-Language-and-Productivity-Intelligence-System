"""
Bilingual domain glossaries for Japanese ↔ English translation.

4 domains are covered:
  - business_general (一般ビジネス用語)
  - technology_ai (IT・AI技術用語)
  - finance_accounting (経理・財務用語)
  - legal_contract (契約・法務用語)

Glossaries are represented as bidirectional dicts (ja→en and en→ja)
built from sorted lists of canonical pairs.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


_DOMAIN_PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "business_general": [
        ("御中", "Dear Sir or Madam"),
        ("お世話になっております", "We are deeply indebted to you"),
        ("いつもお世話になっております", "Thank you for your continued support"),
        ("お疲れ様です", "Good work / Thank you for your hard work"),
        ("よろしくお願い申し上げます", "I look forward to working with you"),
        ("何卒よろしくお願いいたします", "We kindly ask for your favorable consideration"),
        ("ご確認の程、何卒よろしくお願い申し上げます", "Please kindly review and confirm"),
        ("貴社", "your company"),
        ("弊社", "our company"),
        ("当社", "our company"),
        ("御社", "your esteemed company"),
        ("取引先", "business partner / client"),
        ("顧客", "customer"),
        ("納期", "delivery deadline"),
        ("請求書", "invoice"),
        ("見積書", "quotation / estimate"),
        ("契約書", "contract"),
        ("送付状", "cover letter / transmittal"),
        ("職務経歴書", "career history / resume"),
        ("履歴書", "Japanese standard resume"),
        ("中途採用", "mid-career hiring"),
        ("新卒採用", "new-graduate hiring"),
        ("面接", "interview"),
        ("役職", "position / title"),
        ("部署", "department"),
        ("部長", "Department Manager"),
        ("課長", "Section Manager"),
        ("担当", "person in charge"),
        ("本社", "headquarters"),
        ("支社", "branch office"),
        ("営業日", "business day"),
    ],
    "technology_ai": [
        ("人工知能", "artificial intelligence"),
        ("機械学習", "machine learning"),
        ("深層学習", "deep learning"),
        ("自然言語処理", "natural language processing"),
        ("音声認識", "speech recognition"),
        ("画像認識", "image recognition"),
        ("ニューラルネットワーク", "neural network"),
        ("トランスフォーマー", "transformer"),
        ("大規模言語モデル", "large language model"),
        ("埋め込み", "embedding"),
        ("ベクトル", "vector"),
        ("トークン化", "tokenization"),
        ("形態素解析", "morphological analysis"),
        ("品詞", "part of speech"),
        ("固有表現抽出", "named entity recognition"),
        ("感情分析", "sentiment analysis"),
        ("翻訳", "translation"),
        ("要約", "summarization"),
        ("データセット", "dataset"),
        ("学習", "training"),
        ("推論", "inference"),
        ("検証", "validation"),
        ("精度", "accuracy"),
        ("再現率", "recall"),
        ("適合率", "precision"),
        ("過学習", "overfitting"),
        ("API", "API"),
        ("エンドポイント", "endpoint"),
        ("データベース", "database"),
        ("クラウド", "cloud"),
    ],
    "finance_accounting": [
        ("売上高", "net sales / revenue"),
        ("利益", "profit"),
        ("営業利益", "operating profit"),
        ("経常利益", "ordinary income"),
        ("当期純利益", "net income for the period"),
        ("総資産", "total assets"),
        ("純資産", "net assets"),
        ("負債", "liabilities"),
        ("資本金", "capital stock"),
        ("消費税", "consumption tax"),
        ("税込", "tax included"),
        ("税抜", "tax excluded"),
        ("適格請求書", "qualified invoice"),
        ("登録番号", "registration number"),
        ("小計", "subtotal"),
        ("合計", "total"),
        ("支払期限", "payment due date"),
        ("支払方法", "payment method"),
        ("銀行振込", "bank transfer"),
        ("口座振替", "direct debit"),
        ("クレジットカード", "credit card"),
        ("領収書", "receipt"),
        ("返金", "refund"),
        ("貸方", "credit"),
        ("借方", "debit"),
        ("仕訳", "journal entry"),
        ("決算", "closing of accounts / settlement"),
        ("中間決算", "interim settlement"),
        ("四半期決算", "quarterly settlement"),
        ("監査", "audit"),
    ],
    "legal_contract": [
        ("契約", "contract"),
        ("有効", "valid / effective"),
        ("無効", "void / invalid"),
        ("解除", "termination / cancellation"),
        ("解約", "cancellation"),
        ("違反", "breach / violation"),
        ("損害賠償", "damages / indemnification"),
        ("免責", "disclaimer / exemption"),
        ("秘密保持", "confidentiality"),
        ("機密情報", "confidential information"),
        ("知的財産権", "intellectual property rights"),
        ("著作権", "copyright"),
        ("特許", "patent"),
        ("商標", "trademark"),
        ("譲渡", "assignment / transfer"),
        ("再譲渡", "re-assignment / sub-transfer"),
        ("再許諾", "sublicense"),
        ("期間", "term / period"),
        ("自動更新", "automatic renewal"),
        ("合意", "agreement"),
        ("承諾", "acceptance"),
        ("申し込み", "offer"),
        ("当事者", "party / parties"),
        ("甲", "Party A"),
        ("乙", "Party B"),
        ("管轄", "jurisdiction"),
        ("準拠法", "governing law"),
        ("紛争", "dispute"),
        ("仲裁", "arbitration"),
        ("強制執行", "enforcement"),
    ],
}


def _build_bidirectional(pairs: List[Tuple[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    ja_to_en: Dict[str, str] = {}
    en_to_ja: Dict[str, str] = {}
    for ja, en in pairs:
        ja_to_en[ja] = en
        en_to_ja[en.lower()] = ja
    return ja_to_en, en_to_ja


DOMAINS: Dict[str, Dict[str, Dict[str, str]]] = {}
for _name, _pairs in _DOMAIN_PAIRS.items():
    _je, _ej = _build_bidirectional(_pairs)
    DOMAINS[_name] = {"ja_to_en": _je, "en_to_ja": _ej}


def list_domains() -> List[str]:
    return sorted(_DOMAIN_PAIRS.keys())


def get_glossary_pairs(domain: str) -> List[Tuple[str, str]]:
    return list(_DOMAIN_PAIRS.get(domain, []))


def lookup(source_lang: str, target_lang: str, term: str, domain: str | None = None) -> str | None:
    src = source_lang.lower()[:2]
    tgt = target_lang.lower()[:2]
    domains_to_try = [domain] if domain else list(_DOMAIN_PAIRS.keys())
    direction_key = "ja_to_en" if src == "ja" and tgt == "en" else "en_to_ja"
    needle = term if direction_key == "ja_to_en" else term.lower()
    for d in domains_to_try:
        if d is None:
            continue
        table = DOMAINS.get(d, {}).get(direction_key, {})
        if needle in table:
            return table[needle]
    return None


def apply_glossary(text: str, source_lang: str, target_lang: str, domain: str | None = None) -> Tuple[str, int]:
    src = source_lang.lower()[:2]
    tgt = target_lang.lower()[:2]
    if src == tgt:
        return text, 0
    direction_key = "ja_to_en" if src == "ja" and tgt == "en" else "en_to_ja"
    domains_to_try = [domain] if domain else list(_DOMAIN_PAIRS.keys())
    applied = 0
    result = text
    for d in domains_to_try:
        if d is None:
            continue
        table = DOMAINS.get(d, {}).get(direction_key, {})
        for src_term, tgt_term in table.items():
            if src_term and src_term in result:
                result = result.replace(src_term, f"⟪{tgt_term}⟫")
                applied += 1
    if applied:
        result = result.replace("⟪", "").replace("⟫", "")
    return result, applied
