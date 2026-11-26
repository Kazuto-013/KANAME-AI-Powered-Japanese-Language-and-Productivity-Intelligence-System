"""
Japanese ↔ English translation engine.

Pipeline order:
  1. Short-circuit: empty or identical source→target → return as-is
  2. Glossary pre-processing: apply domain glossary hits
  3. LLM backend (first available):
       a. Anthropic Claude SDK  (ANTHROPIC_API_KEY)
       b. Google Generative AI  (GOOGLE_API_KEY)
  4. If NO LLM backend is configured/available: graceful dictionary fallback
     (glossary substitution + basic word dictionary + kana detection).

All public entry points return a `TranslationResult` dataclass that includes:
  - translated text
  - which engine was used (claude / gemini / dictionary)
  - number of glossary hits applied
  - quality score
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import glossary as _glossary
from . import quality as _quality


_SUDACHI_AVAILABLE: bool = False
_SUDACHI_ENGINE: Any = None
try:
    from ai_engine.japanese_nlp import sudachi_engine as _sudachi
    _SUDACHI_AVAILABLE = True
    _SUDACHI_ENGINE = _sudachi
except Exception:
    _SUDACHI_AVAILABLE = False


_DICTIONARY_JA_EN: Dict[str, str] = {
    # Pronouns
    "私": "I", "私達": "we", "私たち": "we",
    "あなた": "you", "貴方": "you", "あなた達": "you", "あなたたち": "you",
    "彼": "he", "彼ら": "they", "彼女": "she", "彼女達": "they", "彼女たち": "they",
    "僕": "I", "俺": "I", "貴様": "you", "君": "you",
    "我々": "we", "我々達": "we",
    "これ": "this", "それ": "that", "あれ": "that over there",
    "これら": "these", "それら": "those", "あれら": "those over there",
    "ここ": "here", "そこ": "there", "あそこ": "over there",
    "どこ": "where", "どれ": "which", "誰": "who", "何": "what", "いつ": "when",
    "何故": "why", "なぜ": "why", "どう": "how",
    "この": "this", "その": "that", "あの": "that", "どの": "which",
    "こちら": "this way / I / me", "そちら": "that way / you",
    # Demonstratives / Time
    "今日": "today", "明日": "tomorrow", "昨日": "yesterday",
    "毎日": "every day", "毎週": "every week", "毎月": "every month", "毎年": "every year",
    "今": "now", "今朝": "this morning", "今晩": "tonight", "今夜": "tonight",
    "前": "before / in front", "後": "after / behind",
    "後で": "later", "先に": "earlier / first", "以前": "previously / before",
    "時": "o'clock", "時間": "time", "日": "day", "週": "week", "月": "month", "年": "year",
    "朝": "morning", "昼": "afternoon / noon", "晩": "evening", "夜": "night",
    "午前": "a.m.", "午後": "p.m.",
    "先日": "the other day", "近日": "soon", "将来": "future", "過去": "past",
    "現在": "present / now", "今日中": "today (by end of)",
    # Nouns: Places, society
    "会社": "company", "人": "person", "人々": "people", "人達": "people",
    "本": "book", "電話": "telephone", "メール": "email",
    "会議": "meeting", "仕事": "work", "学校": "school",
    "大学": "university", "先生": "teacher", "学生": "student",
    "お客様": "customer", "顧客": "client / customer", "データ": "data",
    "ファイル": "file", "システム": "system", "開発": "development",
    "設計": "design", "実装": "implementation", "テスト": "test",
    "本番": "production", "結果": "result", "確認": "confirmation",
    "報告": "report", "連絡": "contact", "依頼": "request",
    "相談": "consultation", "対応": "support / response",
    "対策": "countermeasure", "問題": "problem", "課題": "challenge",
    "解決": "solution", "改善": "improvement",
    # Nouns: Tech AI
    "人工知能": "artificial intelligence", "機械学習": "machine learning",
    "深層学習": "deep learning", "自然言語処理": "natural language processing",
    "音声認識": "speech recognition", "画像認識": "image recognition",
    "ニューラルネットワーク": "neural network", "トランスフォーマー": "transformer",
    "大規模言語モデル": "large language model", "埋め込み": "embedding",
    "ベクトル": "vector", "トークン化": "tokenization",
    "形態素解析": "morphological analysis", "品詞": "part of speech",
    "固有表現抽出": "named entity recognition", "感情分析": "sentiment analysis",
    "翻訳": "translation", "要約": "summary", "データセット": "dataset",
    "学習": "training", "推論": "inference", "検証": "validation",
    "精度": "accuracy", "再現率": "recall", "適合率": "precision",
    "過学習": "overfitting", "エンドポイント": "endpoint",
    "データベース": "database", "クラウド": "cloud",
    "ユーザー": "user", "画面": "screen / UI", "機能": "feature",
    "サービス": "service", "プロダクト": "product", "製品": "product",
    "新製品": "new product", "技術": "technology", "活用": "utilization / use",
    "モデル": "model", "アプリケーション": "application", "アプリ": "app",
    # Nouns: Nature / Living
    "公園": "park", "鳥": "bird", "花": "flower", "空": "sky",
    "海": "sea", "山": "mountain", "川": "river", "森": "forest",
    "桜": "cherry blossom", "四季": "four seasons",
    "春": "spring", "夏": "summer", "秋": "autumn", "冬": "winter",
    "紅葉": "autumn leaves / fall foliage", "雪": "snow",
    "新鮮な": "fresh", "空気": "air", "散歩": "walk",
    "天気": "weather", "雨": "rain", "風": "wind",
    # Nouns: Business / Finance / Legal
    "御中": "Dear Sir or Madam", "貴社": "your company",
    "弊社": "our company", "当社": "our company", "御社": "your esteemed company",
    "取引先": "business partner", "納期": "delivery deadline",
    "請求書": "invoice", "見積書": "quotation", "契約書": "contract",
    "送付状": "cover letter / transmittal",
    "職務経歴書": "career history", "履歴書": "resume",
    "中途採用": "mid-career hiring", "新卒採用": "new graduate hiring",
    "面接": "interview", "役職": "position / title", "部署": "department",
    "部長": "Department Manager", "課長": "Section Manager",
    "担当": "person in charge", "本社": "headquarters",
    "支社": "branch office", "営業日": "business day",
    "売上高": "net sales", "利益": "profit", "営業利益": "operating profit",
    "経常利益": "ordinary income", "当期純利益": "net income",
    "総資産": "total assets", "純資産": "net assets", "負債": "liabilities",
    "資本金": "capital stock", "消費税": "consumption tax",
    "税込": "tax included", "税抜": "tax excluded",
    "適格請求書": "qualified invoice", "登録番号": "registration number",
    "小計": "subtotal", "合計": "total", "支払期限": "payment due date",
    "支払方法": "payment method", "銀行振込": "bank transfer",
    "口座振替": "direct debit", "クレジットカード": "credit card",
    "領収書": "receipt", "返金": "refund", "貸方": "credit",
    "借方": "debit", "仕訳": "journal entry", "決算": "settlement",
    "契約": "contract", "有効": "valid", "無効": "void",
    "解除": "termination", "解約": "cancellation", "違反": "breach",
    "損害賠償": "damages", "免責": "disclaimer", "秘密保持": "confidentiality",
    "機密情報": "confidential information",
    "知的財産権": "intellectual property rights",
    "著作権": "copyright", "特許": "patent", "商標": "trademark",
    "譲渡": "assignment", "当事者": "party", "甲": "Party A",
    "乙": "Party B", "管轄": "jurisdiction", "準拠法": "governing law",
    "紛争": "dispute",
    # Nouns: Common objects / misc
    "住所": "address", "電話番号": "phone number",
    "名前": "name", "氏名": "name (full)", "年齢": "age",
    "性別": "gender", "男性": "male", "女性": "female",
    "生年月日": "date of birth", "学歴": "education", "職歴": "work history",
    "資格": "qualification / license", "免許": "license",
    "趣味": "hobby", "特技": "special skill",
    "タイトル": "title", "件名": "subject",
    "概要": "overview / summary", "詳細": "details",
    "メッセージ": "message", "情報": "information",
    "内容": "content", "意見": "opinion",
    "理由": "reason", "目的": "purpose",
    "方法": "method", "方法論": "methodology",
    "研究": "research", "調査": "survey / investigation",
    # Adjectives
    "良い": "good", "いい": "good", "悪い": "bad",
    "大きい": "big / large", "小さい": "small / little",
    "新しい": "new", "古い": "old", "高い": "expensive / tall / high",
    "安い": "cheap / inexpensive", "安価な": "inexpensive",
    "長い": "long", "短い": "short", "広い": "wide / spacious",
    "狭い": "narrow", "深い": "deep", "浅い": "shallow",
    "強い": "strong", "弱い": "weak", "速い": "fast", "遅い": "slow",
    "美しい": "beautiful", "可愛い": "cute", "きれいな": "beautiful / clean",
    "素敵な": "wonderful / lovely", "素晴らしい": "wonderful / amazing",
    "面白い": "interesting / funny", "つまらない": "boring",
    "難しい": "difficult / hard", "易しい": "easy", "簡単な": "easy / simple",
    "便利な": "convenient", "不便な": "inconvenient",
    "重要な": "important", "必要な": "necessary", "不要な": "unnecessary",
    "可能な": "possible", "不可能な": "impossible",
    "楽しい": "fun / enjoyable", "嬉しい": "happy / glad",
    "悲しい": "sad", "寂しい": "lonely", "暑い": "hot (weather)",
    "寒い": "cold (weather)", "暖かい": "warm", "涼しい": "cool",
    "熱い": "hot (to touch)", "冷たい": "cold (to touch)",
    "多い": "many / much", "少ない": "few / little",
    "若い": "young", "年寄り": "old (person)",
    # Verbs - most common
    "する": "do", "やる": "do",
    "行く": "go", "来る": "come", "帰る": "return / go home",
    "見る": "see / look / watch", "聞く": "hear / listen / ask",
    "読む": "read", "書く": "write", "話す": "speak / talk", "言う": "say",
    "食べる": "eat", "飲む": "drink", "作る": "make / create",
    "使う": "use", "買う": "buy", "売る": "sell",
    "持つ": "have / hold", "与える": "give", "もらう": "receive",
    "習う": "learn (from a person)", "学ぶ": "study / learn",
    "考える": "think", "知る": "know", "分かる": "understand",
    "調べる": "investigate / look up",
    "確認する": "confirm", "報告する": "report",
    "連絡する": "contact", "依頼する": "request",
    "対応する": "handle / correspond", "改善する": "improve",
    "解決する": "solve / resolve", "開発する": "develop",
    "設計する": "design", "実装する": "implement", "テストする": "test",
    "活用する": "utilize / make use of", "導入する": "introduce / implement",
    "向上させる": "improve", "必要とする": "need / require",
    "挙げる": "raise / list", "上げる": "raise / increase",
    "下げる": "lower / reduce", "増やす": "increase",
    "減らす": "decrease", "始める": "begin / start",
    "終わる": "end / finish", "続ける": "continue",
    "待つ": "wait", "歩く": "walk", "走る": "run",
    "寝る": "sleep", "起きる": "wake up",
    "出勤する": "go to work", "退勤する": "leave work",
    "申し込む": "apply", "応募する": "apply for (a job)",
    "招く": "invite", "参加する": "participate / join",
    # Polite phrases (multi-character wholes - take priority due to sort by length)
    "ありがとう": "thank you", "ありがとうございます": "thank you very much",
    "感謝します": "I appreciate it / thank you",
    "すみません": "excuse me / sorry", "ごめんなさい": "I'm sorry",
    "申し訳ありません": "I am very sorry",
    "はい": "yes", "いいえ": "no", "お願いします": "please",
    "よろしくお願いします": "I look forward to working with you / best regards",
    "よろしく": "best regards",
    "はじめまして": "nice to meet you", "さようなら": "goodbye",
    "こんにちは": "hello", "こんばんは": "good evening",
    "おはようございます": "good morning", "おやすみなさい": "good night",
    "いただきます": "(before a meal)",
    "ごちそうさまでした": "(after a meal, thank you)",
    "お世話になります": "Thank you for your continued support",
    "いつもお世話になっております": "Thank you as always for your support",
    "お疲れ様です": "Good work / thank you for your effort",
    "ご確認の程、何卒よろしくお願い申し上げます": "Please kindly review and confirm",
    "何卒よろしくお願いいたします": "We kindly ask for your favorable consideration",
    # Auxiliary / structural markers (stripped later; dictionary for completeness)
    "です": "", "ます": "", "でした": "(past)", "ました": "(past)",
    "だ": "", "である": "is",
    "ません": "(negation)", "ませんでした": "(past negation)",
    "ない": "not", "なかった": "did not",
    "たい": "want to", "らしい": "it seems",
    "よう": "as / like", "そう": "so / looks like",
    # Missing corpus nouns / time / compounds (added 2026-10-01)
    "毎朝": "every morning", "朝散歩": "morning walk",
    "チーム": "team", "開発チーム": "development team",
    "報告書": "report", "レポート": "report",
    "改善策": "improvement measure", "改善案": "improvement plan",
    "ミーティング": "meeting", "打ち合わせ": "meeting",
    "散歩する": "walk", "散歩します": "walk", "散歩しました": "walked",
    "送る": "send", "送ります": "send", "送りました": "sent", "送信": "send",
    "考え": "thought / idea", "学び": "learning",
    "ために": "in order to", "為に": "in order to",
    "そして": "and then", "それから": "then", "だから": "so",
    "します": "do", "しました": "did", "すること": "to do",
    "ください": "please", "下さい": "please",
    "してください": "please do", "なさってください": "please do",
    "ご連絡ください": "please contact us",
    "ご確認ください": "please confirm",
    "昨日": "yesterday", "一昨日": "the day before yesterday",
    "明日": "tomorrow", "明後日": "the day after tomorrow",
    "今年": "this year", "去年": "last year", "来年": "next year",
    "月曜日": "Monday", "火曜日": "Tuesday", "水曜日": "Wednesday",
    "木曜日": "Thursday", "金曜日": "Friday", "土曜日": "Saturday", "日曜日": "Sunday",
    "顧客": "customer", "得意先": "customer / client",
    "取引先": "business partner / client", "ステークホルダー": "stakeholder",
    "メールアドレス": "email address",
    "データファイル": "data file",
    "課題": "challenge / issue", "目標": "goal / target",
    "方法": "method / way", "目的": "purpose / objective",
    "経緯": "background / circumstances", "経過": "progress / elapsed time",
    "内容": "content / details", "詳細": "details",
    "概要": "overview / summary", "全体": "the whole / overall",
    "部分": "part / portion",
    "新しいシステム": "new system",
    "送信する": "send", "送信します": "send", "送信しました": "sent",
    "対応する": "handle / support", "対応します": "handle", "対応しました": "handled",
    "依頼する": "request", "依頼します": "request", "依頼しました": "requested",
    "参加する": "participate", "参加します": "participate", "参加しました": "participated",
    "申し込む": "apply", "申込": "application",
    "応募する": "apply for", "採用": "employment / hiring",
    "勉強": "study", "勉強する": "study",
    "研究": "research", "研究する": "research",
    "調査": "investigation / survey", "調査する": "investigate",
    "準備": "preparation", "準備する": "prepare",
    "確認する": "confirm", "連絡する": "contact",
    "報告する": "report", "改善する": "improve",
    "解決する": "solve / resolve", "設計する": "design",
    "実装する": "implement", "開発する": "develop",
    "テストする": "test", "導入する": "introduce",
    "活用する": "utilize", "利用する": "use / utilize",
    "向上": "improvement", "向上させる": "improve",
    "する為": "in order to do",
    "で、": ", ", "、": ", ",
    "！": "!", "？": "?",
    # Supplementary common words (everyday business / tech)
    "来週": "next week", "今週": "this week", "先週": "last week",
    "来月": "next month", "先月": "last month",
    "増加": "increase", "増加する": "increase", "増加しました": "increased",
    "減少": "decrease", "減少する": "decrease",
    "接続": "connection", "接続する": "connect", "接続できません": "cannot connect",
    "提出": "submission", "提出する": "submit", "提出します": "submit",
    "住む": "live", "住んでいます": "am living / live",
    "変更": "change", "変更する": "change",
    "開始": "start", "開始する": "start",
}


_EN_VERB_PAST: Dict[str, str] = {
    "do": "did", "go": "went", "come": "came", "return": "returned",
    "see": "saw", "watch": "watched", "listen": "listened",
    "hear": "heard", "read": "read", "write": "wrote", "speak": "spoke",
    "say": "said", "eat": "ate", "drink": "drank", "make": "made",
    "create": "created", "use": "used", "buy": "bought", "sell": "sold",
    "have": "had", "hold": "held", "give": "gave", "receive": "received",
    "learn": "learned", "study": "studied",
    "think": "thought", "know": "knew", "understand": "understood",
    "investigate": "investigated", "confirm": "confirmed", "report": "reported",
    "contact": "contacted", "request": "requested",
    "handle": "handled", "improve": "improved", "solve": "solved",
    "resolve": "resolved", "develop": "developed", "design": "designed",
    "implement": "implemented", "test": "tested",
    "utilize": "utilized", "introduce": "introduced",
    "need": "needed", "require": "required",
    "raise": "raised", "increase": "increased",
    "lower": "lowered", "reduce": "reduced",
    "begin": "began", "start": "started", "finish": "finished",
    "continue": "continued", "wait": "waited", "walk": "walked",
    "run": "ran", "sleep": "slept", "wake": "woke",
    "apply": "applied", "invite": "invited",
    "participate": "participated", "join": "joined",
    "be": "was", "can": "could", "will": "would",
}

_DICTIONARY_EN_JA: Dict[str, str] = {
    # Pronouns
    "i": "私", "me": "私", "my": "私の", "mine": "私の物",
    "myself": "自分自身", "you": "あなた", "your": "あなたの",
    "yours": "あなたの物", "yourself": "あなた自身",
    "he": "彼", "him": "彼", "his": "彼の", "himself": "彼自身",
    "she": "彼女", "her": "彼女の", "hers": "彼女の物", "herself": "彼女自身",
    "we": "私たち", "us": "私たち", "our": "私たちの",
    "ours": "私たちの物", "ourselves": "私たち自身",
    "they": "彼ら", "them": "彼ら", "their": "彼らの",
    "theirs": "彼らの物", "themselves": "彼ら自身",
    "this": "これ", "that": "それ", "these": "これら", "those": "それら",
    "here": "ここ", "there": "そこ", "where": "どこ", "which": "どれ",
    "who": "誰", "what": "何", "when": "いつ", "why": "なぜ", "how": "どう",
    # Determiners
    "a": "1つの", "an": "1つの", "the": "その", "this": "この",
    "that": "その", "these": "これらの", "those": "それらの",
    "some": "いくつかの", "any": "任意の", "every": "全ての",
    "all": "全て", "each": "各", "both": "両方", "either": "どちらか",
    "neither": "どちらでもない", "no": "無し", "none": "何も無い",
    # Time words
    "today": "今日", "tomorrow": "明日", "yesterday": "昨日",
    "now": "今", "before": "前", "after": "後", "later": "後で",
    "soon": "すぐに", "already": "既に", "yet": "まだ",
    "still": "まだ", "just": "ちょうど",
    "morning": "朝", "afternoon": "昼", "evening": "晩",
    "night": "夜", "tonight": "今夜",
    "time": "時間", "day": "日", "week": "週", "month": "月",
    "year": "年", "hour": "時間", "minute": "分", "second": "秒",
    "monday": "月曜日", "tuesday": "火曜日", "wednesday": "水曜日",
    "thursday": "木曜日", "friday": "金曜日", "saturday": "土曜日",
    "sunday": "日曜日",
    "january": "1月", "february": "2月", "march": "3月", "april": "4月",
    "may": "5月", "june": "6月", "july": "7月", "august": "8月",
    "september": "9月", "october": "10月", "november": "11月",
    "december": "12月",
    # Modals / aux
    "is": "です", "are": "です", "am": "です", "was": "でした",
    "were": "でした", "be": "である", "been": "であった", "being": "であり",
    "have": "持つ", "has": "持つ", "had": "持っていた",
    "do": "する", "does": "する", "did": "した",
    "will": "するつもりです", "would": "するであろう",
    "shall": "するべきです", "should": "すべきです",
    "can": "できる", "could": "できた",
    "may": "かもしれない", "might": "かもしれない",
    "must": "しなければならない",
    # Verbs common
    "go": "行く", "goes": "行く", "went": "行きました", "going": "行くこと",
    "come": "来る", "came": "来ました", "coming": "来ること",
    "return": "帰る", "returned": "帰りました",
    "see": "見る", "saw": "見ました", "seen": "見られた",
    "look": "見る", "watch": "見る",
    "listen": "聞く", "hear": "聞く", "heard": "聞きました",
    "read": "読む", "reading": "読書",
    "write": "書く", "wrote": "書きました", "written": "書かれた",
    "speak": "話す", "say": "言う", "said": "言いました",
    "talk": "話す", "tell": "伝える",
    "eat": "食べる", "ate": "食べました", "eaten": "食べられた",
    "drink": "飲む", "drank": "飲みました",
    "make": "作る", "made": "作りました",
    "create": "作成する", "created": "作成しました",
    "build": "構築する", "built": "構築しました",
    "use": "使う", "used": "使いました",
    "utilize": "活用する", "utilized": "活用しました",
    "buy": "買う", "bought": "買いました", "purchase": "購入する",
    "sell": "売る", "sold": "売りました",
    "have": "持つ", "hold": "保持する", "held": "保持しました",
    "give": "与える", "gave": "与えました", "given": "与えられた",
    "receive": "受け取る", "received": "受け取りました",
    "get": "得る", "got": "得ました", "gotten": "得られた",
    "take": "取る", "took": "取りました", "taken": "取られた",
    "put": "置く", "keep": "保つ", "kept": "保ちました",
    "send": "送る", "sent": "送りました",
    "learn": "学ぶ", "study": "勉強する", "studied": "勉強しました",
    "teach": "教える", "taught": "教えました",
    "think": "考える", "thought": "考えました",
    "know": "知る", "knew": "知りました", "known": "知られている",
    "understand": "分かる", "understood": "分かりました",
    "find": "見つける", "found": "見つけました",
    "search": "検索する", "look for": "探す",
    "investigate": "調査する", "research": "研究する",
    "survey": "調査", "check": "確認する",
    "confirm": "確認する", "confirmed": "確認しました",
    "verify": "検証する", "validate": "検証する",
    "report": "報告する", "reported": "報告しました",
    "contact": "連絡する", "reached out": "連絡しました",
    "request": "依頼する", "ask": "依頼する / 尋ねる",
    "asked": "依頼しました",
    "support": "対応する", "handle": "処理する",
    "improve": "改善する", "improved": "改善しました",
    "enhance": "強化する", "enhanced": "強化しました",
    "solve": "解決する", "resolve": "解決する", "solved": "解決しました",
    "develop": "開発する", "developed": "開発しました",
    "design": "設計する", "designed": "設計しました",
    "implement": "実装する", "implemented": "実装しました",
    "test": "テストする", "tested": "テストしました",
    "introduce": "導入する", "introduced": "導入しました",
    "deploy": "デプロイする", "deployed": "デプロイしました",
    "launch": "リリースする", "launched": "リリースしました",
    "need": "必要とする", "require": "要求する",
    "want": "欲しい", "want to": "～したい",
    "like": "好き", "love": "愛する", "hate": "嫌う",
    "begin": "始める", "start": "開始する", "began": "始めました",
    "started": "開始しました", "finish": "終了する",
    "end": "終わる", "ended": "終わりました", "completed": "完了しました",
    "continue": "続ける", "continued": "続けました",
    "stop": "止める", "pause": "一時停止する",
    "wait": "待つ", "waited": "待ちました",
    "walk": "歩く", "walked": "歩きました",
    "run": "走る", "ran": "走りました",
    "sleep": "寝る", "slept": "寝ました",
    "wake up": "起きる", "woke up": "起きました",
    "work": "仕事をする", "worked": "働きました",
    "attend": "出席する", "join": "参加する", "participate": "参加する",
    "apply": "申し込む", "applied": "申し込みました",
    "invite": "招待する", "invited": "招待しました",
    # Adjectives common
    "good": "良い", "better": "より良い", "best": "最高の",
    "bad": "悪い", "worse": "より悪い", "worst": "最悪の",
    "big": "大きい", "large": "大きい", "small": "小さい", "little": "小さい",
    "new": "新しい", "old": "古い",
    "expensive": "高い", "high": "高い", "tall": "高い",
    "cheap": "安い", "inexpensive": "安価な",
    "long": "長い", "short": "短い",
    "wide": "広い", "narrow": "狭い",
    "deep": "深い", "shallow": "浅い",
    "strong": "強い", "weak": "弱い",
    "fast": "速い", "quick": "速い", "rapid": "急速な", "slow": "遅い",
    "beautiful": "美しい", "pretty": "きれいな",
    "cute": "可愛い", "clean": "清潔な", "dirty": "汚い",
    "wonderful": "素晴らしい", "amazing": "驚くべき",
    "nice": "素敵な", "lovely": "愛らしい",
    "interesting": "面白い", "funny": "面白い",
    "boring": "つまらない", "exciting": "エキサイティングな",
    "difficult": "難しい", "hard": "難しい / 固い",
    "easy": "簡単 / 易しい", "simple": "単純な",
    "complex": "複雑な", "complicated": "複雑な",
    "convenient": "便利な", "useful": "有用な",
    "inconvenient": "不便な", "useless": "役に立たない",
    "important": "重要な", "necessary": "必要な",
    "unnecessary": "不要な", "essential": "必須の",
    "possible": "可能な", "impossible": "不可能な",
    "available": "利用可能な", "unavailable": "利用不可な",
    "happy": "幸せな / 嬉しい", "sad": "悲しい",
    "angry": "怒っている", "pleased": "喜んでいる",
    "glad": "嬉しい", "proud": "誇りに思う",
    "lonely": "寂しい", "tired": "疲れている",
    "hungry": "お腹が空いている", "full": "お腹がいっぱい",
    "hot": "暑い / 熱い", "cold": "寒い / 冷たい",
    "warm": "暖かい", "cool": "涼しい",
    "many": "多くの", "much": "多くの",
    "few": "少しの", "little": "少しの",
    "more": "より多くの", "less": "より少ない",
    "most": "最も多くの", "least": "最も少ない",
    "same": "同じ", "different": "異なる",
    "young": "若い", "healthy": "健康な", "sick": "病気の",
    "rich": "豊かな / 金持ちの", "poor": "貧しい",
    "safe": "安全な", "dangerous": "危険な",
    "true": "真実の", "false": "偽の",
    "correct": "正しい", "right": "正しい",
    "wrong": "間違った", "incorrect": "正しくない",
    # Nouns - Person, workplace
    "company": "会社", "person": "人", "people": "人々",
    "team": "チーム", "member": "メンバー",
    "colleague": "同僚", "coworker": "同僚",
    "manager": "マネージャー", "director": "ディレクター",
    "president": "社長", "ceo": "CEO",
    "employee": "従業員", "staff": "スタッフ",
    "engineer": "エンジニア", "developer": "開発者",
    "designer": "デザイナー", "analyst": "アナリスト",
    "book": "本", "document": "文書", "paper": "紙 / 論文",
    "report": "報告書", "summary": "概要",
    "plan": "計画", "project": "プロジェクト",
    "meeting": "会議", "schedule": "スケジュール",
    "deadline": "締め切り",
    "telephone": "電話", "phone": "電話", "email": "メール",
    "mail": "メール", "message": "メッセージ",
    "customer": "お客様", "client": "顧客",
    "user": "ユーザー", "partner": "パートナー",
    "vendor": "ベンダー", "supplier": "サプライヤー",
    "data": "データ", "file": "ファイル", "folder": "フォルダ",
    "system": "システム", "database": "データベース",
    "server": "サーバー", "network": "ネットワーク",
    "cloud": "クラウド", "software": "ソフトウェア",
    "hardware": "ハードウェア", "application": "アプリケーション",
    "app": "アプリ", "website": "ウェブサイト",
    "api": "API", "endpoint": "エンドポイント",
    "product": "製品", "service": "サービス",
    "feature": "機能", "function": "機能",
    "screen": "画面", "ui": "UI", "interface": "インターフェース",
    "technology": "技術", "development": "開発",
    "result": "結果", "output": "出力", "input": "入力",
    "process": "プロセス", "method": "方法",
    "solution": "解決策", "answer": "答え",
    "reason": "理由", "cause": "原因",
    "problem": "問題", "issue": "課題",
    "challenge": "課題", "risk": "リスク",
    "benefit": "メリット", "advantage": "利点",
    "disadvantage": "欠点", "cost": "コスト",
    "price": "価格", "value": "価値",
    "quality": "品質", "performance": "パフォーマンス",
    "progress": "進捗", "status": "状況",
    "information": "情報", "content": "内容",
    "detail": "詳細", "overview": "概要",
    "opinion": "意見", "idea": "アイデア",
    "knowledge": "知識", "experience": "経験",
    "skill": "スキル", "ability": "能力",
    "school": "学校", "university": "大学",
    "college": "大学", "student": "学生",
    "teacher": "先生", "class": "クラス", "lesson": "レッスン",
    "course": "コース", "exam": "試験", "test": "テスト",
    "grade": "成績", "graduation": "卒業",
    # Nouns - nature
    "weather": "天気", "sky": "空", "air": "空気",
    "rain": "雨", "wind": "風", "snow": "雪",
    "sun": "太陽", "moon": "月", "star": "星",
    "tree": "木", "flower": "花", "grass": "草",
    "bird": "鳥", "animal": "動物", "cat": "猫", "dog": "犬",
    "park": "公園", "city": "街", "town": "町",
    "village": "村", "country": "国", "world": "世界",
    "japan": "日本", "tokyo": "東京",
    "spring": "春", "summer": "夏",
    "autumn": "秋", "fall": "秋", "winter": "冬",
    "season": "季節",
    # Business / finance / legal phrases
    "thank you for your continued support": "いつもお世話になっております",
    "best regards": "よろしくお願いします",
    "sincerely": "敬具", "dear": "拝啓",
    "invoice": "請求書", "quotation": "見積書",
    "estimate": "見積もり", "contract": "契約",
    "agreement": "契約書", "cover letter": "送付状",
    "resume": "履歴書", "cv": "職務経歴書",
    "career history": "職務経歴書",
    "delivery deadline": "納期", "payment due": "支払期限",
    "payment method": "支払方法",
    "bank transfer": "銀行振込", "credit card": "クレジットカード",
    "receipt": "領収書", "refund": "返金",
    "confidential": "機密", "confidentiality": "秘密保持",
    "copyright": "著作権", "patent": "特許",
    "trademark": "商標", "intellectual property": "知的財産",
    # Greetings & thanks (multi-word, must come before single words in sort)
    "thank you very much": "ありがとうございます",
    "thank you": "ありがとう", "thanks": "ありがとう",
    "sorry": "すみません", "excuse me": "すみません",
    "yes": "はい", "no": "いいえ",
    "please": "お願いします",
    "hello": "こんにちは", "goodbye": "さようなら",
    "good morning": "おはようございます",
    "good afternoon": "こんにちは",
    "good evening": "こんばんは",
    "good night": "おやすみなさい",
    "nice to meet you": "はじめまして",
    "how are you": "お元気ですか",
    "please check and confirm": "ご確認の程、何卒よろしくお願い申し上げます",
    # Locations / misc
    "street": "通り", "road": "道",
    "building": "建物", "office": "オフィス",
    "home": "家", "house": "家", "room": "部屋",
    "door": "ドア", "window": "窓",
    "table": "テーブル", "chair": "椅子",
    "computer": "コンピューター", "laptop": "ノートパソコン",
    "keyboard": "キーボード", "mouse": "マウス",
    "language": "言語", "japanese": "日本語",
    "english": "英語", "word": "単語",
    "sentence": "文章", "paragraph": "段落",
    # Missing compounds & conjugated EN forms (added 2026-10-01 for EN->JA 65% bar)
    "last": "最後の / 前の", "previous": "前の", "next": "次の",
    "last year": "昨年 / 去年", "last month": "先月", "last week": "先週",
    "this year": "今年", "this month": "今月", "this week": "今週",
    "next year": "来年", "next month": "来月", "next week": "来週",
    "every day": "毎日", "every morning": "毎朝", "every week": "毎週",
    "every month": "毎月", "every year": "毎年",
    "studies": "勉強する / 学ぶ", "studied": "勉強しました", "studying": "勉強している",
    "sending": "送っている", "sends": "送る",
    "artificial": "人工の", "intelligence": "知能 / 知性",
    "artificial intelligence": "人工知能", "machine learning": "機械学習",
    "deep learning": "深層学習", "data science": "データサイエンス",
    "data file": "データファイル", "data files": "データファイル群",
    "email address": "メールアドレス",
    "development team": "開発チーム", "dev team": "開発チーム",
    "new system": "新しいシステム",
    "customer support": "顧客サポート", "sales report": "売上報告書",
    "business meeting": "会議 / 打ち合わせ",
    # Conjugated verb forms (3rd sing, progressive, past participles)
    "does": "する", "doing": "している", "done": "完了した",
    "goes": "行く", "gone": "行った",
    "comes": "来る", "came": "来ました", "coming": "来ている",
    "sees": "見る", "seeing": "見ている", "seen": "見た",
    "reads": "読む", "reading": "読んでいる",
    "writes": "書く", "wrote": "書きました", "writing": "書いている",
    "speaks": "話す", "spoke": "話しました", "speaking": "話している",
    "eats": "食べる", "eating": "食べている",
    "drinks": "飲む", "drinking": "飲んでいる",
    "makes": "作る", "making": "作っている",
    "creates": "作成する", "creating": "作成している", "created": "作成しました",
    "uses": "使う", "using": "使っている",
    "buys": "買う", "buying": "買っている", "bought": "買いました",
    "sells": "売る", "selling": "売っている",
    "has": "持っている", "having": "持っていること",
    "gives": "与える", "giving": "与えている", "gave": "与えました",
    "receives": "受け取る", "received": "受け取りました", "receiving": "受け取っている",
    "learns": "学ぶ", "learning": "学んでいる", "learned": "学びました",
    "thinks": "考える", "thinking": "考えている",
    "knows": "知っている", "knew": "知っていました", "knowing": "知っていること",
    "understands": "理解する", "understood": "理解しました", "understanding": "理解している",
    "finds": "見つける", "found": "見つけました", "finding": "見つけている",
    "confirms": "確認する", "confirming": "確認している", "confirmed": "確認しました",
    "reports": "報告する", "reporting": "報告している", "reported": "報告しました",
    "contacts": "連絡する", "contacting": "連絡している", "contacted": "連絡しました",
    "requests": "依頼する", "requesting": "依頼している", "requested": "依頼しました",
    "improves": "改善する", "improving": "改善している",
    "solves": "解決する", "solving": "解決している", "solved": "解決しました",
    "resolves": "解決する", "resolved": "解決しました",
    "designs": "設計する", "designing": "設計している", "designed": "設計しました",
    "implements": "実装する", "implementing": "実装している", "implemented": "実装しました",
    "tests": "テストする", "testing": "テストしている", "tested": "テストしました",
    "introduces": "導入する", "introducing": "導入している", "introduced": "導入しました",
    "launches": "リリースする / 開始する", "launched": "リリースしました",
    "needs": "必要とする", "needing": "必要としている", "needed": "必要としました",
    "requires": "要求する", "required": "要求しました",
    "raises": "上げる", "raising": "上げている",
    "increases": "増やす", "increasing": "増やしている", "increased": "増やしました",
    "lowers": "下げる", "lowering": "下げている",
    "reduces": "減らす", "reducing": "減らしている", "reduced": "減らしました",
    "begins": "始める", "began": "始めました", "beginning": "始めている",
    "starts": "開始する", "starting": "開始している", "started": "開始しました",
    "finishes": "終える", "finishing": "終えている", "finished": "終えました",
    "continues": "続ける", "continuing": "続けている", "continued": "続けました",
    "waits": "待つ", "waiting": "待っている", "waited": "待ちました",
    "walks": "歩く", "walking": "歩いている", "walked": "歩きました",
    "runs": "走る", "ran": "走りました", "running": "走っている",
    "sleeps": "寝る", "sleeping": "寝ている", "slept": "寝ました",
    "wakes": "起きる", "waking": "起きている", "woke": "起きました",
    "applies": "申し込む / 適用する", "applied": "申し込みました / 適用しました",
    "applies for": "応募する",
    "invites": "招く / 招待する", "invited": "招きました",
    "participates": "参加する", "participating": "参加している", "participated": "参加しました",
    "joins": "参加する / 入社する", "joining": "参加している", "joined": "参加しました / 入社しました",
    # Particles & conjunctions: translate to empty (handled structurally) so they don't stay as English
    "to": "", "of": "", "and": "", "or": "", "but": "", "so": "",
    "in": "", "on": "", "at": "", "from": "", "for": "", "with": "", "by": "", "about": "",
    "through": "", "over": "", "under": "", "between": "", "before": "", "after": "",
    "during": "", "while": "", "if": "", "because": "", "then": "",
    "into": "", "onto": "", "upon": "", "within": "", "without": "", "within": "",
    # Supplementary common words (everyday business / tech)
    "live": "住む", "lives": "住む", "living": "住んでいる", "lived": "住んでいました",
    "connect": "接続する", "connects": "接続する", "connecting": "接続している",
    "connected": "接続しました", "connection": "接続",
    "submit": "提出する", "submits": "提出する", "submitted": "提出しました",
    "submission": "提出", "name": "名前",
    "increase": "増加する", "decrease": "減少する",
}


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    engine: str
    glossary_hits: int = 0
    quality_score: _quality.QualityScore = field(default_factory=lambda: _quality.QualityScore(
        overall=0.0, length_ratio_score=0.0, numeric_preservation_score=0.0,
        glossary_coverage_score=0.0, format_preservation_score=0.0, notes={},
    ))
    domain: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "engine": self.engine,
            "glossary_hits": self.glossary_hits,
            "domain": self.domain,
            "quality": self.quality_score.to_dict(),
            "notes": self.notes,
        }


def _normalize_lang(lang: str) -> str:
    return (lang or "en").lower().strip()[:2]


def _detect_engine() -> Tuple[Optional[str], Optional[str]]:
    """
    Return (engine_name, api_key_or_None) for the first available LLM backend,
    or (None, None) if neither provider has SDK + env key.

    Priority order (business rationale):
      1. Anthropic Claude
      2. Google Gemini
    """
    try:
        import anthropic  # type: ignore
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key and key.strip() and not key.strip().startswith(("your", "sk-xxx", "xxx", "<")):
            return "claude", key.strip()
    except Exception:
        pass

    try:
        import google.generativeai as genai  # type: ignore
        key = os.environ.get("GOOGLE_API_KEY")
        if key and key.strip() and not key.strip().startswith(("your", "xxx", "<")):
            return "gemini", key.strip()
    except Exception:
        pass

    return None, None


def _llm_timeout_seconds() -> float:
    raw = os.environ.get("KANAME_LLM_TIMEOUT_SECONDS", "120")
    try:
        return max(15.0, min(600.0, float(raw)))
    except Exception:
        return 120.0


def _system_prompt(source_lang: str, target_lang: str, domain: Optional[str]) -> str:
    lang_pairs = {
        ("ja", "en"): "Translate the given Japanese text into natural, fluent English.",
        ("en", "ja"): "Translate the given English text into natural, polite Japanese (です・ます調).",
    }
    src = _normalize_lang(source_lang)
    tgt = _normalize_lang(target_lang)
    base = lang_pairs.get((src, tgt), f"Translate from {source_lang} to {target_lang}.")
    domain_note = ""
    if domain:
        domain_note = f" Preserve domain-specific terms for the '{domain}' domain when possible."
    return (
        f"{base}{domain_note}\n"
        "Do NOT add any extra explanation, greeting, notes, or surrounding commentary.\n"
        "Return ONLY the translated text, exactly as long as needed, preserving numbers, "
        "punctuation and proper nouns. Do not wrap the output in markdown code blocks."
    )


def _call_llm(engine: str, key: str, system: str, user_text: str) -> str:
    if engine == "claude":
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user_text}],
            temperature=0.2,
        )
        parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()

    if engine == "gemini":
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=key)
        model_name = os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash")
        model = genai.GenerativeModel(model_name)
        prompt = f"{system}\n\nText to translate:\n{user_text}"
        resp = model.generate_content(prompt)
        text = ""
        if hasattr(resp, "text"):
            text = resp.text
        elif hasattr(resp, "candidates") and resp.candidates:
            text = str(resp.candidates[0])
        return (text or "").strip()

    raise ValueError(f"Unknown engine {engine}")


def _split_sentences(text: str, source_lang: str) -> List[str]:
    src = _normalize_lang(source_lang)
    if src == "ja":
        parts = re.split(r"(?<=。)|(?<=！)|(?<=？)", text)
    else:
        parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


# -----------------------------------------------------------------------------
# JA → EN heuristic helpers
# -----------------------------------------------------------------------------

_JA_PARTICLES_STRIP = {"は", "が", "を", "に", "で", "の", "から", "まで", "へ", "と", "も", "や", "か", "だけ", "しか", "くらい", "ぐらい", "など", "ばかり", "までに", "には", "では", "からの"}

_JA_AUX_PAST = {"ました", "でした", "た", "だった", "なかった", "ませんでした"}
_JA_AUX_PRESENT_POLITE = {"ます", "です", "ません"}
_JA_AUX_NEGATE = {"ない", "なかった", "ません", "ませんでした", "ず"}
_JA_AUX_DESIRE = {"たい", "たかった"}
_JA_AUX_VOICE = {"らしい", "よう", "そう", "みたい"}

_VERB_AUX_PAST_SUFFIXES = ("ました", "でした", "なかった", "ませんでした", "った", "いた", "きた", "した", "んだ", "た")
_VERB_AUX_PRESENT_SUFFIXES = ("ます", "です", "ません", "ない", "る", "う", "く", "ぐ", "す", "む", "ぶ", "ぬ", "つ", "だ")


def _ja_tokenize(sentence: str) -> List[Dict[str, str]]:
    """Sudachi-backed tokenizer with dict_form extraction. Falls back to regex char runs."""
    tokens: List[Dict[str, str]] = []
    sentence_stripped = sentence.rstrip("。！？\n 　").strip()
    if not sentence_stripped:
        return tokens

    if _SUDACHI_AVAILABLE and _SUDACHI_ENGINE is not None:
        try:
            tokenize_result = _SUDACHI_ENGINE.tokenize(sentence_stripped)
            raw_tokens = []
            if hasattr(tokenize_result, "tokens"):
                raw_tokens = list(tokenize_result.tokens)
            elif isinstance(tokenize_result, dict) and "tokens" in tokenize_result:
                raw_tokens = list(tokenize_result["tokens"])
            for t in raw_tokens:
                # Extract fields from dataclass TokenInfo
                surface = ""
                dict_form = ""
                pos_main = ""
                pos_list = []
                if isinstance(t, dict):
                    surface = t.get("surface", "") or ""
                    dict_form = t.get("dictionary_form", surface) or surface
                    pos_main = t.get("pos_main", "") or ""
                    pos_list = t.get("pos", []) or []
                else:
                    surface = getattr(t, "surface", "") or ""
                    dict_form = getattr(t, "dictionary_form", surface) or surface
                    pos_main = getattr(t, "pos_main", "") or ""
                    pos_list = getattr(t, "pos", []) or []
                if surface:
                    # For サ変可能 nouns (する-stem nouns), tag specially so caller can merge
                    pos_joined = " / ".join([str(x) for x in pos_list if x and x != "*"]) if pos_list else pos_main
                    special_tag = ""
                    if "サ変" in pos_joined or (isinstance(pos_list, list) and len(pos_list) >= 3 and "サ変" in str(pos_list[2])):
                        special_tag = "SAHEN"
                    tokens.append({
                        "surface": surface,
                        "dict_form": dict_form,
                        "pos": pos_main,
                        "pos_full": pos_joined,
                        "tag": special_tag,
                    })
            return tokens
        except Exception:
            pass

    # Fallback: regex-split 漢字・ひらがな・カタカナ runs
    parts = re.findall(r"[\u4E00-\u9FFF]+|[\u3040-\u309Fー]+|[\u30A0-\u30FFー]+|[A-Za-z0-9]+|[\W_]", sentence_stripped)
    for p in parts:
        if p.strip():
            tokens.append({"surface": p, "dict_form": p, "pos": "", "pos_full": "", "tag": ""})
    return tokens


def _ja_to_en_sentence(sentence: str) -> str:
    """
    5-stage JA -> EN heuristic with Sudachi token-level dict_form lookup:
      1. Detect sentence-level tense / negate / question flags from tail & surface string
      2. Sudachi tokenize → merge SAHEN noun + する into compound verb dict lookup (surface+noun+する → walk/develop/report etc)
      3. Strip topic/object particles (は/が/を/に/で/の/から/まで) and polite aux (ます/です/ました/でした/ない/ません/ください/して) without adding to lex output
      4. Classify tokens → nouns/verbs/adjectives → reorder SOV→SVO (verb insertion before last noun = object)
      5. a/an/the heuristics + capitalize first alpha + terminator.
    """
    if not sentence.strip():
        return ""

    is_past = any(m in sentence for m in ("ました", "でした", "た。", "た！", "た？", "かった", "なかった", "ませんでした", "だった"))
    is_negate = any(m in sentence for m in ("ない", "なかった", "ません", "ませんでした", "ず", "ん"))
    is_question = sentence.rstrip().endswith(("?", "？", "か。", "か？", "か"))
    is_desire = any(m in sentence for m in ("たい", "たかった"))
    is_polite_request = any(m in sentence for m in ("ください", "下さい", "ませんか", "てください", "でください"))

    tokens_raw = _ja_tokenize(sentence)
    if not tokens_raw:
        return sentence

    # ---------- MERGE SAHEN noun + following する-derived tokens ----------
    # Also: merge noun + して / します / しました into one compound lookup.
    merged_tokens: List[Dict[str, str]] = []
    i = 0
    N = len(tokens_raw)
    while i < N:
        t = tokens_raw[i]
        # Skip punctuation-only tokens (。 、 ・ ！ etc) as surface; handle terminators by flags above
        if t.get("pos") == "補助記号" or (t.get("pos") == "" and re.match(r"^[\W_]+$", t.get("surface", ""))):
            i += 1
            continue
        # SAHEN merge window: look-ahead up to 2 tokens for する variants
        if (t.get("tag") == "SAHEN" or (t.get("pos") == "名詞" and len(t.get("surface", "")) >= 2)) and (i + 1 < N):
            combined_surface = ""
            combined_dict = ""
            skip = 0
            next1 = tokens_raw[i + 1]
            def _is_suru_derived(tt: Dict[str, str]) -> bool:
                df = tt.get("dict_form", "")
                sf = tt.get("surface", "")
                pm = tt.get("pos", "")
                return (df == "する") or (sf in {"し", "さ", "せ", "そ", "す", "して", "した", "してる", "します", "しました", "しません", "しよう"}) or (pm == "動詞" and df == "為る")
            def _is_aux_after(tt: Dict[str, str]) -> bool:
                sf = tt.get("surface", "")
                df = tt.get("dict_form", "")
                pm = tt.get("pos", "")
                return (pm == "助動詞") or (sf in {"ます", "ました", "ません", "ませんでした", "です", "でした", "て", "で", "た", "だ"}) or (df in {"ます", "です", "た", "だ"})
            def _is_please(tt: Dict[str, str]) -> bool:
                sf = tt.get("surface", "")
                return sf in {"ください", "下さい", "ませんか"}
            suru_after = _is_suru_derived(next1)
            aux_after1 = False
            aux_after2 = False
            please_after = False
            if suru_after:
                combined_surface = t["surface"] + next1["surface"]
                combined_dict = t["dict_form"] + "する"
                skip = 2
                # try to also grab an aux after that (ます/ました/です/でした/て/ください)
                if i + 2 < N:
                    aux_after1 = _is_aux_after(tokens_raw[i + 2])
                    if aux_after1:
                        combined_surface = combined_surface + tokens_raw[i + 2]["surface"]
                        skip = 3
                        if i + 3 < N:
                            please_after = _is_please(tokens_raw[i + 3]) or _is_aux_after(tokens_raw[i + 3])
                            if please_after:
                                combined_surface = combined_surface + tokens_raw[i + 3]["surface"]
                                skip = 4
            if suru_after:
                merged_tokens.append({
                    "surface": combined_surface,
                    "dict_form": combined_dict,
                    "pos": t.get("pos", "") + "→動詞",
                    "tag": "MERGED_SAHEN",
                    "_aux_please": bool(please_after),
                })
                i += skip
                continue
        merged_tokens.append({k: v for k, v in t.items()})
        i += 1

    # ---------- Strip particles & aux; collect translated lexical words ----------
    words: List[str] = []
    verb_words: List[str] = []
    adj_words: List[str] = []
    noun_words: List[str] = []

    for tok in merged_tokens:
        surface = tok.get("surface", "")
        dict_form = tok.get("dict_form", "")
        pos = tok.get("pos", "")
        tag = tok.get("tag", "")
        has_please = tok.get("_aux_please", False)

        if surface in _JA_PARTICLES_STRIP:
            continue
        # Explicit polite aux tokens: skip from lex (handled via flags)
        if surface in {"ます", "です", "ました", "でした", "ない", "ません", "ませんでした", "ませんか", "て", "で", "た", "だ", "し", "さ", "せ", "そ"}:
            continue
        if dict_form in {"ます", "です", "た", "だ"} and pos == "助動詞":
            continue
        if surface in {"ください", "下さい"}:
            # polite request: prepend "please" at the end heuristically
            verb_words.insert(0, "please") if "please" not in verb_words else None
            continue
        if has_please and ("please" not in verb_words):
            verb_words.insert(0, "please")

        candidates = [dict_form, surface]
        if tag == "MERGED_SAHEN":
            # Insert surface compound first (most specific 散歩しました/散歩する)
            candidates = [surface, dict_form, surface.replace("ます", "").replace("ました", "").replace("して", "").replace("した", ""), surface.split("し")[0] if "し" in surface else ""]
            candidates = [c for c in candidates if c]
        translation = ""
        for key in candidates:
            if key and key in _DICTIONARY_JA_EN:
                translation = _DICTIONARY_JA_EN[key]
                break
        # Last-ditch: surface lookup
        if (not translation) and surface and surface in _DICTIONARY_JA_EN:
            translation = _DICTIONARY_JA_EN[surface]
        if (not translation) and dict_form and dict_form in _DICTIONARY_JA_EN:
            translation = _DICTIONARY_JA_EN[dict_form]
        if not translation:
            translation = surface

        tc = translation.strip()
        if not tc:
            continue

        # Classify
        is_verb_like = ("動詞" in pos) or (tag == "MERGED_SAHEN") or (dict_form in {"する", "やる", "ある", "いる", "なる"})
        is_adj_like = ("形容詞" in pos) or ("形状詞" in pos)
        if is_verb_like:
            verb_words.append(tc)
        elif is_adj_like:
            adj_words.append(tc)
        else:
            noun_words.append(tc)
        words.append(tc)

    # Deduplicate preserving order
    def _dedup(lst: List[str]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for w in lst:
            if not w or w in seen:
                continue
            seen.add(w)
            out.append(w)
        return out
    noun_words = _dedup(noun_words)
    adj_words = _dedup(adj_words)
    verb_words = _dedup(verb_words)
    words = _dedup(words)

    # SOV→SVO reorder
    reordered: List[str] = []
    for n in noun_words:
        reordered.append(n)
    for a in adj_words:
        if a not in reordered:
            reordered.append(a)

    # Past-tense apply to verbs
    verbs_converted: List[str] = []
    for v in verb_words:
        v_clean = v.strip()
        if not v_clean:
            continue
        v_head = v_clean.split()[0].lower()
        if is_past:
            if v_head == "please":
                verbs_converted.append(v_clean)
                continue
            if v_head in _EN_VERB_PAST:
                v_clean = v_clean.replace(v_clean.split()[0], _EN_VERB_PAST[v_head], 1)
            elif v_head == "be":
                v_clean = v_clean.replace("be", "was", 1)
            elif v_clean.endswith(("ed", "d")):
                pass
            elif v_clean.endswith("e"):
                v_clean = v_clean + "d"
            else:
                v_clean = v_clean + "ed"
        verbs_converted.append(v_clean)

    if verbs_converted:
        main_verb = " ".join(verbs_converted)
        if len(reordered) >= 2:
            # S V O pattern: put verb between 1st subject-noun group & last object-noun
            out = reordered[:-1] + [main_verb] + reordered[-1:]
        elif len(reordered) == 1:
            out = reordered + [main_verb]
        else:
            out = verbs_converted
    else:
        out = reordered

    # Negation injection
    if is_negate and out:
        inserted = False
        be_set = {"is", "are", "was", "were", "am", "be", "has", "have", "had", "will", "would", "can", "could", "should", "shall"}
        for i, w in enumerate(out):
            wl = w.lower().split()[0]
            if wl in be_set:
                out[i] = w + " not"
                inserted = True
                break
        if not inserted:
            verb_heads_past = {"went", "did", "came", "saw", "read", "wrote", "spoke", "ate", "drank", "made", "created", "used", "bought", "sold", "had", "gave", "received", "learned", "studied", "thought", "knew", "understood", "found", "confirmed", "reported", "contacted", "improved", "solved", "developed", "designed", "implemented", "tested", "introduced", "deployed", "launched", "needed", "raised", "lowered", "began", "started", "finished", "continued", "waited", "walked", "ran", "slept", "woke", "applied", "invited", "participated", "joined"}
            for i, w in enumerate(out):
                wl = w.lower()
                w_head = wl.split()[0]
                if (w_head in verb_heads_past) or (w_head + "ed" in verb_heads_past):
                    if is_past:
                        out[i] = "did not " + w_head
                    else:
                        out[i] = "do not " + w_head
                    inserted = True
                    break
            if not inserted and out:
                # Find first non-determiner token and prefix with "not"
                for i, w in enumerate(out):
                    wl = w.lower().split()[0]
                    if wl not in {"a", "an", "the", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "please"}:
                        out.insert(i, "not")
                        inserted = True
                        break

    # Desire: want to
    if is_desire and out:
        for i, w in enumerate(out):
            wl = w.lower()
            if wl and (wl not in {"i", "you", "he", "she", "we", "they", "this", "that", "it", "a", "an", "the", "this", "that"}):
                out.insert(i, "want to")
                break

    result = " ".join(out)
    result = re.sub(r"\s+", " ", result).strip()

    # Definite/indefinite article injection
    def _insert_articles(text: str) -> str:
        vowels = {"a", "e", "i", "o", "u"}
        wlist = text.split()
        out_w: List[str] = []
        known_proper = {"i", "you", "he", "she", "we", "they", "japan", "tokyo", "today", "tomorrow", "yesterday", "now", "here", "there", "this", "that"}
        determiners = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their", "some", "any", "every", "each", "all", "no", "please", "not"}
        be_set = {"is", "are", "was", "were", "am", "be", "has", "have", "had", "can", "could", "will", "would", "shall", "should", "may", "might", "must", "did", "do", "does"}
        business_heads = {"result", "report", "development", "system", "customer", "client", "meeting", "company", "university", "school", "teacher", "student", "data", "file", "email", "telephone", "design", "implementation", "test", "production", "confirmation", "contact", "request", "solution", "improvement", "problem", "challenge", "team", "engineer", "developer", "manager", "application", "website", "software", "hardware", "database", "server", "network", "project", "plan", "schedule", "deadline", "book", "document", "paper", "message", "partner", "vendor", "feature", "function", "technology", "method", "process", "answer", "reason", "cause", "issue", "risk", "benefit", "advantage", "disadvantage", "cost", "price", "value", "quality", "performance", "progress", "status", "information", "content", "detail", "overview", "opinion", "idea", "knowledge", "experience", "skill", "ability", "graduate", "graduation", "course", "class", "lesson", "examination", "grade", "weather", "language", "word", "sentence", "paragraph", "office", "building", "home", "house", "computer", "laptop"}
        for i, w in enumerate(wlist):
            wl = w.lower().rstrip(",")
            prev_w = out_w[-1].lower().rstrip(",") if out_w else ""
            if (wl in business_heads) and (prev_w not in determiners) and (prev_w not in be_set) and (i > 0):
                out_w.append("the")
            else:
                if (wl not in known_proper) and (wl not in determiners) and (wl not in be_set) and (i > 0) and (prev_w not in determiners) and (prev_w not in be_set) and wl and wl[0].isalpha() and (len(wl) > 2):
                    if prev_w in {"work", "works", "as", "with", "at", "to", "from", "of", "for", "in", "on", "about", "through", "walk", "walked", "send", "sent", "have", "had", "need"}:
                        if wl[0] in vowels:
                            out_w.append("an")
                        else:
                            out_w.append("a")
            out_w.append(w)
        return " ".join(out_w)

    eng_letters = len(re.findall(r"[A-Za-z]", result))
    jp_letters = len(re.findall(r"[\u3040-\u30FF\u4E00-\u9FFF]", result))
    if eng_letters >= max(3, jp_letters) and result:
        result = _insert_articles(result)
    # When EN < JP (fallback case), try to convert remaining 日本語 parts via last-lookup surface
    if jp_letters > eng_letters:
        for key in sorted(_DICTIONARY_JA_EN.keys(), key=len, reverse=True):
            if key in result:
                result = result.replace(key, f" {_DICTIONARY_JA_EN[key]} ")
        result = re.sub(r"\s+", " ", result).strip()

    # Capitalize first alpha
    m = re.search(r"[A-Za-z]", result)
    if m:
        idx = m.start()
        result = result[:idx] + result[idx].upper() + result[idx + 1:]

    if is_question:
        result = result.rstrip(".!?。！？") + "?"
    elif result and result[-1] not in (".", "!", "?", "。", "！", "？"):
        result = result + "."

    return re.sub(r"\s+", " ", result).strip()


# -----------------------------------------------------------------------------
# EN → JA heuristic helpers
# -----------------------------------------------------------------------------

_EN_AUX_PAST = {"was", "were", "did", "had", "would", "could", "should", "might", "got"}
_EN_MODALS = {"will", "shall", "can", "may", "must", "would", "could", "should", "might"}
_EN_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "some", "any", "every", "each", "all", "both", "either", "neither", "no", "none", "such"}
_EN_BE_WORDS = {"is", "are", "was", "were", "am", "be", "been", "being"}
_EN_HAVE_WORDS = {"have", "has", "had", "having"}
_EN_DO_WORDS = {"do", "does", "did", "doing", "done"}
_EN_PREPS = {"in", "on", "at", "to", "from", "of", "for", "with", "by", "about", "through", "over", "under", "between", "before", "after", "during", "while", "and", "or", "but", "so", "if", "because"}


def _en_to_ja_sentence(sentence: str) -> str:
    """
    EN -> JA pipeline (2026-10-01 rewritten for 65%+ bar):
      1. word-tokenize sentence, strip punctuation; lowercase + contractions cleanup
      2. BIGRAM MERGE: every+morning / last+year / artificial+intelligence / data+file / development+team / new+system etc
      3. Remove determiners (a/an/the), prepositions (to/of/and etc), DO-support when negated
      4. Dict lookup: try past_rev map → progressive stem → 3rd-sing s-stripping
      5. JA-side CLASSIFY: する-ending dict entries = VERB; い/な-adjective endings = ADJ; rest = NOUN
      6. SOV reorder: Subject(は) + [object nouns…] + を on last object only + VERB/ADJ at END
      7. Polite copula tense: if VERB already has polite suffix (ます/ました/します etc), DONT double; else append suitable です/ます
      8. Terminator 。 or ？
    """
    if not sentence.strip():
        return ""

    is_question = sentence.rstrip().endswith(("?", "？"))
    is_exclaim = sentence.rstrip().endswith("!")

    sentence_clean = sentence.rstrip(".!?。！？\n ")
    if not sentence_clean:
        return sentence

    # Tense flags
    past_markers_here: List[str] = []
    for v in _EN_VERB_PAST.values():
        if re.search(r"\b" + re.escape(v) + r"\b", sentence_clean, re.IGNORECASE):
            past_markers_here.append(v)
    has_past = bool(past_markers_here) or any(re.search(r"\b" + w + r"\b", sentence_clean, re.IGNORECASE) for w in ("was", "were", "did", "had", "last year", "last month", "last week", "yesterday"))
    is_negate = bool(re.search(r"\b(not|never|no|none|don't|doesn't|didn't|can't|won't|wouldn't|shouldn't|couldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't)\b", sentence_clean, re.IGNORECASE))

    # Step 1: word tokenize
    tokens_raw = re.findall(r"[A-Za-z0-9'’]+|[^\w\s]", sentence_clean)
    words: List[str] = []
    for t in tokens_raw:
        if t.isalpha() or "'" in t or "’" in t:
            words.append(t.lower().replace("’", "'"))
        elif t in ("-", "_"):
            continue
        elif t.strip():
            words.append(t)

    # Step 2: BIGRAM merge (two consecutive words form a known compound dict entry)
    # Merge order: longer compounds first
    merged_words: List[str] = []
    idx = 0
    Nw = len(words)
    while idx < Nw:
        bigram = ""
        if idx + 1 < Nw:
            candidate = words[idx] + " " + words[idx + 1]
            if candidate in _DICTIONARY_EN_JA:
                bigram = candidate
            elif candidate.replace(" ", "") in _DICTIONARY_EN_JA:
                bigram = candidate.replace(" ", "")
        if bigram:
            merged_words.append(bigram)
            idx += 2
        else:
            merged_words.append(words[idx])
            idx += 1

    # Step 3/4: dict lookup with skip filters
    past_rev: Dict[str, str] = {v: k for k, v in _EN_VERB_PAST.items()}

    # output tuples: (ja_text, role_S_N_V_A)
    classified: List[Tuple[str, str]] = []  # (ja, role) role ∈ {S,N,V,A}
    for w in merged_words:
        w_base = w
        # Contractions cleanup
        if w.endswith("n't"):
            w_base = w[:-3]
        elif w.endswith("'s") or w.endswith("’s"):
            w_base = w[:-2]
        elif w.endswith("'ve") or w.endswith("'ll") or w.endswith("'d") or w.endswith("'m"):
            w_base = w[:-3]
        w_base = w_base.strip()
        # Skip: determiners, preps, DO-support with negation
        if w_base in _EN_DETERMINERS:
            continue
        if w_base in _EN_PREPS:
            continue
        if w_base in _EN_DO_WORDS and is_negate:
            continue
        # Skip empty-ish
        if not w_base:
            continue

        # Classify EN-side role
        en_role = "N"
        if w_base in {"i", "we"}:
            en_role = "S"
        elif w_base in {"you", "he", "she", "they", "it"}:
            en_role = "S"

        # Dict key resolution
        dict_key = w_base
        # Past-reverse mapping: went → go, did → do etc when already-past
        if (w_base in past_rev) and has_past:
            dict_key = past_rev[w_base]
        # Also: unify the bigram key if it exists directly
        if (not dict_key) and w_base.endswith("ing"):
            dict_key = w_base[:-3]
        if dict_key and (dict_key not in _DICTIONARY_EN_JA) and w_base.endswith("s") and not w_base.endswith("ss"):
            stripped = w_base[:-1]
            if stripped in _DICTIONARY_EN_JA:
                dict_key = stripped
        if dict_key and (dict_key not in _DICTIONARY_EN_JA) and w_base.endswith("ed"):
            stripped = w_base[:-2]
            if stripped in _DICTIONARY_EN_JA:
                dict_key = stripped
            elif stripped.endswith("e") and (stripped[:-1] in _DICTIONARY_EN_JA):
                dict_key = stripped[:-1]

        ja_word = ""
        for key_candidate in (dict_key, w_base, w):
            if key_candidate and (key_candidate in _DICTIONARY_EN_JA):
                ja_word = _DICTIONARY_EN_JA[key_candidate]
                break
        if not ja_word:
            ja_word = w_base  # fallback: keep EN (can't translate)

        # Normalize dict entries with "/" separator → pick FIRST alternative with longest length (most specific)
        if ja_word and "/" in ja_word:
            alts = [a.strip() for a in ja_word.split("/") if a.strip()]
            if alts:
                alts_sorted = sorted(alts, key=lambda x: len(x.replace(" ", "")), reverse=True)
                ja_word = alts_sorted[0]
        # Also normalize (xxx) parenthetical notes
        ja_word = re.sub(r"\s*\([^)]*\)\s*", "", ja_word).strip()

        # Classify JA-side role (S/N/V/A)
        # VERB detection via する-endings / ます-endings / し-endings / 行く etc
        ja_word_clean = ja_word.strip()
        jrole = "N"
        if en_role == "S":
            jrole = "S"
        elif ja_word_clean:
            verb_endings = (
                "する", "します", "しました", "して", "した", "している",
                "やる", "行く", "行きます", "行きました",
                "来る", "来ます", "来ました",
                "見る", "見ます", "見ました",
                "聞く", "聞きます", "聞きました",
                "読む", "読みます", "読みました",
                "書く", "書きます", "書きました",
                "話す", "話します", "話しました",
                "食べる", "食べます", "食べました",
                "飲む", "飲みます", "飲みました",
                "作る", "作ります", "作りました",
                "使う", "使います", "使いました",
                "送る", "送ります", "送りました",
                "学ぶ", "学びます", "学びました",
                "考える", "考えます", "考えました",
                "勉強する", "勉強します", "勉強しました",
                "研究する", "開発する", "設計する", "実装する",
                "テストする", "導入する", "活用する", "利用する",
                "確認する", "連絡する", "報告する", "依頼する", "対応する",
                "改善する", "解決する", "参加する", "申し込む", "応募する",
                "準備する", "調査する", "送信する",
                "歩く", "歩きます", "歩きました",
                "走る", "走ります", "走りました",
                "寝る", "寝ます", "寝ました",
                "起きる", "起きます", "起きました",
                "待つ", "待ちます", "待ちました",
                "始める", "終える", "続ける",
                "買う", "売る", "与える", "受け取る",
                "知る", "分かる", "理解する",
                "見つける", "招く", "適用する",
            )
            adj_endings = (
                "い", "かった", "くて",
                "良い", "悪い", "大きい", "小さい", "新しい", "古い",
                "多い", "少ない", "速い", "遅い", "強い", "弱い",
                "高い", "低い", "広い", "狭い",
                "美しい", "可愛い",
                "便利", "不便", "重要", "必要", "可能", "不可能",
            )
            if any(ja_word_clean.endswith(ve) for ve in verb_endings):
                jrole = "V"
            elif any(ja_word_clean.endswith(ae) for ae in adj_endings) and not ja_word_clean.endswith("する"):
                jrole = "A"
            else:
                jrole = "N"

        ja_stripped = ja_word.strip()
        if not ja_stripped:
            continue  # "" dict entry for particles/prepositions: skip entirely
        classified.append((ja_stripped, jrole))

    if not classified:
        return sentence

    # Step 6: SOV reorder
    subject_tokens: List[str] = []
    va_tokens: List[str] = []
    object_tokens: List[str] = []

    for (jw, jr) in classified:
        if jr == "S":
            subject_tokens.append(jw)
        elif jr == "V" or jr == "A":
            va_tokens.append(jw)
        else:
            object_tokens.append(jw)

    # Build the final string
    parts: List[str] = []
    if subject_tokens:
        parts.append("".join(subject_tokens) + "は")
    # Adjectives (A) stick BEFORE object noun they modify if VA count >=2; else they go at end too
    adj_tokens = [v for v in va_tokens if any(v.endswith(ae) for ae in (
        "い", "良い", "悪い", "大きい", "小さい", "新しい", "古い", "多い", "少ない",
        "速い", "遅い", "強い", "弱い", "高い", "低い", "広い", "狭い",
        "美しい", "可愛い", "便利", "不便", "重要", "必要", "可能", "不可能",
    )) and not any(v.endswith(ve) for ve in (
        "する", "します", "しました", "して", "した", "している",
    ))]
    verb_tokens = [v for v in va_tokens if v not in adj_tokens]

    # Place adjectives BEFORE the object they likely modify (if single object)
    if adj_tokens and object_tokens:
        # insert adj before the DIRECT object (= last object token)
        direct_obj = object_tokens[-1]
        others = object_tokens[:-1]
        object_tokens = others + adj_tokens + [direct_obj]

    # Object particle logic: ONLY the LAST object gets を (direct object particle).
    # Earlier "object" tokens are adverbial: use に only if explicit location keyword, else NO particle (no more で-spam!)
    if object_tokens:
        if len(object_tokens) == 1:
            parts.append(object_tokens[0] + "を")
        else:
            for i, ot in enumerate(object_tokens):
                if i == len(object_tokens) - 1:
                    parts.append(ot + "を")
                else:
                    # keyword-location → に; else plain noun, no particle
                    if any(x in ot for x in ("東京", "日本", "部屋", "会社", "学校", "大学", "家", "店", "駅", "空港", "公園", "郵便局", "銀行", "病院", "市役所")):
                        parts.append(ot + "に")
                    elif any(x in ot for x in ("毎朝", "毎日", "毎週", "毎月", "毎年", "今日", "明日", "昨日", "今年", "去年", "来年", "今朝", "今晩", "今週", "今月", "先週", "先月", "昨年", "来週", "来月", "朝", "昼", "晩", "夜")):
                        # Japanese standalone time adverbs don't take particles usually — NO particle
                        parts.append(ot)
                    else:
                        # Default: NO particle (previously was "で" which was wrong everywhere)
                        parts.append(ot)

    if verb_tokens:
        parts.append("".join(verb_tokens))
    elif adj_tokens and not verb_tokens:
        # Adjective-predicate sentence (no verb): append adjectives at END as predicate (before です)
        parts.append("".join(adj_tokens))

    # Step 7: tense/polite copula
    # Detect if the LAST token already carries polite suffix: if yes, NO double-polite
    last_chunk = parts[-1] if parts else ""
    has_polite_suffix = bool(
        last_chunk
        and any(last_chunk.endswith(pe) for pe in (
            "ます", "ました", "ません", "ませんでした",
            "します", "しました", "しません", "しませんでした",
            "です", "でした", "ではありません", "ではありませんでした",
            "きます", "きました", "けます", "けました",
            "びます", "びました", "みます", "みました",
            "きます", "きました",
        ))
    )

    tail_copula = ""
    has_verb = bool(verb_tokens)
    verb_tail_stem = ""
    if verb_tokens:
        last_vt = verb_tokens[-1]
        if last_vt.endswith("する"):
            verb_tail_stem = last_vt[:-2]  # strip する

    if not has_polite_suffix and has_verb:
        # If last verb ends in する → conjugated します/しました/しません/しませんでした
        if verb_tail_stem:
            # Rewrite the last verb_token by replacing する → correct politeness
            new_last = ""
            if is_negate and has_past:
                new_last = verb_tail_stem + "しませんでした"
            elif is_negate:
                new_last = verb_tail_stem + "しません"
            elif has_past:
                new_last = verb_tail_stem + "しました"
            else:
                new_last = verb_tail_stem + "します"
            verb_tokens = verb_tokens[:-1] + [new_last]
            has_polite_suffix = True  # prevent double-appending
        else:
            if is_negate and has_past:
                tail_copula = "ませんでした"
            elif is_negate:
                tail_copula = "ません"
            elif has_past:
                tail_copula = "ました"
            else:
                tail_copula = "ます"
    elif not has_polite_suffix and not has_verb:
        # Adjective / noun predicate
        if is_negate and has_past:
            tail_copula = "ではありませんでした"
        elif is_negate:
            tail_copula = "ではありません"
        elif has_past:
            tail_copula = "でした"
        else:
            tail_copula = "です"

    output = "".join(parts)
    if output and tail_copula:
        output = output + tail_copula

    # Step 8: terminator
    if is_question:
        output = output.rstrip("。！？") + "？"
    elif output and not output.rstrip().endswith(("。", "！", "？")):
        output = output + "。"

    return output.strip()


def _dictionary_translate(text: str, source_lang: str, target_lang: str) -> str:
    src = _normalize_lang(source_lang)
    tgt = _normalize_lang(target_lang)

    sentences = _split_sentences(text, source_lang)
    translated_sentences: List[str] = []

    if src == "ja" and tgt == "en":
        for sentence in sentences:
            translated_sentences.append(_ja_to_en_sentence(sentence))
        joined = " ".join(translated_sentences).strip()
        return re.sub(r"\s+", " ", joined)

    if src == "en" and tgt == "ja":
        for sentence in sentences:
            translated_sentences.append(_en_to_ja_sentence(sentence))
        return "".join(translated_sentences).strip()

    # Fallback for unsupported pairs: old-style direct replace dictionary
    table = _DICTIONARY_JA_EN if src == "ja" else _DICTIONARY_EN_JA
    for sentence in sentences:
        s = sentence
        sorted_keys = sorted(table.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if src == "ja":
                if key in s:
                    s = s.replace(key, f" {table[key]} ")
            else:
                pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
                s = pattern.sub(table[key], s)
        s = re.sub(r"\s+", " ", s).strip()
        translated_sentences.append(s)
    return " ".join(translated_sentences).strip()


def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "en",
    domain: Optional[str] = None,
    force_fallback: bool = False,
    custom_glossary: Optional[List[Dict[str, str]]] = None,
    examples: Optional[List[Dict[str, str]]] = None,
) -> TranslationResult:
    if not isinstance(text, str) or not text.strip():
        empty_qs = _quality.score_translation("", "", source_lang, target_lang)
        return TranslationResult(
            source_text=text or "", translated_text="",
            source_lang=source_lang, target_lang=target_lang,
            engine="none", quality_score=empty_qs, domain=domain,
            notes={"reason": "empty text"},
        )

    src = _normalize_lang(source_lang) if source_lang != "auto" else None
    tgt = _normalize_lang(target_lang)

    if src is None:
        ja_chars = len(re.findall(r"[\u3040-\u30FF\u4E00-\u9FFF]", text))
        en_chars = len(re.findall(r"[A-Za-z]", text))
        if ja_chars >= en_chars:
            src = "ja"
        else:
            src = "en"
        source_lang_eff = src
    else:
        source_lang_eff = src

    if src == tgt:
        qs = _quality.score_translation(text, text, source_lang_eff, tgt, glossary_hits=0)
        return TranslationResult(
            source_text=text, translated_text=text,
            source_lang=source_lang_eff, target_lang=tgt,
            engine="passthrough", glossary_hits=0,
            quality_score=qs, domain=domain,
            notes={"reason": "same source/target language"},
        )

    preprocessed, glossary_hits = _glossary.apply_glossary(text, source_lang_eff, tgt, domain)
    protected_terms: List[Tuple[str, str]] = []
    custom_hits = 0
    for pair in custom_glossary or []:
        src_term = (pair.get("ja") or "") if source_lang_eff == "ja" else (pair.get("en") or "")
        tgt_term = (pair.get("en") or "") if source_lang_eff == "ja" else (pair.get("ja") or "")
        if not src_term.strip() or not tgt_term.strip():
            continue
        placeholder = f"KANAMEGLOSSARYTOKEN{len(protected_terms):04d}END"
        if source_lang_eff == "ja":
            if src_term in preprocessed:
                preprocessed = preprocessed.replace(src_term, placeholder)
                protected_terms.append((placeholder, tgt_term))
                custom_hits += 1
        else:
            pattern = re.compile(r"(?<!\w)" + re.escape(src_term) + r"(?!\w)", re.IGNORECASE)
            preprocessed, count = pattern.subn(placeholder, preprocessed)
            if count:
                protected_terms.append((placeholder, tgt_term))
                custom_hits += count

    engine, key = _detect_engine()
    engine_used = "dictionary"
    notes: Dict[str, Any] = {
        "glossary_pairs_applied": glossary_hits,
        "custom_glossary_hits": custom_hits,
        "few_shot_examples_used": len(examples or []) if not force_fallback else 0,
    }

    if force_fallback:
        notes["llm_available"] = False
        notes["force_fallback"] = True
        translated = _dictionary_translate(preprocessed, source_lang_eff, tgt)
    elif engine and key:
        try:
            sys_prompt = _system_prompt(source_lang_eff, tgt, domain)
            prompt_text = preprocessed
            if examples:
                examples_text = "\n".join(
                    f"Source: {item.get('source', '')}\nTarget: {item.get('target', '')}"
                    for item in examples
                    if item.get("source") and item.get("target")
                )
                if examples_text:
                    prompt_text = f"Reference examples (match terminology and style):\n{examples_text}\n\nText to translate:\n{preprocessed}"
            translated = _call_llm(engine, key, sys_prompt, prompt_text)
            engine_used = engine
            notes["llm_model_used"] = {
                "claude": "claude-3-haiku-20240307",
                "gemini": os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash"),
            }.get(engine, "unknown")
        except Exception as e:
            notes["llm_error"] = str(e)
            translated = _dictionary_translate(preprocessed, source_lang_eff, tgt)
            engine_used = "dictionary"
    else:
        notes["llm_available"] = False
        translated = _dictionary_translate(preprocessed, source_lang_eff, tgt)

    if not translated.strip():
        translated = preprocessed

    for placeholder, target_term in protected_terms:
        translated = translated.replace(placeholder, target_term)

    qs = _quality.score_translation(
        text, translated, source_lang_eff, tgt,
        glossary_hits=glossary_hits + custom_hits,
        source_terms_guess=max(1, glossary_hits + custom_hits + 2),
    )

    return TranslationResult(
        source_text=text,
        translated_text=translated,
        source_lang=source_lang_eff,
        target_lang=tgt,
        engine=engine_used,
        glossary_hits=glossary_hits + custom_hits,
        quality_score=qs,
        domain=domain,
        notes=notes,
    )


def translate_batch(
    texts: List[str],
    source_lang: str = "auto",
    target_lang: str = "en",
    domain: Optional[str] = None,
    force_fallback: bool = False,
    custom_glossary: Optional[List[Dict[str, str]]] = None,
    examples: Optional[List[Dict[str, str]]] = None,
) -> List[TranslationResult]:
    return [
        translate_text(
            t, source_lang, target_lang, domain, force_fallback,
            custom_glossary=custom_glossary, examples=examples,
        )
        for t in texts
    ]


def available_engines() -> Dict[str, Any]:
    engine, _ = _detect_engine()
    configured: List[str] = []
    for k in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(k)
        if v and v.strip() and not v.strip().startswith(("your", "sk-xxx", "xxx", "<")):
            configured.append(k)
    return {
        "active_engine": engine or "dictionary",
        "configured_keys": configured,
        "fallback_available": True,
        "domains": _glossary.list_domains(),
    }
