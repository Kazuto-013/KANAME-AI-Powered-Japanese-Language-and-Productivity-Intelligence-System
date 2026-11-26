"""
Labeled Japanese sample data used to train the built-in classifiers.

5 categories, ~5 sentences each for a total of ~25 labeled samples.
These are intentionally small to stay self-contained without external datasets;
the goal is to demonstrate end-to-end scikit-learn pipeline integration
with a Sudachi tokenizer wrapper, not to be a production-grade classifier.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


TRAIN_CATEGORIES: Dict[str, List[str]] = {
    "technology": [
        "Pythonで機械学習モデルを構築し、精度を検証しました。",
        "データベースのパフォーマンスチューニングを実施し、クエリ時間を短縮しました。",
        "新しいWeb APIを開発し、バックエンドシステムと連携しています。",
        "クラウド環境にサーバーをデプロイし、運用監視を開始しました。",
        "自然言語処理のライブラリSudachiを使って形態素解析を行いました。",
    ],
    "business": [
        "今月の売上は先月比12%増となり、目標を達成しました。",
        "新規顧客との商談を実施し、来期の取引に向けて合意しました。",
        "半期決算の準備を進めており、各部門から資料を回収しています。",
        "マーケティングキャンペーンの効果を測定し、ROIを算出しました。",
        "サプライヤーと価格交渉を行い、調達コストを5%削減しました。",
    ],
    "education": [
        "大学の講義でデータサイエンスの基礎を学び、レポートを提出しました。",
        "新入社員向けの研修プログラムを作成し、来月から実施します。",
        "オンライン講座を受講し、Pythonプログラミングの認定資格を取得しました。",
        "学生の理解度を確認するため、小テストを実施して採点しました。",
        "研究室のゼミで論文を発表し、教授からフィードバックを受けました。",
    ],
    "health": [
        "定期健診の結果、血圧とコレステロール値が改善していました。",
        "朝晩30分のウォーキングを続け、体重が2キロ減少しました。",
        "病院で検査を受け、処方箋をもらって薬を服用しています。",
        "栄養バランスの取れた食事を心がけ、睡眠時間を7時間確保しています。",
        "ストレッチと瞑想を習慣にし、肩こりと疲労が軽減されました。",
    ],
    "lifestyle": [
        "週末に友人と新宿で映画を観て、美味しいラーメンを食べました。",
        "引っ越したばかりのアパートのインテリアを自分好みに整えました。",
        "趣味の写真撮影で、秋の京都を巡って紅葉の写真をたくさん撮りました。",
        "近所のカフェで読書をしながら、ゆったりとした午後を過ごしました。",
        "来月の旅行計画を立て、ホテルと新幹線の予約を済ませました。",
    ],
}


INTENT_TRAIN: List[Tuple[str, str]] = [
    # ---- greeting (挨拶) ----
    ("こんにちは、お元気ですか？", "greeting"),
    ("おはようございます。今日もよろしくお願いします。", "greeting"),
    ("はじめまして、山田と申します。", "greeting"),
    ("お疲れ様です。本日もよろしくお願いいたします。", "greeting"),
    ("こんばんは、夜分に失礼いたします。", "greeting"),
    ("お世話になっております。いつもありがとうございます。", "greeting"),
    ("あけましておめでとうございます。本年もよろしくお願いします。", "greeting"),
    ("初めまして、どうぞよろしくお願いいたします。", "greeting"),
    ("おやすみなさい、また明日お会いしましょう。", "greeting"),
    ("ご無沙汰しております。お変わりありませんか？", "greeting"),
    # ---- question (質問) ----
    ("すみません、この資料の内容について教えていただけますか？", "question"),
    ("次の会議は何時からどこで開催されますか？", "question"),
    ("なぜこの処理が遅くなっているのでしょうか？", "question"),
    ("この機能の使い方を教えてください。", "question"),
    ("会議の議事録はどこに保存されていますか？", "question"),
    ("納期はいつ頃になりますか？", "question"),
    ("この件について、どなたに確認すればよろしいですか？", "question"),
    ("請求書の再発行は可能でしょうか？", "question"),
    ("どの資料を先に確認すればよいですか？", "question"),
    ("エラーの原因は何ですか？", "question"),
    # ---- request (依頼) ----
    ("来週までにこのレポートを仕上げてください。", "request"),
    ("コードレビューをお願いしてもよろしいでしょうか？", "request"),
    ("打ち合わせの日程を調整していただけますと幸いです。", "request"),
    ("恐れ入りますが、この書類の確認をお願いできますか？", "request"),
    ("お手数ですが、明日までにご返信いただけますでしょうか？", "request"),
    ("参考資料の送付をお願いいたします。", "request"),
    ("こちらの修正をご検討いただけますと幸いです。", "request"),
    ("書類に署名をお願いしてもよろしいでしょうか？", "request"),
    ("至急ご対応いただけますようお願い申し上げます。", "request"),
    ("会議の日程を再調整していただけますか？", "request"),
    # ---- command (命令) ----
    ("書類に不備がありますので、至急修正してください。", "command"),
    ("ここに署名して、捺印の上ご返送ください。", "command"),
    ("会議室に移動してください。", "command"),
    ("このファイルを削除してください。", "command"),
    ("明日の会議に必ず出席してください。", "command"),
    ("報告書を金曜日までに提出してください。", "command"),
    ("その資料をコピーして全員に配布してください。", "command"),
    ("パスワードを直ちに変更してください。", "command"),
    ("会議室の予約をキャンセルしてください。", "command"),
    ("ここに印鑑を押してください。", "command"),
    # ---- opinion (意見・感想) ----
    ("新しいデザインは非常に洗練されていて良いと思います。", "opinion"),
    ("個人的には、この方針は長期的にリスクが高いと考えます。", "opinion"),
    ("私たちのチームは今期、大変素晴らしい成果を上げました。", "opinion"),
    ("この提案は非常に現実的だと思います。", "opinion"),
    ("私はこの計画に賛成です。", "opinion"),
    ("その判断は少しリスクが高いと感じます。", "opinion"),
    ("もっと時間をかけるべきだと考えます。", "opinion"),
    ("このデザインはとても使いやすいと感じました。", "opinion"),
    ("今回の結果は十分に満足できるものだと思います。", "opinion"),
    ("正直なところ、この方針には反対です。", "opinion"),
    # ---- other (その他) ----
    ("今日は天気が良いですね。", "other"),
    ("添付ファイルを確認しました。", "other"),
    ("返信が遅くなり、申し訳ありません。", "other"),
    ("本日は一日中雨が降っていました。", "other"),
    ("資料を受け取りました。ありがとうございます。", "other"),
    ("昼食は12時からにします。", "other"),
    ("電車が10分ほど遅延しています。", "other"),
    ("コーヒーを飲みながら作業を進めました。", "other"),
    ("来週は出張で東京に行きます。", "other"),
    ("会議は予定どおり終了しました。", "other"),
]


def get_labeled_texts() -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for cat, sentences in TRAIN_CATEGORIES.items():
        for s in sentences:
            texts.append(s)
            labels.append(cat)
    return texts, labels


def get_intent_labeled() -> Tuple[List[str], List[str]]:
    texts = [t for t, _ in INTENT_TRAIN]
    labels = [l for _, l in INTENT_TRAIN]
    return texts, labels


def category_names() -> List[str]:
    return sorted(TRAIN_CATEGORIES.keys())


def intent_names() -> List[str]:
    return sorted({label for _, label in INTENT_TRAIN})
