"""
Japanese NLP engine powered by SudachiPy.

Provides tokenization, morphological analysis, POS-based filtering,
reading extraction, TF-IDF style keyword extraction, and rule-based
sentiment analysis for Japanese text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

from sudachipy import Dictionary, Tokenizer, dictionary


_JA_STOPWORDS = frozenset({
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
    "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと",
    "として", "い", "や", "れる", "など", "なっ", "なり", "ます",
    "ん", "なら", "によって", "により", "おり", "より", "による",
    "ず", "ながら", "ので", "のみ", "ため", "これ", "それ", "あれ",
    "この", "その", "あの", "どの", "ここ", "そこ", "あそこ",
    "私", "あなた", "彼", "彼女", "たち", "自分", "もの", "事",
    "方", "何", "どこ", "いつ", "だれ", "どれ", "どの", "どう",
    "すべて", "全部", "部分", "ただ", "そして", "しかし", "けれど",
    "が", "けど", "です", "だ", "である", "また", "さらに", "特に",
    "主に", "ついて", "と共に", "のある", "まま", "ほど", "くらい",
    "なお", "且つ", "又", "或いは", "若しくは", "及び", "並びに",
    "その他", "その後", "前述", "上記", "下記", "中", "時", "後",
    "前", "内", "外", "他", "以上", "以下", "以内", "以後", "以前",
    "毎", "各", "他", "今", "後で", "前に", "際に", "において",
    "にかけて", "につれて", "ともに", "一方", "他方", "それぞれ",
    "一つ", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "円", "月", "日", "年", "回", "目", "的", "性",
    "化", "者", "達", "過ぎ", "ばかり", "だけ", "しまう", "おく",
    "くださる", "なさる", "下さる", "致す", "申す", "参る", "いただく",
    "ください", "ませ", "ました", "ません", "でしょう", "だろう",
    "らしい", "よう", "みたい", "そう", "はず", "わけ", "つもり",
    "たい", "ほしい", "がたい", "やすい", "にくい", "べき",
    "ものの", "くせに", "のに", "せいで", "おかげで", "ことには",
    "ことか", "ものか", "もの", "ところ", "はじめ", "はずか",
    "今日", "明日", "昨日", "毎日", "毎週", "毎月", "毎年",
})


_POSITIVE_WORDS = {
    "すばらしい", "素晴らしい", "すごい", "凄い", "素敵", "すてき",
    "よい", "良い", "いい", "良かった", "よかった", "優れる", "優れて",
    "上手", "じょうず", "上手い", "うまい", "天才的", "最高", "さいこう",
    "大好き", "だいすき", "好き", "すき", "愛する", "あいする",
    "誇り", "ほこり", "満足", "まんぞく", "幸せ", "しあわせ",
    "嬉しい", "うれしい", "楽しい", "たのしい", "面白い", "おもしろい",
    "心地よい", "ここちよい", "快適", "かいてき", "安全", "あんぜん",
    "安心", "あんしん", "信頼", "しんらい", "感動", "かんどう",
    "感謝", "かんしゃ", "ありがとう", "おめでとう", "お疲れ様",
    "つかれさま", "頑張る", "がんばる", "努力", "どりょく",
    "成功", "せいこう", "達成", "たっせい", "成長", "せいちょう",
    "進歩", "しんぽ", "改善", "かいぜん", "革新", "かくしん",
    "効率的", "こうりつてき", "生産的", "せいさんてき", "創造的",
    "そうぞうてき", "柔軟", "じゅうなん", "迅速", "じんそく",
    "丁寧", "ていねい", "親切", "しんせつ", "優しい", "やさしい",
    "強い", "つよい", "豊か", "ゆたか", "豊富", "ほうふ",
    "安定", "あんてい", "確実", "かくじつ", "完璧", "かんぺき",
    "綺麗", "きれい", "美しい", "うつくしい", "立派", "りっぱ",
    "期待", "きたい", "希望", "きぼう", "夢", "ゆめ",
    "光栄", "こうえい", "栄光", "えいこう", "祝福", "しゅくふく",
    "あかるい", "明るい", "たのもしい", "頼もしい", "すがすがしい",
    "清々しい", "賢い", "かしこい", "利口", "りこう",
    "高品質", "こうひんしつ", "高級", "こうきゅう", "豪華", "ごうか",
    "格好", "かっこう", "素早い", "すばやい", "鋭い", "するどい",
    "適切", "てきせつ", "的確", "てっかく", "有効", "ゆうこう",
    "有益", "ゆうえき", "価値", "かち", "成果", "せいか",
    "実り", "みのり", "豊作", "ほうさく", "繁盛", "はんじょう",
    "人気", "にんき", "有名", "ゆうめい", "名高い", "なだかい",
    "親身", "しんみ", "誠実", "せいじつ", "真面目", "まじめ",
    "情熱", "じょうねつ", "熱意", "ねつい", "誠意", "せいい",
    "積極的", "せっきょくてき", "意欲的", "いよくてき", "自主的",
    "自主", "自由", "じゆう", "平等", "びょうどう", "平和", "へいわ",
    "尊敬", "そんけい", "尊重", "そんちょう", "称賛", "しょうさん",
    "賞賛", "しょうさん", "称揚", "しょうよう", "推薦", "すいせん",
    "推奨", "すいしょう", "感心", "かんしん", "感銘", "かんめい",
    "驚嘆", "きょうたん", "嘆賞", "たんしょう", "満喫", "まんきつ",
    "堪能", "たんのう", "陶酔", "とうすい", "酔いしれ",
}


_NEGATIVE_WORDS = {
    "悪い", "わるい", "最悪", "さいあく", "最低", "さいてい",
    "ひどい", "酷い", "大嫌い", "だいきらい", "嫌い", "きらい",
    "嫌な", "いやな", "いや", "不快", "ふかい", "不安", "ふあん",
    "心配", "しんぱい", "悲しい", "かなしい", "辛い", "つらい",
    "苦しい", "くるしい", "痛い", "いたい", "痛み", "いたみ",
    "怒る", "おこる", "怒り", "いかり", "腹立たしい", "はらだたしい",
    "立腹", "りっぷく", "怒鳴る", "どなる", "憎い", "にくい",
    "憎む", "にくむ", "憎しみ", "にくしみ", "嫌がる", "いやがる",
    "失望", "しつぼう", "絶望", "ぜつぼう", "後悔", "こうかい",
    "反省", "はんせい", "恥じる", "はじる", "恥", "はじ",
    "恥ずかしい", "はずかしい", "照れ", "てれ", "焦る", "あせる",
    "焦り", "あせり", "心細い", "こころぼそい", "寂しい", "さびしい",
    "孤独", "こどく", "退屈", "たいくつ", "つまらない", "詰まらない",
    "だるい", "怠い", "疲れた", "つかれた", "疲労", "ひろう",
    "飽きた", "あきた", "嫌気", "いやけ", "面倒", "めんどう",
    "億劫", "おっくう", "困る", "こまる", "困難", "こんなん",
    "迷惑", "めいわく", "被害", "ひがい", "損害", "そんがい",
    "損", "そん", "損失", "そんしつ", "失敗", "しっぱい",
    "間違い", "まちがい", "ミス", "過ち", "あやまち", "違反",
    "違反", "いはん", "違法", "いほう", "罪", "つみ",
    "罰", "ばつ", "刑罰", "けいばつ", "処罰", "しょばつ",
    "汚い", "きたない", "汚れ", "よごれ", "臭い", "くさい",
    "うるさい", "五月蝿い", "暑い", "あつい", "寒い", "さむい",
    "まずい", "不味い", "まずしい", "貧しい", "乏しい", "とぼしい",
    "不足", "ふそく", "欠乏", "けつぼう", "欠け", "かけ",
    "欠陥", "けっかん", "不完全", "ふかんぜん", "不十分", "ふじゅうぶん",
    "不適切", "ふてきせつ", "不正確", "ふせいかく", "無効", "むこう",
    "無駄", "むだ", "無意味", "むいみ", "無能", "むのう",
    "怠慢", "たいまん", "遅刻", "ちこく", "遅延", "ちえん",
    "渋滞", "じゅうたい", "混雑", "こんざつ", "汚染", "おせん",
    "破損", "はそん", "故障", "こしょう", "事故", "じこ",
    "災害", "さいがい", "惨事", "さんじ", "悲劇", "ひげき",
    "脅迫", "きょうはく", "恐喝", "きょうかつ", "暴力", "ぼうりょく",
    "暴言", "ぼうげん", "中傷", "ちゅうしょう", "誹謗", "ひぼう",
    "侮辱", "ぶじょく", "差別", "さべつ", "偏見", "へんけん",
    "不公平", "ふこうへい", "不当", "ふとう", "違法",
    "汚職", "おしょく", "収賄", "しゅうわい", "賄賂", "わいろ",
    "詐欺", "さぎ", "窃盗", "せっとう", "盗難", "とうなん",
    "犯罪", "はんざい", "有罪", "ゆうざい", "前科", "ぜんか",
    "苦情", "くじょう", "文句", "もんく", "抗議", "こうぎ",
    "反対", "はんたい", "拒否", "きょひ", "拒絶", "きょぜつ",
    "拒む", "こばむ", "否定", "ひてい", "否認", "ひにん",
    "批判", "ひはん", "批評", "ひひょう", "非難", "ひなん",
    "難", "なん", "不満", "ふまん", "不機嫌", "ふきげん",
    "情けない", "なさけない", "みっともない", "見苦しい", "みぐるしい",
    "未熟", "みじゅく", "未熟な", "下手", "へた", "苦手", "にがて",
    "不得手", "ふえて", "不手際", "ふてぎわ", "まず", "不味",
    "下らない", "くだらない", "退屈な", "つまらぬ", "不毛", "ふもう",
    "無能な", "不細工", "ぶさいく", "醜い", "みにくい",
    "臆病", "おくびょう", "卑怯", "ひきょう", "怠け者", "なまけもの",
    "横着", "おうちゃく", "ずるい", "狡い", "ずる賢い",
    "偽り", "いつわり", "嘘", "うそ", "欺瞞", "ぎまん",
    "不実", "ふじつ", "不信", "ふしん", "疑惑", "ぎわく",
}


_SUDACHI_DICT: Optional[Dictionary] = None

# JLPT vocabulary levels (N5 = beginner/easiest, N1 = most advanced).
# Dictionary-form based mapping. Falls back to: kanji complexity heuristic,
# then None when unknown.  Readings are also mapped for pure-kana tokens.

_JLPT_DICT_FORM: Dict[str, str] = {
    # ---- N5 (beginner, ~800 words) ----
    "私": "N5", "あなた": "N5", "彼": "N5", "彼女": "N5", "人": "N5", "友達": "N5",
    "先生": "N5", "学生": "N5", "学校": "N5", "大学": "N5", "会社": "N5", "仕事": "N5",
    "本": "N5", "手紙": "N5", "新聞": "N5", "日": "N5", "月": "N5", "年": "N5", "今日": "N5",
    "明日": "N5", "昨日": "N5", "毎日": "N5", "毎週": "N5", "毎月": "N5", "毎年": "N5",
    "今": "N5", "前": "N5", "後": "N5", "時": "N5", "時間": "N5", "分": "N5", "朝": "N5",
    "昼": "N5", "晩": "N5", "夜": "N5", "今日": "N5", "行く": "N5", "来る": "N5", "帰る": "N5",
    "走る": "N5", "歩く": "N5", "見る": "N5", "聞く": "N5", "読む": "N5", "書く": "N5", "話す": "N5",
    "食べる": "N5", "飲む": "N5", "買う": "N5", "作る": "N5", "する": "N5", "やる": "N5",
    "です": "N5", "ます": "N5", "だ": "N5", "ある": "N5", "いる": "N5", "ない": "N5",
    "いい": "N5", "よい": "N5", "良い": "N5", "悪い": "N5", "大きい": "N5", "小さい": "N5",
    "高い": "N5", "安い": "N5", "安価": "N5", "長い": "N5", "短い": "N5", "新しい": "N5",
    "古い": "N5", "暑い": "N5", "寒い": "N5", "熱い": "N5", "冷たい": "N5", "面白い": "N5",
    "面白くない": "N5", "美味しい": "N5", "まずい": "N5", "早い": "N5", "速い": "N5",
    "遅い": "N5", "多い": "N5", "少ない": "N5", "遠い": "N5", "近い": "N5", "明るい": "N5",
    "暗い": "N5", "広い": "N5", "狭い": "N5", "大切": "N5", "便利": "N5", "不便": "N5",
    "元気": "N5", "病気": "N5", "暇": "N5", "忙しい": "N5", "好き": "N5", "嫌い": "N5",
    "綺麗": "N5", "きれい": "N5", "丁寧": "N5", "静か": "N5", "賑やか": "N5", "優しい": "N5",
    "日本": "N5", "中国": "N5", "アメリカ": "N5", "イギリス": "N5", "韓国": "N5", "東京": "N5",
    "英語": "N5", "日本語": "N5", "中国語": "N5", "勉強": "N5", "練習": "N5", "質問": "N5",
    "答える": "N5", "問題": "N5", "答え": "N5", "家族": "N5", "父": "N5", "母": "N5",
    "兄": "N5", "姉": "N5", "弟": "N5", "妹": "N5", "子供": "N5", "男": "N5", "女": "N5",
    "犬": "N5", "猫": "N5", "車": "N5", "自転車": "N5", "電車": "N5", "バス": "N5",
    "飛行機": "N5", "駅": "N5", "店": "N5", "家": "N5", "部屋": "N5", "窓": "N5",
    "ドア": "N5", "机": "N5", "椅子": "N5", "ペン": "N5", "鉛筆": "N5", "時計": "N5",
    "傘": "N5", "靴": "N5", "服": "N5", "帽子": "N5", "コーヒー": "N5", "お茶": "N5",
    "水": "N5", "牛乳": "N5", "ご飯": "N5", "パン": "N5", "肉": "N5", "魚": "N5",
    "野菜": "N5", "果物": "N5", "りんご": "N5", "バナナ": "N5",

    # ---- N4 (~1500 words) ----
    "相手": "N4", "仲間": "N4", "両親": "N4", "親戚": "N4", "結婚": "N4", "葬式": "N4",
    "誕生日": "N4", "記念日": "N4", "約束": "N4", "予定": "N4", "計画": "N4", "準備": "N4",
    "集合": "N4", "出発": "N4", "到着": "N4", "旅行": "N4", "泊まる": "N4", "泊": "N4",
    "訪ねる": "N4", "訪問": "N4", "案内": "N4", "説明": "N4", "意見": "N4", "考える": "N4",
    "思う": "N4", "知る": "N4", "分かる": "N4", "覚える": "N4", "忘れる": "N4", "決める": "N4",
    "待つ": "N4", "助ける": "N4", "教える": "N4", "習う": "N4", "調べる": "N4", "並ぶ": "N4",
    "入る": "N4", "出る": "N4", "開ける": "N4", "閉める": "N4", "閉まる": "N4", "開く": "N4",
    "始める": "N4", "終わる": "N4", "続く": "N4", "続ける": "N4", "増える": "N4", "減る": "N4",
    "上がる": "N4", "下がる": "N4", "変わる": "N4", "変える": "N4", "直す": "N4", "治る": "N4",
    "疲れる": "N4", "病む": "N4", "怪我": "N4", "具合": "N4", "調子": "N4", "薬": "N4",
    "医者": "N4", "看護師": "N4", "ホテル": "N4", "レストラン": "N4", "スーパー": "N4",
    "デパート": "N4", "郵便局": "N4", "銀行": "N4", "図書館": "N4", "病院": "N4",
    "市役所": "N4", "公園": "N4", "海": "N4", "山": "N4", "川": "N4", "空": "N4",
    "雨": "N4", "雪": "N4", "風": "N4", "曇り": "N4", "晴れ": "N4", "天気": "N4",
    "季節": "N4", "春": "N4", "夏": "N4", "秋": "N4", "冬": "N4",
    "上手": "N4", "下手": "N4", "得意": "N4", "苦手": "N4", "真剣": "N4", "真面目": "N4",
    "重要": "N4", "必要": "N4", "十分": "N4", "不足": "N4", "不足する": "N4",
    "安心": "N4", "心配": "N4", "期待": "N4", "経験": "N4", "チャンス": "N4",
    "立派": "N4", "見事": "N4", "汚い": "N4", "危ない": "N4", "柔らかい": "N4",
    "硬い": "N4", "強い": "N4", "弱い": "N4", "太る": "N4", "痩せる": "N4",
    "普通": "N4", "特別": "N4", "当然": "N4", "確か": "N4", "確かに": "N4",
    "特に": "N4", "やっと": "N4", "ずっと": "N4", "きっと": "N4", "かなり": "N4",

    # ---- N3 (~3500 words) ----
    "担当": "N3", "担当者": "N3", "責任者": "N3", "管理": "N3", "経理": "N3", "総務": "N3",
    "人事部": "N3", "開発": "N3", "事業": "N3", "事業部": "N3", "部署": "N3", "役職": "N3",
    "社長": "N3", "部長": "N3", "課長": "N3", "主任": "N3", "契約": "N3", "契約書": "N3",
    "取引": "N3", "取引先": "N3", "顧客": "N3", "得意先": "N3", "見積もり": "N3",
    "見積書": "N3", "請求": "N3", "請求書": "N3", "納品": "N3", "発注": "N3", "注文": "N3",
    "受注": "N3", "売上": "N3", "売上高": "N3", "利益": "N3", "損失": "N3", "経費": "N3",
    "予算": "N3", "決算": "N3", "会計": "N3", "監査": "N3", "法務": "N3", "審査": "N3",
    "提出": "N3", "承認": "N3", "許可": "N3", "申請": "N3", "申込み": "N3", "届出": "N3",
    "報告": "N3", "報告書": "N3", "連絡": "N3", "確認": "N3", "調整": "N3", "交渉": "N3",
    "対応": "N3", "対策": "N3", "改善": "N3", "改革": "N3", "変更": "N3", "更新": "N3",
    "修正": "N3", "追加": "N3", "削除": "N3", "登録": "N3", "解除": "N3", "禁止": "N3",
    "義務": "N3", "権利": "N3", "責任": "N3", "規則": "N3", "規定": "N3", "規約": "N3",
    "条件": "N3", "項目": "N3", "手続き": "N3", "流れ": "N3", "仕組み": "N3",
    "結果": "N3", "原因": "N3", "理由": "N3", "手段": "N3", "方法": "N3", "内容": "N3",
    "詳細": "N3", "概略": "N3", "要約": "N3", "一覧": "N3", "一連": "N3", "一貫": "N3",
    "妥協": "N3", "競争": "N3", "勝つ": "N3", "負ける": "N3", "成功": "N3", "失敗": "N3",
    "成果": "N3", "業績": "N3", "目標": "N3", "目的": "N3", "達成": "N3", "実現": "N3",
    "効果": "N3", "効率": "N3", "生産性": "N3", "品質": "N3", "基準": "N3", "水準": "N3",
    "傾向": "N3", "傾き": "N3", "傾斜": "N3", "動向": "N3", "状況": "N3", "状態": "N3",
    "環境": "N3", "社会": "N3", "経済": "N3", "政治": "N3", "文化": "N3", "教育": "N3",
    "科学": "N3", "技術": "N3", "工学": "N3", "医療": "N3", "福祉": "N3", "行政": "N3",
    "企業": "N3", "産業": "N3", "組織": "N3", "団体": "N3", "機関": "N3", "制度": "N3",
    "複雑": "N3", "簡単": "N3", "単純": "N3", "明確": "N3", "曖昧": "N3", "詳しい": "N3",
    "全体": "N3", "部分": "N3", "共通": "N3", "個別": "N3", "相互": "N3", "各自": "N3",
    "著しい": "N3", "著者": "N3", "専門": "N3", "専門家": "N3", "有能": "N3",
    "情勢": "N3", "内情": "N3", "背景": "N3", "構造": "N3", "仕組み": "N3",

    # ---- N2 (~6000 words) ----
    "企業戦略": "N2", "経営": "N2", "経営陣": "N2", "取締役": "N2", "監査役": "N2",
    "株主": "N2", "株式": "N2", "上場": "N2", "決算": "N2", "中間決算": "N2",
    "連結": "N2", "有価証券": "N2", "証券": "N2", "金融": "N2", "融資": "N2",
    "貸付": "N2", "担保": "N2", "保証": "N2", "債権": "N2", "債務": "N2",
    "資産": "N2", "負債": "N2", "純資産": "N2", "収益": "N2", "費用": "N2",
    "減価償却": "N2", "減損": "N2", "繰越": "N2", "引当金": "N2", "賞与": "N2",
    "退職金": "N2", "福利厚生": "N2", "雇用": "N2", "採用": "N2", "中途採用": "N2",
    "新卒採用": "N2", "異動": "N2", "昇進": "N2", "降格": "N2", "退職": "N2",
    "解雇": "N2", "懲戒": "N2", "就業規則": "N2", "労働組合": "N2", "争議": "N2",
    "仲裁": "N2", "審理": "N2", "判決": "N2", "和解": "N2", "控訴": "N2",
    "強制執行": "N2", "差押": "N2", "抵当": "N2", "根抵当": "N2", "特許": "N2",
    "実用新案": "N2", "商標": "N2", "意匠": "N2", "著作権": "N2", "知的財産": "N2",
    "侵害": "N2", "ライセンス": "N2", "ライセンス契約": "N2", "独占": "N2",
    "流通": "N2", "物流": "N2", "在庫": "N2", "棚卸": "N2", "仕入": "N2",
    "委託": "N2", "下請": "N2", "発注書": "N2", "納品書": "N2", "検収": "N2",
    "品質管理": "N2", "品質保証": "N2", "統計": "N2", "統計学": "N2", "分析": "N2",
    "多変量解析": "N2", "因子分析": "N2", "重回帰": "N2", "仮説": "N2", "仮説検定": "N2",
    "有意水準": "N2", "相関": "N2", "回帰": "N2", "分散": "N2", "標準偏差": "N2",
    "確率": "N2", "確率変数": "N2", "推測": "N2", "予測": "N2", "推定": "N2",
    "画期的": "N2", "独創的": "N2", "体系的": "N2", "網羅的": "N2", "具体的": "N2",
    "抽象的": "N2", "論理的": "N2", "合理的": "N2", "総合的": "N2", "動的": "N2",
    "静的": "N2", "逐次": "N2", "一括": "N2", "随時": "N2", "定例": "N2",
    "突発": "N2", "恒常": "N2", "一時的": "N2", "恒久的": "N2", "長期的": "N2",
    "短期的": "N2", "中期的": "N2", "画一的": "N2", "弾力的": "N2", "機動的": "N2",

    # ---- N1 (~10000 advanced) ----
    "経営戦略": "N1", "中長期計画": "N1", "中期経営計画": "N1", "企業統治": "N1", "コーポレートガバナンス": "N1",
    "内部統制": "N1", "リスクマネジメント": "N1", "コンプライアンス": "N1", "法令遵守": "N1",
    "不祥事": "N1", "スキャンダル": "N1", "インサイダー": "N1", "インサイダー取引": "N1",
    "市場": "N1", "証券市場": "N1", "株式市場": "N1", "為替": "N1", "外国為替": "N1",
    "変動相場制": "N1", "固定相場制": "N1", "金融政策": "N1", "金融緩和": "N1",
    "金融引き締め": "N1", "景気": "N1", "景気後退": "N1", "景気浮揚": "N1",
    "デフレ": "N1", "インフレ": "N1", "スタグフレーション": "N1", "財政": "N1",
    "財政政策": "N1", "予算編成": "N1", "債務超過": "N1", "破綻": "N1", "連鎖破綻": "N1",
    "反トラスト": "N1", "独占禁止法": "N1", "下請法": "N1", "割賦": "N1",
    "先物取引": "N1", "オプション": "N1", "デリバティブ": "N1", "原資産": "N1",
    "証券化": "N1", "仕組み商品": "N1", "ノンバンク": "N1", "信託": "N1",
    "信託銀行": "N1", "投資顧問": "N1", "運用": "N1", "ポートフォリオ": "N1",
    "アセットアロケーション": "N1", "ベンチマーク": "N1", "ヘッジ": "N1",
    "機械学習": "N1", "深層学習": "N1", "ディープラーニング": "N1", "ニューラルネットワーク": "N1",
    "畳み込み": "N1", "リカレント": "N1", "トランスフォーマー": "N1", "言語モデル": "N1",
    "大規模言語モデル": "N1", "強化学習": "N1", "転移学習": "N1", "教師あり学習": "N1",
    "教師なし学習": "N1", "クラスタリング": "N1", "次元削減": "N1", "自然言語処理": "N1",
    "形態素解析": "N1", "固有表現": "N1", "固有表現抽出": "N1", "係り受け": "N1",
    "構文解析": "N1", "意味解析": "N1", "機械翻訳": "N1", "ニューラル機械翻訳": "N1",
    "推論": "N1", "プロンプト": "N1", "プロンプトエンジニアリング": "N1",
    "埋め込み": "N1", "埋め込みベクトル": "N1", "ベクトル": "N1", "テンソル": "N1",
    "コンテナ": "N1", "オーケストレーション": "N1", "マイクロサービス": "N1",
    "Kubernetes": "N1", "仮想化": "N1", "SRE": "N1", "データベース": "N1",
    "分散データベース": "N1", "トランザクション": "N1", "同時実行制御": "N1",
    "悲観的ロック": "N1", "楽観的ロック": "N1", "デッドロック": "N1",
    "レプリケーション": "N1", "シャーディング": "N1", "パーティショニング": "N1",
    "API": "N1", "REST": "N1", "GraphQL": "N1", "gRPC": "N1", "OAuth": "N1",
    "OpenID": "N1", "認可": "N1", "認証": "N1", "多要素認証": "N1", "生体認証": "N1",
    "暗号化": "N1", "公開鍵暗号": "N1", "共通鍵暗号": "N1", "ハッシュ": "N1",
    "デジタル署名": "N1", "電子署名": "N1", "タイムスタンプ": "N1", "監査証跡": "N1",
    "準拠": "N1", "ゼロトラスト": "N1", "ゼロデイ": "N1", "脆弱性": "N1",
    "社会的責任": "N1", "ESG": "N1", "サステナビリティ": "N1", "カーボンニュートラル": "N1",
    "脱炭素": "N1", "気候変動": "N1", "SDGs": "N1",
}

# Readings-to-JLPT for pure-kana tokens
_JLPT_READING: Dict[str, str] = {
    "わたし": "N5", "あなた": "N5", "かのじょ": "N5", "かれ": "N5", "ひと": "N5",
    "ともだち": "N5", "せんせい": "N5", "がくせい": "N5", "がっこう": "N5",
    "だいがく": "N5", "かいしゃ": "N5", "しごと": "N5", "ほん": "N5", "てがみ": "N5",
    "しんぶん": "N5", "きょう": "N5", "あした": "N5", "きのう": "N5", "まいにち": "N5",
    "まいしゅう": "N5", "まいつき": "N5", "まいとし": "N5", "いま": "N5", "まえ": "N5",
    "あと": "N5", "とき": "N5", "じかん": "N5", "ふん": "N5", "あさ": "N5", "ひる": "N5",
    "ばん": "N5", "よる": "N5", "いく": "N5", "くる": "N5", "かえる": "N5",
    "はしる": "N5", "あるく": "N5", "みる": "N5", "きく": "N5", "よむ": "N5", "かく": "N5",
    "はなす": "N5", "たべる": "N5", "のむ": "N5", "かう": "N5", "つくる": "N5",
    "する": "N5", "です": "N5", "ます": "N5", "だ": "N5", "ある": "N5", "いる": "N5",
    "ない": "N5", "いい": "N5", "よい": "N5", "わるい": "N5", "おおきい": "N5",
    "ちいさい": "N5", "たかい": "N5", "やすい": "N5", "ながい": "N5", "みじかい": "N5",
    "あたらしい": "N5", "ふるい": "N5", "あつい": "N5", "さむい": "N5",
    "つめたい": "N5", "おもしろい": "N5", "おいしい": "N5", "まずい": "N5",
    "はやい": "N5", "おそい": "N5", "おおい": "N5", "すくない": "N5", "とおい": "N5",
    "ちかい": "N5", "あかるい": "N5", "くらい": "N5", "ひろい": "N5", "せまい": "N5",
    "たいせつ": "N5", "べんり": "N5", "ふべん": "N5", "げんき": "N5", "びょうき": "N5",
    "ひま": "N5", "いそがしい": "N5", "すき": "N5", "きらい": "N5", "きれい": "N5",
    "ていねい": "N5", "しずか": "N5", "にぎやか": "N5", "やさしい": "N5",
    "にほん": "N5", "ちゅうごく": "N5", "えいご": "N5", "にほんご": "N5",
    "べんきょう": "N5", "れんしゅう": "N5", "しつもん": "N5", "こたえる": "N5",
    "もんだい": "N5", "こたえ": "N5", "かぞく": "N5", "ちち": "N5", "はは": "N5",
    "あに": "N5", "あね": "N5", "おとうと": "N5", "いもうと": "N5", "こども": "N5",
    "おとこ": "N5", "おんな": "N5", "いぬ": "N5", "ねこ": "N5", "くるま": "N5",
    "じてんしゃ": "N5", "でんしゃ": "N5", "えき": "N5", "みせ": "N5", "いえ": "N5",
    "うち": "N5", "へや": "N5", "まど": "N5", "つくえ": "N5", "いす": "N5",
    "えんぴつ": "N5", "とけい": "N5", "かさ": "N5", "くつ": "N5", "ふく": "N5",
    "ぼうし": "N5", "こーひー": "N5", "おちゃ": "N5", "みず": "N5", "ぎゅうにゅう": "N5",
    "ごはん": "N5", "ぱん": "N5", "にく": "N5", "さかな": "N5", "やさい": "N5",
    "くだもの": "N5", "りんご": "N5", "ばなな": "N5",
    "あいて": "N4", "りょうしん": "N4", "けっこん": "N4", "やくそく": "N4",
    "よてい": "N4", "けいかく": "N4", "じゅんび": "N4", "しゅっぱつ": "N4",
    "とうちゃく": "N4", "りょこう": "N4", "とまる": "N4", "たずねる": "N4",
    "ほうもん": "N4", "あんない": "N4", "せつめい": "N4", "いけん": "N4",
    "かんがえる": "N4", "おもう": "N4", "しる": "N4", "わかる": "N4",
    "おぼえる": "N4", "わすれる": "N4", "きめる": "N4", "まつ": "N4",
    "たすける": "N4", "おしえる": "N4", "ならう": "N4", "しらべる": "N4",
    "ならぶ": "N4", "はいる": "N4", "でる": "N4", "あける": "N4", "しめる": "N4",
    "しまる": "N4", "ひらく": "N4", "はじめる": "N4", "おわる": "N4",
    "つづく": "N4", "つづける": "N4", "ふえる": "N4", "へる": "N4",
    "あがる": "N4", "さがる": "N4", "かわる": "N4", "かえる": "N4",
    "なおす": "N4", "なおる": "N4", "つかれる": "N4", "けが": "N4",
    "ぐあい": "N4", "ちょうし": "N4", "くすり": "N4", "いしゃ": "N4",
    "かんごし": "N4", "ゆうびんきょく": "N4", "ぎんこう": "N4", "としょかん": "N4",
    "びょういん": "N4", "しやくしょ": "N4", "こうえん": "N4", "うみ": "N4",
    "やま": "N4", "かわ": "N4", "そら": "N4", "あめ": "N4", "ゆき": "N4",
    "かぜ": "N4", "くもり": "N4", "はれ": "N4", "てんき": "N4", "きせつ": "N4",
    "はる": "N4", "なつ": "N4", "あき": "N4", "ふゆ": "N4",
    "じょうず": "N4", "へた": "N4", "とくい": "N4", "にがて": "N4",
    "しんけん": "N4", "まじめ": "N4", "じゅうよう": "N4", "ひつよう": "N4",
    "じゅうぶん": "N4", "ふそく": "N4", "あんしん": "N4", "しんぱい": "N4",
    "きたい": "N4", "けいけん": "N4", "りっぱ": "N4", "みごと": "N4",
    "きたない": "N4", "あぶない": "N4", "やわらかい": "N4", "かたい": "N4",
    "つよい": "N4", "よわい": "N4", "ふとる": "N4", "やせる": "N4",
    "ふつう": "N4", "とくべつ": "N4", "とうぜん": "N4", "たしか": "N4",
    "たんとう": "N3", "たんとうしゃ": "N3", "せきにんしゃ": "N3", "かんり": "N3",
    "けいり": "N3", "そうむ": "N3", "じんじぶ": "N3", "かいはつ": "N3",
    "じぎょう": "N3", "ぶしょ": "N3", "やくしょく": "N3", "しゃちょう": "N3",
    "ぶちょう": "N3", "かちょう": "N3", "しゅにん": "N3", "けいやく": "N3",
    "けいやくしょ": "N3", "とりひき": "N3", "とりひきさき": "N3",
    "こきゃく": "N3", "とくいさき": "N3", "みつもり": "N3", "みつもりしょ": "N3",
    "せいきゅうしょ": "N3", "のうひん": "N3", "はっちゅう": "N3", "ちゅうもん": "N3",
    "じゅちゅう": "N3", "うりあげ": "N3", "うりあげだか": "N3",
    "りえき": "N3", "そんしつ": "N3", "けいひ": "N3", "よさん": "N3",
    "けっさん": "N3", "かいけい": "N3", "かんさ": "N3", "ほうむ": "N3",
    "しんさ": "N3", "ていしゅつ": "N3", "しょうにん": "N3", "きょか": "N3",
    "しんせい": "N3", "もうしこみ": "N3", "とどけで": "N3", "ほうこく": "N3",
    "ほうこくしょ": "N3", "れんらく": "N3", "かくにん": "N3", "ちょうせい": "N3",
    "こうしょう": "N3", "たいおう": "N3", "たいさく": "N3", "かいぜん": "N3",
    "かいかく": "N3", "へんこう": "N3", "こうしん": "N3", "しゅうせい": "N3",
    "ついか": "N3", "さくじょ": "N3", "とうろく": "N3", "かいじょ": "N3",
    "きんし": "N3", "ぎむ": "N3", "けんり": "N3", "せきにん": "N3",
    "きそく": "N3", "きてい": "N3", "きやく": "N3", "じょうけん": "N3",
    "こうもく": "N3", "てつづき": "N3", "ながれ": "N3", "しくみ": "N3",
    "けっか": "N3", "げんいん": "N3", "りゆう": "N3", "しゅだん": "N3",
    "ほうほう": "N3", "ないよう": "N3", "しょうさい": "N3", "がいりゃく": "N3",
    "ようやく": "N3", "いちらん": "N3", "いちれん": "N3", "いっかん": "N3",
    "だきょう": "N3", "きょうそう": "N3", "かつ": "N3", "まける": "N3",
    "せいこう": "N3", "しっぱい": "N3", "せいか": "N3", "ぎょうせき": "N3",
    "もくひょう": "N3", "もくてき": "N3", "たっせい": "N3", "じつげん": "N3",
    "こうか": "N3", "こうりつ": "N3", "せいさんせい": "N3", "ひんしつ": "N3",
    "きじゅん": "N3", "すいじゅん": "N3", "けいこう": "N3", "どうこう": "N3",
    "じょうきょう": "N3", "じょうたい": "N3", "かんきょう": "N3", "しゃかい": "N3",
    "けいざい": "N3", "せいじ": "N3", "ぶんか": "N3", "きょういく": "N3",
    "かがく": "N3", "ぎじゅつ": "N3", "こうがく": "N3", "いりょう": "N3",
    "ふくし": "N3", "ぎょうせい": "N3", "きぎょう": "N3", "さんぎょう": "N3",
    "そしき": "N3", "だんたい": "N3", "きかん": "N3", "せいど": "N3",
    "ふくざつ": "N3", "かんたん": "N3", "たんじゅん": "N3", "めいかく": "N3",
    "あいまい": "N3", "くわしい": "N3", "ぜんたい": "N3", "ぶぶん": "N3",
    "きょうつう": "N3", "こべつ": "N3", "そうご": "N3", "かくじ": "N3",
    "いじる": "N3", "せんもん": "N3", "せんもんか": "N3", "ゆうのう": "N3",
    "じょうせい": "N3", "ないじょう": "N3", "はいけい": "N3", "こうぞう": "N3",
    "けいえい": "N2", "けいえいじん": "N2", "とりしまりやく": "N2",
    "かんさやく": "N2", "かびぬし": "N2", "かぶしき": "N2", "じょうじょう": "N2",
    "ちゅうかんけっさん": "N2", "れんけつ": "N2", "ゆうかしょうけん": "N2",
    "しょうけん": "N2", "きんゆう": "N2", "ゆうし": "N2", "かしつけ": "N2",
    "たんぽ": "N2", "ほしょう": "N2", "さいけん": "N2", "さいむ": "N2",
    "しさん": "N2", "ふさい": "N2", "じゅんしさん": "N2", "しゅうえき": "N2",
    "ひよう": "N2", "げんかしょうきゃく": "N2", "げんそん": "N2",
    "くりこし": "N2", "ひきあてきん": "N2", "しょうよ": "N2", "たいしょくきん": "N2",
    "ふくりこうせい": "N2", "こよう": "N2", "さいよう": "N2",
    "ちゅうとさいよう": "N2", "しんそつさいよう": "N2", "いどう": "N2",
    "しょうしん": "N2", "こうかく": "N2", "たいしょく": "N2", "かいこ": "N2",
    "ちょうかい": "N2", "しゅうぎょうきそく": "N2", "ろうどうくみあい": "N2",
    "そうぎ": "N2", "ちゅうさい": "N2", "しんり": "N2", "はんけつ": "N2",
    "わかい": "N2", "こうそ": "N2", "きょうせいしっこう": "N2",
    "さしおさえ": "N2", "ていとう": "N2", "ねていとう": "N2", "とっきょ": "N2",
    "じつようしんあん": "N2", "しょうひょう": "N2", "いしょう": "N2",
    "ちょさくけん": "N2", "ちてきざいさん": "N2", "しんがい": "N2",
    "りゅうつう": "N2", "ぶつりゅう": "N2", "ざいこ": "N2", "たなおろし": "N2",
    "しいれ": "N2", "いたく": "N2", "したうけ": "N2", "はっちゅうしょ": "N2",
    "のうひんしょ": "N2", "けんしゅう": "N2", "ひんしつかんり": "N2",
    "ひんしつほしょう": "N2", "とうけい": "N2", "とうけいがく": "N2",
    "ぶんせき": "N2", "たへんりょうかいせき": "N2", "いんしかいせき": "N2",
    "じゅうかいき": "N2", "かせつ": "N2", "かせつけんてい": "N2",
    "ゆういすいじゅん": "N2", "そうかん": "N2", "かいき": "N2", "ぶんさん": "N2",
    "ひょうじゅんへんさ": "N2", "かくりつ": "N2", "かくりつへんすう": "N2",
    "すいそく": "N2", "よそく": "N2", "すいてい": "N2",
    "かっきてき": "N2", "どくそうてき": "N2", "たいけいてき": "N2",
    "もうらてき": "N2", "ぐたいてき": "N2", "ちゅうしょうてき": "N2",
    "ろんりてき": "N2", "ごうりてき": "N2", "そうごうてき": "N2",
    "どうてき": "N2", "せいてき": "N2", "ちくじ": "N2", "いっかつ": "N2",
    "ずいじ": "N2", "ていれい": "N2", "とっぱつ": "N2", "こうじょう": "N2",
    "いちじてき": "N2", "こうきゅうてき": "N2", "ちょうきてき": "N2",
    "たんきてき": "N2", "ちゅうきてき": "N2", "がくいってき": "N2",
    "だんりょくてき": "N2", "きどうてき": "N2",
}

# JLPT levels sorted easiest-first for "pick the easier of two candidates" fallback
_JLPT_RANK = {"N5": 0, "N4": 1, "N3": 2, "N2": 3, "N1": 4}


def _jlpt_kanji_fallback(word: str) -> Optional[str]:
    """Assign a rough JLPT level purely on kanji counts when dictionary lookup fails.
    0 kanji → N5 (simple particles / pure-kana).  1-3 common kanji → N4-N3.
    ≥4 kanji or kigo/rare → N2/N1."""
    if not word:
        return None
    count = sum(1 for c in word if "\u4e00" <= c <= "\u9fff")
    if count == 0:
        return None
    if count <= 1:
        return "N4"
    if count <= 2:
        return "N3"
    if count <= 3:
        return "N2"
    return "N1"


def jlpt_level_for(word: str, reading: Optional[str] = None, dictionary_form: Optional[str] = None) -> Optional[str]:
    """Return JLPT level (N5/N4/N3/N2/N1) or None if unknown."""
    if not word:
        return None
    best_rank = 99
    best = None
    candidates = [word]
    if dictionary_form and dictionary_form != word:
        candidates.append(dictionary_form)
    if reading and reading != word:
        candidates.append(reading)
    for c in candidates:
        lvl = _JLPT_DICT_FORM.get(c) or _JLPT_READING.get(c)
        if lvl and lvl in _JLPT_RANK:
            r = _JLPT_RANK[lvl]
            if r < best_rank:
                best_rank = r
                best = lvl
    if best is None:
        best = _jlpt_kanji_fallback(word)
    return best


def build_alternatives(surface: str, reading: Optional[str] = None,
                       normalized_form: Optional[str] = None,
                       dictionary_form: Optional[str] = None,
                       jlpt: Optional[str] = None) -> Dict[str, Any]:
    """All alternative representations known for the current token:
       reading (kana), romaji, dictionary vs normalized form, keigo/synonyms,
       plus jlpt label when available. Returns an order-stable dict."""
    alts: Dict[str, Any] = {}
    if surface:
        alts["surface"] = surface
    if reading and reading != surface:
        alts["reading"] = reading
        rom = _kana_to_romaji(reading)
        if rom and rom != reading and rom.lower() != reading.lower():
            alts["romaji"] = rom
    if dictionary_form and dictionary_form and dictionary_form not in (surface, ""):
        alts["dictionary_form"] = dictionary_form
        synonym = _JA_SYNONYMS.get(dictionary_form) or _JA_SYNONYMS.get(surface or "")
        if synonym and synonym not in (surface, dictionary_form, ""):
            alts["synonym"] = synonym
        keigo = _KEIGO_MAP_14.get(dictionary_form) or _KEIGO_MAP_14.get(surface or "")
        if keigo and keigo not in (surface, dictionary_form, ""):
            alts["keigo_honorific"] = keigo
    if normalized_form and normalized_form not in (surface, dictionary_form, ""):
        alts["normalized_form"] = normalized_form
    if jlpt:
        alts["jlpt"] = jlpt
    return alts


def _get_dictionary() -> Dictionary:
    """Lazily load and cache the SudachiPy dictionary."""
    global _SUDACHI_DICT
    if _SUDACHI_DICT is None:
        _SUDACHI_DICT = dictionary.Dictionary()
    return _SUDACHI_DICT


def get_tokenizer(mode: str = "C") -> Tokenizer:
    """
    Create a SudachiPy tokenizer.

    Parameters
    ----------
    mode:
        Tokenization mode - one of "A", "B", "C".
        - C (default): longest unit (most coarse)
        - B: middle unit
        - A: shortest unit (most fine-grained)
    """
    mode_map = {
        "A": Tokenizer.SplitMode.A,
        "B": Tokenizer.SplitMode.B,
        "C": Tokenizer.SplitMode.C,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of A, B, C.")
    return _get_dictionary().create(mode_map[mode])


@dataclass
class TokenInfo:
    """Structured morphological info for a single token."""
    surface: str
    pos: List[str]
    pos_main: str
    reading: str
    normalized_form: str
    dictionary_form: str
    start: int
    end: int
    jlpt_level: Optional[str] = None
    alternatives: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _extract_tokens(text: str, mode: str) -> List[TokenInfo]:
    tokenizer = get_tokenizer(mode)
    mlist = tokenizer.tokenize(text)
    results: List[TokenInfo] = []
    for morph in mlist:
        pos = list(morph.part_of_speech())
        surface = morph.surface()
        reading = morph.reading_form() or ""
        dictionary_form = morph.dictionary_form() or surface
        normalized_form = morph.normalized_form() or surface
        jlpt = jlpt_level_for(surface, reading=reading, dictionary_form=dictionary_form)
        results.append(
            TokenInfo(
                surface=surface,
                pos=pos,
                pos_main=pos[0] if pos else "",
                reading=reading,
                normalized_form=normalized_form,
                dictionary_form=dictionary_form,
                start=morph.begin(),
                end=morph.end(),
                jlpt_level=jlpt,
                alternatives=build_alternatives(
                    surface=surface,
                    reading=reading,
                    normalized_form=normalized_form,
                    dictionary_form=dictionary_form,
                    jlpt=jlpt,
                ),
            )
        )
    return results


@dataclass
class TokenizeResult:
    text: str
    mode: str
    token_count: int
    tokens: List[TokenInfo] = field(default_factory=list)
    token_surfaces: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "mode": self.mode,
            "token_count": self.token_count,
            "token_surfaces": self.token_surfaces,
            "tokens": [t.to_dict() for t in self.tokens],
        }


def tokenize(text: str, mode: str = "C") -> TokenizeResult:
    """
    Split Japanese text into tokens with morphological features.

    Parameters
    ----------
    text :
        Japanese input text.
    mode :
        Sudachi split mode ("A", "B", or "C").
    """
    if not isinstance(text, str) or not text.strip():
        return TokenizeResult(text=text or "", mode=mode, token_count=0)

    tokens = _extract_tokens(text, mode)
    return TokenizeResult(
        text=text,
        mode=mode,
        token_count=len(tokens),
        tokens=tokens,
        token_surfaces=[t.surface for t in tokens],
    )


@dataclass
class AnalyzeResult:
    text: str
    mode: str
    token_count: int
    token_surfaces: List[str] = field(default_factory=list)
    tokens: List[TokenInfo] = field(default_factory=list)
    pos_counts: Dict[str, int] = field(default_factory=dict)
    readings: List[str] = field(default_factory=list)
    wakati: str = ""
    kana_only_text: str = ""
    noun_list: List[str] = field(default_factory=list)
    verb_list: List[str] = field(default_factory=list)
    adjective_list: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "mode": self.mode,
            "token_count": self.token_count,
            "token_surfaces": self.token_surfaces,
            "wakati": self.wakati,
            "readings": self.readings,
            "kana_only_text": self.kana_only_text,
            "pos_counts": self.pos_counts,
            "noun_list": self.noun_list,
            "verb_list": self.verb_list,
            "adjective_list": self.adjective_list,
            "tokens": [t.to_dict() for t in self.tokens],
        }


def analyze(text: str, mode: str = "C") -> AnalyzeResult:
    """
    Full morphological analysis with POS counts, readings, wakati, and POS filtering.
    """
    tok = tokenize(text, mode)

    pos_counts: Dict[str, int] = {}
    readings: List[str] = []
    nouns: List[str] = []
    verbs: List[str] = []
    adjectives: List[str] = []

    for t in tok.tokens:
        pos_counts[t.pos_main] = pos_counts.get(t.pos_main, 0) + 1
        readings.append(t.reading if t.reading else t.surface)

        if t.pos_main == "名詞":
            nouns.append(t.surface)
        elif t.pos_main in ("動詞", "接尾辞") and "動詞" in t.pos:
            verbs.append(t.dictionary_form)
        elif t.pos_main == "動詞":
            verbs.append(t.dictionary_form)
        elif t.pos_main == "形容詞":
            adjectives.append(t.dictionary_form)

    kana_text = "".join(readings)
    wakati = " ".join(tok.token_surfaces)

    return AnalyzeResult(
        text=tok.text,
        mode=tok.mode,
        token_count=tok.token_count,
        token_surfaces=tok.token_surfaces,
        tokens=tok.tokens,
        pos_counts=pos_counts,
        readings=readings,
        wakati=wakati,
        kana_only_text=kana_text,
        noun_list=nouns,
        verb_list=verbs,
        adjective_list=adjectives,
    )


@dataclass
class KeywordItem:
    keyword: str
    score: float
    frequency: int
    pos_hint: str
    jlpt_level: Optional[str] = None
    alternatives: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "keyword": self.keyword,
            "score": round(self.score, 4),
            "frequency": self.frequency,
            "pos_hint": self.pos_hint,
            "jlpt_level": self.jlpt_level,
            "alternatives": self.alternatives,
        }


@dataclass
class KeywordsResult:
    text: str
    mode: str
    top_n: int
    keywords: List[KeywordItem] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "mode": self.mode,
            "top_n": self.top_n,
            "keyword_count": len(self.keywords),
            "keywords": [k.to_dict() for k in self.keywords],
        }


def extract_keywords(
    text: str,
    mode: str = "C",
    top_n: int = 10,
    include_pos: Optional[Tuple[str, ...]] = None,
) -> KeywordsResult:
    """
    Extract keywords from Japanese text using POS filtering + stopword removal
    + TF-IDF-style frequency × log-length scoring.

    Parameters
    ----------
    text:
        Japanese input text.
    mode:
        Sudachi split mode.
    top_n:
        Maximum number of keywords to return.
    include_pos:
        Tuple of Japanese POS names (e.g. ("名詞", "動詞", "形容詞")).
        Defaults to nouns + sahen nouns + proper nouns.
    """
    if not isinstance(text, str) or not text.strip():
        return KeywordsResult(text=text or "", mode=mode, top_n=top_n)

    if include_pos is None:
        include_pos = ("名詞",)

    tokens = _extract_tokens(text, mode)
    freq: Dict[str, int] = {}
    pos_of: Dict[str, str] = {}
    meta_of: Dict[str, Dict[str, str]] = {}
    text_len = len(text)

    for t in tokens:
        if t.pos_main not in include_pos:
            continue
        surface = t.surface
        norm = t.normalized_form or surface
        if len(norm) < 2:
            continue
        if norm in _JA_STOPWORDS:
            continue
        if surface.isdigit() or norm.isdigit():
            continue
        if all(ord(c) < 128 for c in norm) and len(norm) < 3:
            continue
        key = norm
        freq[key] = freq.get(key, 0) + 1
        if key not in pos_of:
            pos_of[key] = t.pos_main
        if key not in meta_of:
            meta_of[key] = {
                "reading": t.reading or "",
                "dictionary_form": t.dictionary_form or key,
                "normalized_form": key,
                "surface_seen": t.surface,
                "jlpt": t.jlpt_level,
            }
        else:
            m = meta_of[key]
            if not m.get("reading") and t.reading:
                m["reading"] = t.reading
            if not m.get("jlpt") and t.jlpt_level:
                m["jlpt"] = t.jlpt_level

    doc_log_weight = math.log1p(text_len / 10.0) if text_len else 1.0
    scored: List[KeywordItem] = []
    for key, f in freq.items():
        length_boost = math.log1p(len(key))
        idf_approx = math.log1p(doc_log_weight / (1.0 + f / 2.0))
        score = f * idf_approx * length_boost
        m = meta_of.get(key, {})
        jlpt = m.get("jlpt") or jlpt_level_for(
            key, reading=m.get("reading"), dictionary_form=m.get("dictionary_form")
        )
        scored.append(KeywordItem(
            keyword=key,
            score=score,
            frequency=f,
            pos_hint=pos_of.get(key, ""),
            jlpt_level=jlpt,
            alternatives=build_alternatives(
                surface=m.get("surface_seen") or key,
                reading=m.get("reading"),
                normalized_form=key,
                dictionary_form=m.get("dictionary_form"),
                jlpt=jlpt,
            ),
        ))

    scored.sort(key=lambda x: (-x.score, -x.frequency, x.keyword))
    selected = scored[: max(1, top_n)]
    return KeywordsResult(text=text, mode=mode, top_n=top_n, keywords=selected)


@dataclass
class NamedEntity:
    text: str
    label: str
    start: int
    end: int
    reading: str = ""
    source: str = "sudachi_dictionary_pos"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NamedEntitiesResult:
    text: str
    mode: str
    method: str
    entities: List[NamedEntity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode,
            "method": self.method,
            "entity_count": len(self.entities),
            "entities": [entity.to_dict() for entity in self.entities],
        }


def extract_named_entities(text: str, mode: str = "C") -> NamedEntitiesResult:
    """Extract contiguous proper-noun spans using Sudachi dictionary POS tags.

    This is a dictionary/POS-based baseline, not a contextual or trained NER
    model. Labels are only specialized when Sudachi explicitly provides them.
    """
    method = "SudachiPy dictionary POS tags; not a learned contextual model"
    if not isinstance(text, str) or not text.strip():
        return NamedEntitiesResult(text=text or "", mode=mode, method=method)

    tokens = _extract_tokens(text, mode)
    entities: List[NamedEntity] = []
    pending: Optional[NamedEntity] = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            pending.text = text[pending.start:pending.end]
            entities.append(pending)
            pending = None

    for token in tokens:
        pos = token.pos
        if len(pos) < 3 or pos[1] != "固有名詞":
            flush()
            continue

        label = {
            "人名": "PERSON",
            "地名": "LOCATION",
            "組織": "ORGANIZATION",
        }.get(pos[2], "PROPER_NOUN")

        if pending and pending.label == label and pending.end == token.start:
            pending.end = token.end
            pending.reading += token.reading or token.surface
        else:
            flush()
            pending = NamedEntity(
                text=token.surface,
                label=label,
                start=token.start,
                end=token.end,
                reading=token.reading or token.surface,
            )

    flush()
    return NamedEntitiesResult(text=text, mode=mode, method=method, entities=entities)


@dataclass
class SentimentResult:
    text: str
    score: float
    label: str
    confidence: float
    positive_matches: List[str] = field(default_factory=list)
    negative_matches: List[str] = field(default_factory=list)
    positive_jlpt: Dict[str, Optional[str]] = field(default_factory=dict)
    negative_jlpt: Dict[str, Optional[str]] = field(default_factory=dict)
    matches_alternatives: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    positive_count: int = 0
    negative_count: int = 0
    token_count: int = 0
    mode: str = "C"

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "mode": self.mode,
            "score": round(self.score, 4),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "positive_matches": sorted(set(self.positive_matches)),
            "negative_matches": sorted(set(self.negative_matches)),
            "positive_jlpt": {w: self.positive_jlpt.get(w) for w in sorted(set(self.positive_matches))},
            "negative_jlpt": {w: self.negative_jlpt.get(w) for w in sorted(set(self.negative_matches))},
            "matches_alternatives": self.matches_alternatives,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "token_count": self.token_count,
        }


def analyze_sentiment(text: str, mode: str = "C") -> SentimentResult:
    """
    Rule-based Japanese sentiment analysis.

    Score range: -1.0 (strongly negative) to +1.0 (strongly positive).
    Label: "positive" | "negative" | "neutral".
    Confidence: 0.0 - 1.0 based on matched polarity strength.
    """
    if not isinstance(text, str) or not text.strip():
        return SentimentResult(
            text=text or "",
            score=0.0,
            label="neutral",
            confidence=0.0,
            mode=mode,
        )

    tokens = _extract_tokens(text, mode)
    positive_hits: List[str] = []
    negative_hits: List[str] = []
    pos_jlpt: Dict[str, Optional[str]] = {}
    neg_jlpt: Dict[str, Optional[str]] = {}
    alt_map: Dict[str, Dict[str, Any]] = {}

    for t in tokens:
        surface = t.surface
        dict_form = t.dictionary_form or surface
        candidates = [surface, dict_form, t.normalized_form or surface]
        hit_word = None
        polarity = None
        for cand in candidates:
            if cand in _POSITIVE_WORDS:
                hit_word = cand
                polarity = "pos"
                break
            if cand in _NEGATIVE_WORDS:
                hit_word = cand
                polarity = "neg"
                break
        if hit_word is None:
            continue
        if polarity == "pos":
            positive_hits.append(hit_word)
            if hit_word not in pos_jlpt:
                pos_jlpt[hit_word] = jlpt_level_for(hit_word, reading=t.reading, dictionary_form=dict_form)
        else:
            negative_hits.append(hit_word)
            if hit_word not in neg_jlpt:
                neg_jlpt[hit_word] = jlpt_level_for(hit_word, reading=t.reading, dictionary_form=dict_form)
        if hit_word not in alt_map:
            jl = (pos_jlpt if polarity == "pos" else neg_jlpt).get(hit_word)
            synonym = _JA_SYNONYMS.get(dict_form) or _JA_SYNONYMS.get(hit_word)
            alt_obj = build_alternatives(
                surface=surface,
                reading=t.reading,
                normalized_form=t.normalized_form or surface,
                dictionary_form=dict_form,
                jlpt=jl,
            )
            if synonym and synonym != hit_word:
                alt_obj["synonym_honorific_alt"] = synonym
            # keigo / honorific alternative if present
            keigo = None
            if polarity and dict_form in _KEIGO_MAP_14:
                keigo = _KEIGO_MAP_14[dict_form]
            elif hit_word in _KEIGO_MAP_14:
                keigo = _KEIGO_MAP_14[hit_word]
            if keigo and keigo != hit_word and keigo != alt_obj.get("dictionary_form"):
                alt_obj["honorific_keigo"] = keigo
            alt_map[hit_word] = alt_obj

    pos_count = len(positive_hits)
    neg_count = len(negative_hits)
    token_count = len(tokens)

    total = pos_count + neg_count
    if total == 0:
        return SentimentResult(
            text=text,
            score=0.0,
            label="neutral",
            confidence=0.0,
            positive_matches=[],
            negative_matches=[],
            positive_jlpt={},
            negative_jlpt={},
            matches_alternatives={},
            positive_count=0,
            negative_count=0,
            token_count=token_count,
            mode=mode,
        )

    raw = (pos_count - neg_count) / total
    score = max(-1.0, min(1.0, raw))

    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"

    coverage = min(1.0, total / max(1, token_count / 8.0))
    margin = abs(score)
    confidence = min(1.0, (0.35 + 0.65 * margin) * (0.5 + 0.5 * coverage))

    return SentimentResult(
        text=text,
        score=score,
        label=label,
        confidence=confidence,
        positive_matches=positive_hits,
        negative_matches=negative_hits,
        positive_jlpt=pos_jlpt,
        negative_jlpt=neg_jlpt,
        matches_alternatives=alt_map,
        positive_count=pos_count,
        negative_count=neg_count,
        token_count=token_count,
        mode=mode,
    )


# ---------------- NEW: JLPT LEVEL DETECTION & SCORE ----------------

_JLPT_LEVEL_SCORE: Dict[str, float] = {"N5": 20.0, "N4": 40.0, "N3": 60.0, "N2": 80.0, "N1": 100.0}
_LEVEL_RANK: Dict[str, int] = {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}

_JLPT_LEX: Dict[str, Dict[str, str]] = {}

def _jlpt_add(entries: List[str], level: str, kind: str = "vocab") -> None:
    for k in entries:
        if k not in _JLPT_LEX:
            _JLPT_LEX[k] = {"level": level, "kind": kind}

_jlpt_add([
    "私", "あなた", "彼", "彼女", "私達", "彼ら", "僕", "俺", "自分", "人",
    "学校", "会社", "本", "手紙", "映画", "音楽", "新聞", "先生", "学生", "時",
    "今日", "明日", "昨日", "毎日", "毎朝", "朝", "晩", "天気", "雨", "庭",
    "公園", "駅", "銀行", "病院", "薬局", "地図", "鉛筆", "時計", "靴", "車",
    "自転車", "家", "部屋", "机", "椅子", "窓", "ドア", "犬", "猫", "花",
    "水", "お茶", "コーヒー", "ジュース", "ご飯", "パン", "魚", "肉", "野菜",
    "果物", "りんご", "バナナ", "友達", "家族", "父", "母", "兄", "姉", "弟",
    "妹", "子供", "男", "女", "子", "大人", "国", "町", "村", "海", "山",
    "川", "空", "月", "火", "木", "金", "土", "日", "週", "年",
], "N5", "vocab")
_jlpt_add([
    "食べる", "飲む", "見る", "聞く", "読む", "書く", "話す", "行く", "来る", "帰る",
    "買う", "作る", "使う", "する", "なる", "ある", "いる", "待つ", "歩く", "走る",
    "乗る", "降りる", "取る", "持つ", "開ける", "閉める", "切る", "洗う", "掃除する", "料理する",
    "勉強する", "働く", "遊ぶ", "寝る", "起きる", "座る", "立つ", "入る", "出る", "会う",
], "N5", "vocab")
_jlpt_add([
    "大きい", "小さい", "新しい", "古い", "良い", "悪い", "高い", "安い", "長い", "短い",
    "面白い", "難しい", "易しい", "楽しい", "嬉しい", "悲しい", "暑い", "寒い", "危ない", "早い",
    "遅い", "速い", "重い", "軽い", "太い", "細い", "広い", "狭い", "深い", "浅い",
], "N5", "vocab")
_jlpt_add([
    "綺麗", "元気", "静か", "賑やか", "有名", "便利", "立派", "簡単", "丁寧", "親切",
    "安全", "安心", "大切", "重要", "大好き", "嫌い", "きれい",
], "N5", "vocab")
_jlpt_add([
    "は", "が", "を", "に", "で", "の", "から", "まで", "と", "や", "も", "か", "へ",
], "N5", "vocab")
_jlpt_add([
    "です", "ます", "ました", "でした", "ません", "ない", "た", "だ", "である",
], "N5", "vocab")
_jlpt_add([
    "この", "その", "あの", "どの", "これ", "それ", "あれ", "どれ", "ここ", "そこ",
    "あそこ", "どこ", "どうして", "どう", "いくら", "いくつ", "何", "誰", "いつ",
], "N5", "vocab")

_jlpt_add([
    "宿題", "試験", "質問", "答え", "旅行", "予約", "勉強", "研究", "料理", "掃除",
    "洗濯", "買い物", "散歩", "仕事", "授業", "提出", "締切", "就職", "転職", "合格",
    "失敗", "注意", "心配", "説明", "招待", "手続", "翻訳", "通訳", "約束", "訪問",
    "相談", "連絡", "報告", "準備", "整理", "経験", "記憶", "意見", "考え", "夢",
    "希望", "計画", "理由", "目的", "方法", "種類", "程度", "関係", "問題", "答え",
    "道具", "材料", "部品", "機会", "場所", "時間", "代金", "料金", "費用", "値段",
    "数量", "額", "品物", "商品", "注文",
], "N4", "vocab")
_jlpt_add([
    "送る", "受ける", "届ける", "教える", "習う", "決める", "調べる", "集める", "続ける", "始める",
    "終わる", "建てる", "壊す", "運ぶ", "並ぶ", "止める", "考える", "感じる", "信じる", "忘れる",
    "覚える", "選ぶ", "預ける", "貸す", "借りる", "返す", "洗う", "磨く", "直す", "見つける",
], "N4", "vocab")
_jlpt_add([
    "暖かい", "涼しい", "辛い", "甘い", "塩辛い", "苦い", "酸っぱい", "美味しい", "まずい", "柔らかい",
    "固い", "厚い", "薄い",
], "N4", "vocab")
_jlpt_add([
    "まだ", "もう", "すぐ", "ちょうど", "ちょっと", "だんだん", "段々", "やっと", "やはり", "とても",
    "たいへん", "ときどき", "時々", "いつも", "よく",
], "N4", "vocab")

_jlpt_add([
    "環境", "経営", "戦略", "課題", "目標", "計画", "進捗", "結果", "報告", "連絡",
    "依頼", "相談", "改善", "向上", "対応", "対策", "解決", "開発", "設計", "実装",
    "テスト", "導入", "活用", "利用", "準備", "参加", "応募", "面接", "採用", "退職",
    "辞める", "管理", "担当", "進行", "維持", "運用", "変更", "更新", "削除", "検索",
    "分析", "処理", "生成", "保存", "共有", "確認", "検討", "調整", "評価", "選択",
    "判断", "記録", "監視", "復旧", "対処",
], "N3", "vocab")
_jlpt_add([
    "更に", "特に", "主に", "ついて", "に対して", "比較的", "絶対に", "おそらく", "たぶん", "実は",
    "本来", "原則", "基本的", "重要", "必要", "可能", "不可能", "適切", "不適切", "効率的",
    "生産的", "効果的", "合理的", "客観的", "主観的", "継続的", "定期的", "迅速に", "確実に", "積極的",
], "N3", "vocab")

_jlpt_add([
    "事業", "企業", "産業", "市場", "競争", "優位性", "差別化", "販売", "購入", "投資",
    "融資", "株式", "決算", "経理", "財務", "人事", "総務", "法務", "知的財産", "特許",
    "商標", "著作権", "契約", "紛争", "訴訟", "管轄", "準拠法", "解除", "違反", "損害賠償",
    "免責", "秘密保持", "機密", "譲渡", "当事者", "甲", "乙", "債権", "債務", "担保",
    "保証", "履行", "催告", "抗弁", "相殺",
], "N2", "vocab")
_jlpt_add([
    "統計", "仮説", "検証", "相関", "回帰", "分散", "標準偏差", "有意差", "確率", "分布",
    "推定", "予測", "分類", "クラスタリング", "次元削減", "ベクトル空間", "埋め込み", "勾配降下", "過学習", "汎化性能",
    "精度", "再現率", "適合率", "F値", "AUC", "モデル", "アルゴリズム", "パラメータ", "ハイパーパラメータ",
], "N2", "vocab")

_jlpt_add([
    "経済", "財政", "金融", "貿易", "収支", "インフレ", "デフレ", "景気", "循環", "構造改革",
    "規制緩和", "少子高齢化", "持続可能性", "気候変動", "再生可能エネルギー", "原子力", "二酸化炭素", "排出権", "SDGs", "多様性",
    "包摂", "エクイティ", "ガバナンス", "コンプライアンス", "リスクマネジメント", "ファイナンス", "マーケティング", "ブランディング", "ステークホルダー", "デリバティブ",
    "レバレッジ", "イノベーション", "ディスラプション", "エコシステム", "パラダイムシフト", "ベストプラクティス", "ベンチマーク", "KPI", "OKR", "ROI",
    "EBITDA", "インフラストラクチャ", "レジリエンス", "脆弱性", "脅威", "侵害", "詐欺", "横領", "贈収賄", "談合",
    "カルテル", "独禁法", "行政指導", "省令", "告示", "通達", "判例", "法理", "解釈", "準則",
    "ガイドライン", "憲章", "条約", "協定", "覚書",
], "N1", "vocab")

_jlpt_add([
    "てください", "ましょう", "てもいい", "たりたりする", "ために", "ので",
], "N4", "grammar")
_jlpt_add([
    "なければなりません", "なくてもいい", "ながら", "られる", "せる", "させる", "ば", "ように", "のに",
], "N3", "grammar")
_jlpt_add([
    "ばかり", "につれて", "わけではない", "にもかかわらず", "に関する", "ものの", "ことには", "はずだ", "ようだ", "そうだ",
    "について", "に対する",
], "N2", "grammar")
_jlpt_add([
    "によると", "を余儀なくされる", "に堪えない", "かたがた", "かたわら", "がてら", "のみならず", "とあいまって",
    "に即して", "にひきかえ", "をめぐって", "ならでは", "に応じて",
], "N1", "grammar")


def _jlpt_level_of(surface: str, dict_form: str, reading: str, pos_main: str) -> Optional[str]:
    candidates = []
    for key in (surface, dict_form, reading):
        if key and key in _JLPT_LEX:
            candidates.append(_JLPT_LEX[key]["level"])
    haystack = reading or surface or ""
    for marker, info in _JLPT_LEX.items():
        if info.get("kind") == "grammar" and marker and marker in haystack:
            candidates.append(info["level"])
    if not candidates:
        return None
    candidates.sort(key=lambda L: _LEVEL_RANK.get(L, 0), reverse=True)
    return candidates[0]


@dataclass
class JlptAnalyzeResult(AnalyzeResult):
    jlpt_level: str = ""
    jlpt_score: float = 0.0
    jlpt_breakdown: Dict[str, int] = field(default_factory=dict)
    token_jlpt: List[Optional[str]] = field(default_factory=list)
    difficulty_label: str = ""

    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            "jlpt_level": self.jlpt_level,
            "jlpt_score": round(self.jlpt_score, 4),
            "jlpt_breakdown": dict(self.jlpt_breakdown),
            "token_jlpt": list(self.token_jlpt),
            "difficulty_label": self.difficulty_label,
        })
        return base


def analyze_with_jlpt(text: str, mode: str = "C") -> JlptAnalyzeResult:
    base = analyze(text, mode)
    token_jlpt: List[Optional[str]] = []
    breakdown: Dict[str, int] = {}
    scores: List[float] = []
    for t in base.tokens:
        level = _jlpt_level_of(t.surface, t.dictionary_form, t.reading, t.pos_main)
        token_jlpt.append(level)
        if level is not None:
            breakdown[level] = breakdown.get(level, 0) + 1
            scores.append(_JLPT_LEVEL_SCORE.get(level, 0.0))
    base_avg = sum(scores) / len(scores) if scores else 0.0
    te_form_count = sum(1 for t in base.tokens if t.surface.endswith("て"))
    sentence_bonus = 0.0
    if base.token_count > 20:
        sentence_bonus += 3.0
    if te_form_count >= 2:
        sentence_bonus += 3.0
    elif te_form_count >= 1:
        sentence_bonus += 1.5
    score = min(100.0, base_avg + sentence_bonus)
    jlpt_level = "N/A"
    ranked_levels = [L for L in token_jlpt if L is not None]
    if ranked_levels:
        ranked_levels.sort(key=lambda L: _LEVEL_RANK.get(L, 0), reverse=True)
        jlpt_level = ranked_levels[0]
    rank = _LEVEL_RANK.get(jlpt_level, 0)
    if rank == 0:
        difficulty_label = "Unrated"
    elif rank <= 2:
        difficulty_label = "Beginner (N5/N4)"
    elif rank == 3:
        difficulty_label = "Intermediate (N3)"
    elif rank == 4:
        difficulty_label = "Upper-Intermediate (N2)"
    else:
        difficulty_label = "Advanced (N1)"
    return JlptAnalyzeResult(
        text=base.text,
        mode=base.mode,
        token_count=base.token_count,
        token_surfaces=list(base.token_surfaces),
        tokens=list(base.tokens),
        pos_counts=dict(base.pos_counts),
        readings=list(base.readings),
        wakati=base.wakati,
        kana_only_text=base.kana_only_text,
        noun_list=list(base.noun_list),
        verb_list=list(base.verb_list),
        adjective_list=list(base.adjective_list),
        jlpt_level=jlpt_level,
        jlpt_score=round(score, 4),
        jlpt_breakdown=breakdown,
        token_jlpt=token_jlpt,
        difficulty_label=difficulty_label,
    )


# ---------------- NEW: ALTERNATIVE SENTENCE REWRITES (6 styles) ----------------

@dataclass
class AlternativeSentence:
    style: str
    text: str
    reason_jp: str
    reason_en: str

    def to_dict(self) -> Dict:
        return {
            "style": self.style,
            "text": self.text,
            "reason_jp": self.reason_jp,
            "reason_en": self.reason_en,
        }


_ROMAJI: Dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo",
    "ャ": "ya", "ュ": "yu", "ョ": "yo",
    "ヴ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
}

_SMALL_COMBOS = frozenset({"ゃ", "ゅ", "ょ", "ャ", "ュ", "ョ"})


def _kata_to_hira(s: str) -> str:
    out: List[str] = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _kana_to_romaji(kana: str) -> str:
    out: List[str] = []
    small_buffer = False
    prev_romaji_tail = ""
    i = 0
    while i < len(kana):
        ch = kana[i]
        if ch == "っ" or ch == "ッ":
            small_buffer = True
            i += 1
            continue
        if ch == "ー" and out:
            last = out[-1]
            if last:
                out.append(last[-1])
            i += 1
            continue
        if ch in _SMALL_COMBOS and out:
            combo_rom = _ROMAJI.get(ch, ch)
            last = out.pop()
            if len(last) >= 2:
                consonant_head = last[:-1]
            else:
                consonant_head = last
            combined = consonant_head + combo_rom
            out.append(combined)
            i += 1
            continue
        rom = _ROMAJI.get(ch, ch)
        if small_buffer and rom and rom[0] in "bcdfghjklmnpqrstvwxyz":
            out.append(rom[0])
            small_buffer = False
        out.append(rom)
        i += 1
    return "".join(out)


_JA_SYNONYMS: Dict[str, str] = {
    "開発": "研究開発", "研究": "調査", "報告": "報告事項", "改善": "改良", "検討": "考察",
    "説明": "解説", "対応": "対応措置", "向上": "改善向上", "目標": "目的", "課題": "問題点",
    "結果": "成果", "計画": "企画", "進捗": "進行状況", "会議": "打ち合わせ", "仕事": "業務",
    "試験": "検査", "質問": "問い合わせ", "答え": "回答", "意見": "見解", "問題": "課題",
    "機能": "機能性", "方法": "手法", "情報": "知見", "内容": "中身", "詳細": "明細",
    "全体": "全体像", "部分": "一部", "理由": "原因", "影響": "作用", "効果": "成果",
    "利点": "メリット", "欠点": "デメリット", "価値": "値打ち", "品質": "クオリティ", "生産": "製造",
    "消費": "利用", "供給": "提供", "需要": "要求", "増加": "上昇", "減少": "低下",
    "変化": "推移", "継続": "持続", "停止": "中止", "開始": "着手", "終了": "完了",
    "参加": "参画", "応募": "申し込み", "導入": "導入施策", "活用": "利活用", "利用": "使用",
    "共有": "共有化", "管理": "マネジメント", "運用": "オペレーション", "対策": "対処法", "解決": "解消",
    "分析": "解析", "処理": "執務", "検索": "探索", "比較": "対比", "選択": "選定",
    "決定": "確定", "確認": "照合", "保存": "保管", "更新": "改訂", "削除": "廃棄",
    "担当": "責任", "提出": "提出書類", "締切": "期限", "予算": "予算額", "費用": "コスト",
    "売上": "売上高", "利益": "収益", "顧客": "得意先", "取引先": "パートナー", "従業員": "スタッフ",
    "教育": "研修", "研修": "トレーニング", "面接": "インタビュー", "募集": "求人", "採用": "雇用",
    "退職": "離職", "能力": "スキル", "経験": "キャリア", "希望": "要望", "連絡": "通知",
    "相談": "カウンセリング", "依頼": "委託", "見積": "お見積もり", "契約": "合意", "資料": "ドキュメント",
    "返信": "回答", "質疑": "Q&A", "効率": "生産性", "戦略": "戦術", "理念": "ビジョン",
    "目的": "ゴール", "手段": "アプローチ", "評価": "査定", "認識": "理解", "認可": "承認",
    "許可": "免許", "登録": "記載", "発行": "交付", "受領": "受け取り", "送付": "発送",
    "対象": "該当者", "範囲": "領域", "基準": "水準", "条件": "前提", "状況": "状態",
    "体制": "仕組み", "環境": "状況", "背景": "経緯", "経緯": "いきさつ", "特徴": "特性",
    "メリット": "長所", "デメリット": "短所", "チャンス": "機会", "リスク": "危険性", "タスク": "作業",
    "プロジェクト": "事業", "プロセス": "手順", "フィードバック": "反省", "サポート": "支援", "コメント": "所見",
    "結果": "帰結", "要因": "因子", "仕組み": "システム", "立場": "ポジション", "現状": "ステータス",
    "内容": "コンテンツ", "商品": "プロダクト", "顧客": "クライアント", "従業員": "メンバー", "活動": "アクティビティ",
    "質問": "クエスチョン", "推奨": "レコメンド", "入力": "インプット", "出力": "アウトプット",
}

_KEIGO_MAP_14: Dict[str, str] = {
    "食べる": "召し上がる",
    "飲む": "召し上がる",
    "行く": "いらっしゃる",
    "来る": "おいでになる",
    "見る": "ご覧になる",
    "聞く": "伺う",
    "訪ねる": "伺う",
    "する": "なさる",
    "言う": "おっしゃる",
    "寝る": "お休みになる",
    "知る": "ご存知",
    "持つ": "お持ちになる",
    "着る": "お召しになる",
    "置く": "お置きになる",
    "買う": "お求めになる",
}


def _u_stem(s: str) -> str:
    if not s:
        return s
    gojuon_a = {
        "う": "わ", "く": "か", "ぐ": "が", "す": "さ", "つ": "た",
        "ぬ": "な", "ぶ": "ば", "む": "ま", "る": "ら",
    }
    gojuon_i = {
        "う": "い", "く": "き", "ぐ": "ぎ", "す": "し", "つ": "ち",
        "ぬ": "に", "ぶ": "び", "む": "み", "る": "り",
    }
    last = s[-1]
    if last in gojuon_i:
        return s[:-1] + gojuon_i[last]
    return s


_IRREGULAR_ICHIDAN = frozenset({
    "見る", "着る", "寝る", "起きる", "降りる", "借りる", "足りる", "いる", "出来る",
    "できる", "受ける", "続ける", "始める", "開ける", "閉める", "教える", "決める",
    "集める", "忘れる", "覚える", "感じる", "考える", "届ける", "信じる",
})


def _is_ichidan_verb(df: str) -> bool:
    if not df.endswith("る") or len(df) < 2:
        return False
    if df in _IRREGULAR_ICHIDAN:
        return True
    return df[-2] in "いきぎしじちぢにひびぴみりえけげせぜてでねへべぺめれ"


def _rewrite_polite(s: str, tokens: List[TokenInfo]) -> str:
    if "ます" in s or "です" in s:
        return s
    out: List[str] = []
    changed_any = False
    for t in tokens:
        df = t.dictionary_form or t.surface
        surf = t.surface
        if t.pos_main == "動詞":
            if df == "する" or df.endswith("する"):
                if surf.endswith("た"):
                    out.append("しました")
                elif "ている" in surf:
                    out.append(surf.replace("ている", "ています"))
                else:
                    out.append("します")
                changed_any = True
                continue
            if df in ("来る", "くる"):
                if surf.endswith("た") or "来た" in surf or "きた" in surf:
                    out.append("来ました")
                else:
                    out.append("来ます")
                changed_any = True
                continue
            if _is_ichidan_verb(df):
                stem = df[:-1]
                if surf.endswith("た"):
                    out.append(stem + "ました")
                elif "ている" in surf:
                    out.append(surf.replace("ている", "ています"))
                else:
                    out.append(stem + "ます")
                changed_any = True
                continue
            stem = _u_stem(df)
            if surf.endswith("た"):
                out.append(stem + "ました")
            elif "ている" in surf:
                out.append(surf.replace("ている", "ています"))
            else:
                out.append(stem + "ます")
            changed_any = True
            continue
        if t.pos_main == "助動詞" and (df == "だ" or df == "である"):
            if surf.endswith("た") or surf == "だった":
                out.append("でした")
            else:
                out.append("です")
            changed_any = True
            continue
        if df == "ない":
            if surf.endswith("かった"):
                out.append("ありませんでした")
            else:
                out.append("ありません")
            changed_any = True
            continue
        out.append(surf)
    result = "".join(out)
    if not changed_any:
        if result and result[-1] not in "。！？":
            result += "です"
        elif result.endswith("。"):
            result = result[:-1] + "です。"
    return result


def _rewrite_casual(s: str, tokens: List[TokenInfo]) -> str:
    result = s
    ordered_pairs = [
        ("ませんでした", "なかった"),
        ("でしょう", "だろう"),
        ("ましょう", "よう"),
        ("でした", "だった"),
        ("ました", "た"),
        ("ております", "てる"),
        ("ています", "てる"),
        ("ません", "ない"),
        ("します", "する"),
        ("きます", "くる"),
        ("ですか", "だい"),
        ("ますか", "るかい"),
        ("ください", "ちょうだい"),
        ("です", "だ"),
        ("ます", "る"),
    ]
    changed = False
    for old, new in ordered_pairs:
        if old in result:
            result = result.replace(old, new)
            changed = True
    if not changed and "ます" in result:
        result = result.replace("ます", "")
    return result


def _rewrite_kana_easy(s: str, tokens: List[TokenInfo]) -> str:
    out: List[str] = []
    for t in tokens:
        if t.surface in "。！？、":
            out.append(t.surface)
            continue
        r = t.reading or t.surface
        out.append(_kata_to_hira(r))
    return "".join(out)


def _rewrite_romaji(s: str, tokens: List[TokenInfo]) -> str:
    token_roms: List[str] = []
    for t in tokens:
        reading_text = t.reading if t.reading else t.surface
        rom = _kana_to_romaji(reading_text)
        if rom.strip():
            token_roms.append(rom)
    return " ".join(token_roms)


def _rewrite_synonyms(s: str, tokens: List[TokenInfo]) -> str:
    freq: Dict[str, int] = {}
    for t in tokens:
        if t.pos_main == "名詞" and t.surface not in _JA_STOPWORDS and len(t.surface) >= 2:
            freq[t.surface] = freq.get(t.surface, 0) + 1
    sorted_nouns = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    top = [n for n, _ in sorted_nouns[:3]]
    result = s
    replaced = set()
    for n in top:
        if n in _JA_SYNONYMS and n not in replaced:
            syn = _JA_SYNONYMS[n]
            result = result.replace(n, syn, 1)
            replaced.add(n)
    return result


def _rewrite_honorific_light(s: str, tokens: List[TokenInfo]) -> str:
    out: List[str] = []
    for t in tokens:
        df = t.dictionary_form or t.surface
        surf = t.surface
        if df in _KEIGO_MAP_14:
            honorific = _KEIGO_MAP_14[df]
            has_masu = "ます" in surf or "ました" in surf
            past = "ました" in surf
            if has_masu and honorific in ("なさる", "おっしゃる", "いらっしゃる", "おいでになる", "ご覧になる", "召し上がる", "伺う", "お休みになる", "お持ちになる", "お召しになる", "お置きになる", "お求めになる"):
                if honorific == "なさる":
                    stem = "なさい"
                elif honorific == "おっしゃる":
                    stem = "おっしゃい"
                elif honorific == "いらっしゃる":
                    stem = "いらっしゃい"
                elif honorific.endswith("る"):
                    stem = honorific[:-1]
                else:
                    stem = honorific
                suffix = "ました" if past else "ます"
                out.append(stem + suffix)
            else:
                out.append(honorific)
        else:
            out.append(surf)
    result = "".join(out)
    fixes = [
        ("なさるます", "なさいます"),
        ("なさるました", "なさいました"),
        ("おっしゃるます", "おっしゃいます"),
        ("おっしゃるました", "おっしゃいました"),
        ("いらっしゃるます", "いらっしゃいます"),
        ("いらっしゃるました", "いらっしゃいました"),
    ]
    for bad, good in fixes:
        result = result.replace(bad, good)
    return result


from typing import Any


def generate_alternatives(text: str, mode: str = "C") -> List[Dict[str, Any]]:
    tok = tokenize(text, mode)
    tokens = tok.tokens
    try:
        polite = AlternativeSentence(
            style="polite",
            text=_rewrite_polite(text, tokens),
            reason_jp="丁寧語で表現を整え、ビジネスや目上の人との対話に適した形にしました。",
            reason_en="Polished with です/ます polite register, suitable for business or conversations with superiors.",
        )
    except Exception:
        polite = AlternativeSentence(style="polite", text=text, reason_jp="丁寧語化", reason_en="Polite conversion.")
    try:
        casual = AlternativeSentence(
            style="casual",
            text=_rewrite_casual(text, tokens),
            reason_jp="親しい間柄で使えるよう、だ・である体や短い語尾に置き換えました。",
            reason_en="Swapped to casual plain style and short endings for close relationships.",
        )
    except Exception:
        casual = AlternativeSentence(style="casual", text=text, reason_jp="口語化", reason_en="Casual conversion.")
    try:
        kana_easy = AlternativeSentence(
            style="kana_easy",
            text=_rewrite_kana_easy(text, tokens),
            reason_jp="漢字の読みにくさをなくし、すべてひらがなで表記することで学習初級者にも読めるようにしました。",
            reason_en="Replaced with all-hiragana readings so beginners without kanji knowledge can read it.",
        )
    except Exception:
        kana_easy = AlternativeSentence(style="kana_easy", text=text, reason_jp="ひらがな化", reason_en="All-hiragana version.")
    try:
        romaji = AlternativeSentence(
            style="romaji",
            text=_rewrite_romaji(text, tokens),
            reason_jp="日本語キーボードがなくても読めるよう、ローマ字表記に変換しました。",
            reason_en="Converted to romaji so it can be read without a Japanese keyboard or kana knowledge.",
        )
    except Exception:
        romaji = AlternativeSentence(style="romaji", text=text, reason_jp="ローマ字化", reason_en="Romaji version.")
    try:
        synonym = AlternativeSentence(
            style="synonym_swap",
            text=_rewrite_synonyms(text, tokens),
            reason_jp="主要な名詞を表記ゆれや同義語に差し替え、バリエーションを持たせました。",
            reason_en="Top-occurrence nouns swapped with close synonyms to provide a phrasing variation.",
        )
    except Exception:
        synonym = AlternativeSentence(style="synonym_swap", text=text, reason_jp="類語置換", reason_en="Synonym swap variation.")
    try:
        honorific = AlternativeSentence(
            style="honorific",
            text=_rewrite_honorific_light(text, tokens),
            reason_jp="動作主体を敬う尊敬表現（いらっしゃる・なさる・おっしゃるなど）に置き換え、敬意を表す形にしました。",
            reason_en="Replaced verbs with sonkeigo honorifics (いらっしゃる, なさる, おっしゃる etc.) to show respect to the subject.",
        )
    except Exception:
        honorific = AlternativeSentence(style="honorific", text=text, reason_jp="敬語化", reason_en="Honorific version.")
    alts = [polite, casual, kana_easy, romaji, synonym, honorific]
    return [a.to_dict() for a in alts]


# ============================================================
# RICH PROFICIENCY ASSESSMENT (reading difficulty + JLPT level)
# ============================================================

@dataclass
class ProficiencyAssessment:
    """Comprehensive Japanese proficiency / readability assessment.

    Combines:
      - Overall JLPT level estimate (N5..N1) with confidence
      - Readability score (0..100, higher = easier for a native beginner)
      - Vocabulary / Kanji / Grammar breakdowns
      - Per-level token counts and unknown ratio
      - CEFR approximation for broader audience
    """
    text: str
    mode: str
    overall_jlpt: str
    overall_jlpt_score: float      # 0..100
    confidence: float             # 0..1
    readability_score: float      # 0..100
    cefr_approx: str              # A1..C2
    difficulty_label: str
    # Breakdown
    total_tokens: int
    known_tokens: int
    unknown_tokens: int
    unknown_ratio: float
    kanji_count: int
    kanji_per_100chars: float
    average_kanji_per_word: float
    unique_kanji_count: int
    level_counts: Dict[str, int] = field(default_factory=dict)
    level_percentages: Dict[str, float] = field(default_factory=dict)
    # Top hardest vocabulary found (by JLPT rank)
    hardest_words: List[Dict[str, Any]] = field(default_factory=list)
    # Suggestions
    learning_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode,
            "overall_jlpt": self.overall_jlpt,
            "overall_jlpt_score": round(self.overall_jlpt_score, 3),
            "confidence": round(self.confidence, 3),
            "readability_score": round(self.readability_score, 3),
            "cefr_approx": self.cefr_approx,
            "difficulty_label": self.difficulty_label,
            "total_tokens": self.total_tokens,
            "known_tokens": self.known_tokens,
            "unknown_tokens": self.unknown_tokens,
            "unknown_ratio": round(self.unknown_ratio, 4),
            "kanji_count": self.kanji_count,
            "kanji_per_100chars": round(self.kanji_per_100chars, 3),
            "average_kanji_per_word": round(self.average_kanji_per_word, 3),
            "unique_kanji_count": self.unique_kanji_count,
            "level_counts": dict(self.level_counts),
            "level_percentages": {k: round(v, 2) for k, v in self.level_percentages.items()},
            "hardest_words": self.hardest_words[:15],
            "learning_notes": list(self.learning_notes),
        }


def _count_kanji(s: str) -> int:
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff")


def _unique_kanji(s: str) -> int:
    return len({c for c in s if "\u4e00" <= c <= "\u9fff"})


_CEFR_FROM_JLPT: Dict[str, str] = {
    "N5": "A1",
    "N4": "A2",
    "N3": "B1",
    "N2": "B2",
    "N1": "C1",
}


def assess_proficiency(text: str, mode: str = "C") -> ProficiencyAssessment:
    """Analyze a sentence / paragraph for Japanese proficiency level.

    Returns a rich ProficiencyAssessment including overall JLPT level,
    readability score, CEFR approximation, hardest vocabulary and
    learning notes for a learner.
    """
    if not isinstance(text, str) or not text.strip():
        return ProficiencyAssessment(
            text=text or "",
            mode=mode,
            overall_jlpt="N/A",
            overall_jlpt_score=0.0,
            confidence=0.0,
            readability_score=100.0,
            cefr_approx="A1",
            difficulty_label="Empty",
            total_tokens=0,
            known_tokens=0,
            unknown_tokens=0,
            unknown_ratio=0.0,
            kanji_count=0,
            kanji_per_100chars=0.0,
            average_kanji_per_word=0.0,
            unique_kanji_count=0,
        )

    tok = tokenize(text, mode)
    level_counts: Dict[str, int] = {"N5": 0, "N4": 0, "N3": 0, "N2": 0, "N1": 0}
    score_list: List[float] = []
    known = 0
    unknown = 0
    hardest: List[Tuple[int, float, TokenInfo]] = []

    for t in tok.tokens:
        # Skip punctuation and pure number-only tokens
        if t.pos_main in ("補助記号", "記号"):
            continue
        if t.surface.isdigit() or all(not ord(c) > 127 and not c.isalpha() for c in t.surface):
            pass  # count numbers, still useful
        level = _jlpt_level_of(t.surface, t.dictionary_form, t.reading, t.pos_main)
        # Fallback to the global jlpt_level_for function (dictionary form)
        if level is None:
            level = jlpt_level_for(t.surface, reading=t.reading, dictionary_form=t.dictionary_form)
        if level is not None and level in level_counts:
            level_counts[level] += 1
            score_list.append(_JLPT_LEVEL_SCORE.get(level, 0.0))
            known += 1
            rank = _LEVEL_RANK.get(level, 0)
            if rank >= 3:  # N3 or harder — track as potentially "hard"
                hardest.append((rank, _JLPT_LEVEL_SCORE.get(level, 0.0), t))
        else:
            unknown += 1
            # Kanji-complexity heuristic fallback
            kb = _jlpt_kanji_fallback(t.surface)
            if kb and kb in level_counts:
                level_counts[kb] = level_counts.get(kb, 0) + 1
                score_list.append(_JLPT_LEVEL_SCORE.get(kb, 0.0) * 0.6)  # less confident
                known += 1

    total_tokens = max(1, known + unknown)
    unknown_ratio = unknown / total_tokens

    # Overall JLPT: weighted 60% hardest-decile, 40% average
    avg_score = sum(score_list) / len(score_list) if score_list else 0.0
    score_list_sorted = sorted(score_list, reverse=True)
    top_tail = score_list_sorted[: max(1, len(score_list_sorted) // 5)]
    hard_score = sum(top_tail) / len(top_tail) if top_tail else avg_score
    overall_score = 0.6 * hard_score + 0.4 * avg_score

    # Map back to JLPT band
    if overall_score >= 92:
        overall_jlpt = "N1"
    elif overall_score >= 76:
        overall_jlpt = "N2"
    elif overall_score >= 58:
        overall_jlpt = "N3"
    elif overall_score >= 36:
        overall_jlpt = "N4"
    elif overall_score > 0:
        overall_jlpt = "N5"
    else:
        overall_jlpt = "N/A"

    # Confidence: fraction of classified tokens
    confidence = max(0.0, min(1.0, (known / total_tokens) * (1.0 - unknown_ratio * 0.5)))

    # Kanji metrics
    total_chars = max(1, len(text))
    kanji_count = _count_kanji(text)
    unique_kanji = _unique_kanji(text)
    kanji_per_100 = (kanji_count / total_chars) * 100.0
    avg_k_per_w = (kanji_count / max(1, tok.token_count))

    # Readability score (higher = easier). Factors:
    #   - fewer kanji per 100 chars -> easier
    #   - lower unknown ratio -> easier
    #   - more N5/N4 tokens -> easier
    easy_pct = ((level_counts.get("N5", 0) + level_counts.get("N4", 0)) / total_tokens) * 100.0
    hard_pct = ((level_counts.get("N1", 0) + level_counts.get("N2", 0)) / total_tokens) * 100.0
    readability = (
        + 50.0
        + easy_pct * 0.6
        - hard_pct * 0.7
        - kanji_per_100 * 0.7
        - unknown_ratio * 60.0
        - min(50.0, avg_k_per_w * 20.0)
    )
    readability = max(0.0, min(100.0, readability))

    # CEFR & label
    cefr = _CEFR_FROM_JLPT.get(overall_jlpt, "A1" if overall_jlpt == "N5" else "B2")
    if overall_jlpt == "N/A":
        cefr = "A1"
        difficulty_label = "Unrated (all tokens unknown or punctuation)"
    elif overall_jlpt in ("N5", "N4"):
        difficulty_label = "Beginner / Elementary"
    elif overall_jlpt == "N3":
        difficulty_label = "Intermediate"
    elif overall_jlpt == "N2":
        difficulty_label = "Upper-Intermediate"
    else:
        difficulty_label = "Advanced / Near-Native"

    # Level percentages
    total_known = max(1, known)
    level_pct = {lv: (cnt / total_known) * 100.0 for lv, cnt in level_counts.items()}

    # Hardest words (descending rank)
    hardest.sort(key=lambda x: (-x[0], -x[1]))
    hardest_words: List[Dict[str, Any]] = []
    seen_keys = set()
    for rank, sc, t in hardest:
        key = t.dictionary_form or t.surface
        if key in seen_keys:
            continue
        seen_keys.add(key)
        jlpt_here = None
        for lvl in ("N1", "N2", "N3", "N4", "N5"):
            if _LEVEL_RANK.get(lvl) == rank:
                jlpt_here = lvl
                break
        hardest_words.append({
            "word": t.surface,
            "reading": t.reading,
            "dictionary_form": t.dictionary_form,
            "pos": t.pos_main,
            "jlpt": jlpt_here,
            "score": round(sc, 2),
            "romaji": _kana_to_romaji(t.reading) if t.reading else "",
        })
        if len(hardest_words) >= 20:
            break

    # Learning notes
    notes: List[str] = []
    if overall_jlpt != "N/A":
        notes.append(
            f"This text is roughly {overall_jlpt} ({cefr}) level. "
            f"Learners at this level should recognise ~{min(95, int((1-unknown_ratio)*100))}% of tokens."
        )
    if hard_pct > 20:
        notes.append(
            f"Advanced vocabulary is dense ({round(hard_pct,1)}% N1/N2 tokens). "
            "Pre-learn the 'hardest words' list before reading aloud."
        )
    if kanji_per_100 > 25:
        notes.append(
            f"Kanji density is high ({round(kanji_per_100,1)} per 100 chars). "
            "Readers below N3 should enable kana-only or romaji alternatives."
        )
    if unknown_ratio > 0.35:
        notes.append(
            f"High unknown-token ratio ({round(unknown_ratio*100,1)}%). "
            "A larger JLPT lexicon is needed — consider adding more N{min(3, 2+1)} vocabulary."
        )
    if readability < 40:
        notes.append("Readability is low — consider providing a kana-easy rewrite for learners.")

    return ProficiencyAssessment(
        text=text,
        mode=mode,
        overall_jlpt=overall_jlpt,
        overall_jlpt_score=overall_score,
        confidence=confidence,
        readability_score=readability,
        cefr_approx=cefr,
        difficulty_label=difficulty_label,
        total_tokens=total_tokens,
        known_tokens=known,
        unknown_tokens=unknown,
        unknown_ratio=unknown_ratio,
        kanji_count=kanji_count,
        kanji_per_100chars=kanji_per_100,
        average_kanji_per_word=avg_k_per_w,
        unique_kanji_count=unique_kanji,
        level_counts=level_counts,
        level_percentages=level_pct,
        hardest_words=hardest_words,
        learning_notes=notes,
    )
