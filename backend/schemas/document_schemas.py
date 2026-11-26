from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocFormat = Literal["docx", "pdf"]


class NlpReportRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text to include in the report")
    mode: Literal["A", "B", "C"] = Field("C", description="Sudachi split mode")
    title: str = Field("KANAME NLP Analysis Report", max_length=200)
    include_bilingual: bool = Field(True, description="Include English translations section")
    doc_format: DocFormat = Field("docx", description="Output format (docx or pdf)")
    author: str | None = Field("KANAME [要]", max_length=100)


class NlpReportResponse(BaseModel):
    filename: str
    format: str
    size_bytes: int
    download_url: str | None = None
    message: str


class DocumentHealthResponse(BaseModel):
    status: str
    engine: str
    supported_formats: list[str]
    available_generators: list[str]


class BatchDocumentItem(BaseModel):
    template_id: str
    payload: dict[str, Any]


class BatchDocumentRequest(BaseModel):
    items: list[BatchDocumentItem] = Field(..., min_length=1, max_length=20)
    auto_translate: bool = True


class OutlineDraftRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=1000)
    audience: str = Field("", max_length=500)
    language: Literal["ja", "en"] = "ja"
    section_count: int = Field(5, ge=3, le=12)


class TemplateFieldInfo(BaseModel):
    name: str
    label_jp: str
    label_en: str
    field_type: Literal["string", "text", "date", "integer", "number", "boolean", "select", "list", "object_list"]
    required: bool = False
    placeholder: str | None = None
    description_jp: str | None = None
    description_en: str | None = None
    options: list[dict[str, str]] | None = None
    default: Any | None = None
    children: list[TemplateFieldInfo] | None = None


class DocumentTemplateInfo(BaseModel):
    template_id: str
    name_jp: str
    name_en: str
    category: Literal["employment", "finance", "business", "correspondence", "report"]
    description_jp: str
    description_en: str
    typical_use_case_jp: str
    typical_use_case_en: str
    formats: list[DocFormat]
    fields: list[TemplateFieldInfo]


# ----------- 履歴書 Resume -----------
class ResumeEducation(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    school_name: str
    department: str | None = None
    degree: str | None = None
    is_graduated: bool = True


class ResumeWorkHistory(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    company_name: str
    position: str | None = None
    employment_type: Literal["full_time", "contract", "temp", "outsourcing", "part_time"] | None = Field(
        None,
        description="雇用形態：正社員/契約社員/派遣/業務委託/アルバイト",
    )
    description: str | None = None
    resign_reason: str | None = Field(None, max_length=300, description="退職理由（任意）")
    is_current: bool = False


class ResumeLicense(BaseModel):
    acquisition_date: str | None = None
    license_name: str
    issuer: str | None = None


class ResumeFamilyMember(BaseModel):
    relationship: str = Field(..., description="続柄（配偶者/子/父/母 etc）")
    name: str | None = None
    age: int | None = None
    is_dependent: bool | None = Field(False, description="扶養家族かどうか")


class ResumeRequest(BaseModel):
    full_name_kanji: str = Field(..., max_length=64, description="氏名（漢字）")
    full_name_kana: str = Field(..., max_length=128, description="氏名（ふりがな）")
    issue_date: str | None = Field(None, description="作成日 / 記入日 (YYYY-MM-DD)。空欄は今日の日付")
    gender: Literal["male", "female", "other"] = Field("other")
    date_of_birth: str | None = Field(None, description="YYYY-MM-DD")
    marital_status: Literal["single", "married", "other"] | None = Field(None, description="婚姻状態：未婚/既婚/その他")
    family_members: list[ResumeFamilyMember] = Field(default_factory=list, description="家族構成（続柄・氏名・年齢）")
    phone: str | None = Field(None, max_length=32)
    email: str | None = Field(None, max_length=128)
    postal_code: str | None = Field(None, max_length=16)
    address_prefecture: str | None = Field(None, max_length=32)
    address_city: str | None = Field(None, max_length=128)
    address_detail: str | None = Field(None, max_length=256)
    education_list: list[ResumeEducation] = Field(default_factory=list)
    work_history_list: list[ResumeWorkHistory] = Field(default_factory=list)
    licenses_list: list[ResumeLicense] = Field(default_factory=list)
    self_pr: str | None = Field(None, max_length=800, description="自己PR")
    motivation: str | None = Field(None, max_length=800, description="志望動機")
    hobbies: str | None = Field(None, max_length=400, description="趣味・特技")
    requests: str | None = Field(None, max_length=800, description="本人希望記入欄（給料・職種・勤務時間・勤務地など）")
    photo_path: str | None = Field(None, description="顔写真ファイルパス（PNG/JPEG）。未指定の場合は「顔写真貼付欄」プレースホルダ")
    doc_format: DocFormat = Field("docx")


# ----------- 職務経歴書 Career History -----------
class CareerProject(BaseModel):
    period_from: str | None = None
    period_to: str | None = None
    project_name: str
    role: str | None = None
    team_size: int | None = None
    technologies: list[str] = Field(default_factory=list)
    overview: str | None = None
    achievements: list[str] = Field(default_factory=list)


class CareerRole(BaseModel):
    period_from: str
    period_to: str | None = None
    company_name: str
    position: str
    department: str | None = None
    overview: str | None = None
    projects: list[CareerProject] = Field(default_factory=list)


class CareerHistoryRequest(BaseModel):
    full_name_kanji: str
    full_name_kana: str | None = None
    phone: str | None = None
    email: str | None = None
    issue_date: str | None = Field(None, description="作成日 (YYYY-MM-DD)。空欄は今日の日付")
    career_summary: str | None = Field(None, max_length=1200, description="職務要約")
    roles: list[CareerRole] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list, description="保有スキル一覧")
    tech_stack: dict[str, list[str]] = Field(default_factory=dict, description="技術スタック")
    education: str | None = None
    certifications: list[str] = Field(default_factory=list)
    doc_format: DocFormat = Field("docx")


# ----------- 請求書 Invoice -----------
class InvoiceItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float
    tax_rate: float | None = Field(0.10, description="Default 10% consumption tax")


class InvoiceRequest(BaseModel):
    invoice_number: str
    issue_date: str | None = None
    due_date: str | None = None
    seller_name: str
    seller_department: str | None = None
    seller_address: str | None = None
    seller_phone: str | None = None
    seller_tax_id: str | None = Field(None, description="適格請求書発行事業者登録番号 T-xxxxxxxx")
    buyer_name: str
    buyer_department: str | None = None
    buyer_address: str | None = None
    title: str | None = Field("請求書")
    items: list[InvoiceItem] = Field(default_factory=list)
    subtotal: float | None = None
    total_tax: float | None = None
    total_amount: float | None = None
    notes: str | None = Field(None, max_length=600, description="特記事項")
    payment_method: Literal["bank_transfer", "credit_card", "cash", "konbini"] | None = "bank_transfer"
    bank_name: str | None = None
    bank_branch: str | None = None
    bank_account_type: Literal["ordinary", "current"] | None = None
    bank_account_number: str | None = None
    bank_account_holder: str | None = None
    doc_format: DocFormat = Field("docx")


# ----------- 一般報告書 General Business Report -----------
class BusinessReportRequest(BaseModel):
    title: str
    subtitle: str | None = None
    author: str
    author_department: str | None = None
    report_date: str | None = None
    recipient: str | None = Field(None, description="提出先")
    recipient_department: str | None = None
    executive_summary: str | None = Field(None, description="概要")
    background: str | None = Field(None, description="背景")
    objectives: list[str] = Field(default_factory=list, description="目的")
    methods: str | None = Field(None, description="調査・実施方法")
    results: str | None = Field(None, description="結果")
    discussion: str | None = Field(None, description="考察")
    issues: list[str] = Field(default_factory=list, description="課題")
    conclusion: str | None = Field(None, description="結論")
    recommendations: list[str] = Field(default_factory=list, description="提言・今後の方針")
    references: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    doc_format: DocFormat = Field("docx")


# ----------- 送付状 Cover Letter / Transmittal -----------
class CoverLetterRequest(BaseModel):
    sender_company: str | None = None
    sender_department: str | None = None
    sender_name: str
    sender_postal_code: str | None = None
    sender_address: str | None = None
    sender_phone: str | None = None
    sender_email: str | None = None
    recipient_company: str
    recipient_department: str | None = None
    recipient_title: str | None = None
    recipient_name: str = Field("御中")
    letter_date: str | None = None
    subject: str
    greeting: str | None = Field("お世話になっております。")
    body_paragraphs: list[str] = Field(default_factory=list, description="本文パラグラフ（複数可）")
    closing: str | None = Field("ご確認の程、何卒よろしくお願い申し上げます。")
    enclosures: list[str] = Field(default_factory=list, description="送付物一覧")
    enclosures_title: str | None = Field("添付書類一覧")
    doc_format: DocFormat = Field("docx")


# ----------- 会社データ分析報告書 Company Data Analysis Report -----------
class CompanyAnalysisReportRequest(BaseModel):
    report_title: str = Field("会社データ分析報告書", max_length=200)
    report_date: str | None = Field(None, description="YYYY-MM-DD (空欄は今日)")
    period_covered: str | None = Field(None, description="例：2026年7月〜9月")
    company_name: str = Field(..., max_length=200)
    department: str | None = Field(None, max_length=200)
    author: str = Field("KANAME [要]", max_length=100, description="報告書作成者")
    author_name: str | None = Field(None, max_length=100, description="報告書作成者（エイリアス：author と同じ）")
    analyst_name: str | None = Field(None, max_length=100)
    recipient: str | None = Field(None, description="提出先")
    recipient_name: str | None = Field(None, description="提出先（エイリアス：recipient と同じ）")
    source_filename: str | None = Field(None, description="データソースファイル名")
    dataset_id: str | None = Field(None, description="保存済み dataset_id （未指定の場合は file を使用）")
    analysis_id: str | None = Field(None, description="将来：analysis_run_id")
    iqr_k: float = Field(1.5, ge=0.1, le=10.0, description="IQR 外れ値 k 係数")
    corr_method: Literal["pearson", "spearman", "kendall"] = Field("pearson")
    executive_summary_override: str | None = Field(None, description="空なら自動生成")
    conclusion_override: str | None = Field(None, description="空なら自動生成")
    recommendations_override: list[str] = Field(default_factory=list, description="空なら自動生成")
    override_discussion: str | None = Field(None, description="【上書き】考察・示唆 (空なら自動生成)")
    override_action_plan: list[str] = Field(default_factory=list, description="【上書き】アクションプラン (空なら自動生成)")
    override_report_title: str | None = Field(None, description="【上書き】表紙タイトル")
    include_appendix: bool = Field(True, description="付録（サンプル行・生データ）を含む")
    doc_format: DocFormat = Field("docx")


# ----------- 議事録 Meeting Minutes -----------
class MeetingActionItem(BaseModel):
    task: str = Field(..., description="アクションアイテムのタスク内容")
    owner: str | None = Field(None, description="担当者")
    due_date: str | None = Field(None, description="期限 (YYYY-MM-DD)")


class MeetingMinutesRequest(BaseModel):
    title: str = Field("議事録", max_length=200, description="表題")
    meeting_date: str | None = Field(None, description="開催日 (YYYY-MM-DD)")
    start_time: str | None = Field(None, description="開始時刻 例：14:00")
    end_time: str | None = Field(None, description="終了時刻 例：15:30")
    location: str | None = Field(None, description="開催場所")
    meeting_type: Literal["regular", "extraordinary", "emergency"] | None = Field(
        "regular", description="会議種別：定例/臨時/緊急"
    )
    chairperson: str | None = Field(None, description="議長")
    recorder: str | None = Field(None, description="書記")
    attendees: list[str] = Field(default_factory=list, description="出席者")
    absentees: list[str] = Field(default_factory=list, description="欠席者")
    agenda: list[str] = Field(default_factory=list, description="議題")
    discussion: str | None = Field(None, description="議事内容（本文）")
    decisions: list[str] = Field(default_factory=list, description="決定事項")
    action_items: list[MeetingActionItem] = Field(default_factory=list, description="アクションアイテム")
    next_meeting_date: str | None = Field(None, description="次回開催日 (YYYY-MM-DD)")
    doc_format: DocFormat = Field("docx")


# ----------- 自由書式文書 Custom / User-defined Document -----------
class CustomSection(BaseModel):
    heading: str = Field(..., description="セクション見出し")
    body: str = Field("", description="セクション本文")


class CustomField(BaseModel):
    label: str = Field(..., description="項目名")
    value: str = Field("", description="値")


class CustomDocumentRequest(BaseModel):
    title: str = Field(..., max_length=200, description="表題")
    author: str | None = Field(None, description="作成者")
    document_date: str | None = Field(None, description="文書日付 (YYYY-MM-DD)")
    recipient: str | None = Field(None, description="宛先")
    intro: str | None = Field(None, description="前書き・導入文")
    sections: list[CustomSection] = Field(default_factory=list, description="本文セクション（見出し＋本文）")
    fields: list[CustomField] = Field(default_factory=list, description="カスタム項目（ラベル＋値）")
    closing: str | None = Field(None, description="結語・締めの文")
    doc_format: DocFormat = Field("docx")
