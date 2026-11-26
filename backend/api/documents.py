from __future__ import annotations

import io
import os
import tempfile
import zipfile
from typing import Any
from urllib.parse import quote
from urllib.parse import quote as _urlquote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session


def _content_disposition_headers(filename: str) -> dict[str, str]:
    ascii_name = "".join(c for c in filename if ord(c) < 128) or "document"
    if not ascii_name.rsplit(".", 1)[-1] if "." in ascii_name else False:
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1]
            ascii_name = f"{ascii_name}.{ext}" if not ascii_name.endswith(f".{ext}") else ascii_name
    encoded = _urlquote(filename, safe="")
    return {"Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"}

from ai_engine.japanese_nlp import sudachi_engine
from backend.database.config import get_db
from backend.schemas.document_schemas import (
    BusinessReportRequest,
    CareerHistoryRequest,
    CompanyAnalysisReportRequest,
    CoverLetterRequest,
    CustomDocumentRequest,
    DocumentHealthResponse,
    DocumentTemplateInfo,
    InvoiceRequest,
    MeetingMinutesRequest,
    NlpReportRequest,
    NlpReportResponse,
    ResumeRequest,
    TemplateFieldInfo,
    BatchDocumentRequest,
    OutlineDraftRequest,
)
from backend.services import data_service as data_svc
from backend.services import document_service as doc_svc
from document_engine import official_docs, report_generator
from document_engine import localize as doc_localize
from document_engine.docx_translate import translate_docx_bytes
from ai_engine.translation import translator as _translator

router = APIRouter(prefix="/documents", tags=["Document Intelligence"])


@router.get("/", status_code=200)
async def documents_root():
    return {
        "module": "Document Generation Intelligence",
        "status": "implemented",
        "engine": "python-docx (DOCX) + ReportLab (PDF with Japanese CID font)",
        "supported_formats": ["docx", "pdf"],
        "available_generators": [
            "nlp-report (bilingual NLP analysis report)",
            "japanese-resume (履歴書 Rirekisho)",
            "career-history (職務経歴書)",
            "invoice (請求書 Seikyu-Sho)",
            "business-report (一般報告書)",
            "cover-letter (送付状 Soufu-Jo)",
            "company-analysis-report (会社データ分析報告書)",
            "meeting-minutes (議事録)",
            "custom-document (自由書式文書)",
        ],
        "endpoints": {
            "health": "GET /api/documents/health",
            "templates": "GET /api/documents/templates",
            "template_details": "GET /api/documents/templates/{template_id}",
            "generate_nlp_report": "POST /api/documents/generate/nlp-report",
            "download_nlp_report": "POST /api/documents/generate/nlp-report/download",
            "generate_company_analysis_report": "POST /api/documents/generate/company-analysis-report",
            "download_company_analysis_report": "POST /api/documents/generate/company-analysis-report/download",
            "generate_company_report_from_dataset": "POST /api/documents/generate/company-analysis-report/from-dataset (multipart file + Form)",
            "generate_template": "POST /api/documents/generate/{template_id} (generic: japanese-resume, career-history, invoice, business-report, cover-letter, company-analysis-report)",
            "download_template": "POST /api/documents/generate/{template_id}/download (generic)",
            "list_documents": "GET /api/documents",
            "get_by_id": "GET /api/documents/{document_id}",
            "download_by_id": "GET /api/documents/{document_id}/download",
        },
    }


@router.get("/health", response_model=DocumentHealthResponse, tags=["Document Intelligence"])
async def documents_health():
    return DocumentHealthResponse(
        status="ok",
        engine="python-docx + ReportLab",
        supported_formats=["docx", "pdf"],
        available_generators=[
            "nlp-report",
            "japanese-resume",
            "career-history",
            "invoice",
            "business-report",
            "cover-letter",
            "company-analysis-report",
            "meeting-minutes",
            "custom-document",
        ],
    )


def _build_report(req: NlpReportRequest):
    try:
        analysis = sudachi_engine.analyze(req.text, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP analysis error: {e}")

    try:
        report = report_generator.generate_nlp_report(
            text=req.text,
            analysis_result=analysis.to_dict(),
            title=req.title,
            author=req.author or "KANAME [要]",
            include_bilingual=req.include_bilingual,
            doc_format=req.doc_format,
            output_dir=None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generator error: {e}")

    return report, analysis


@router.post("/generate/nlp-report", response_model=NlpReportResponse, tags=["Document Intelligence"])
async def generate_nlp_report_metadata(req: NlpReportRequest):
    report, _analysis = _build_report(req)
    return NlpReportResponse(
        filename=report.filename,
        format=report.format,
        size_bytes=report.size_bytes,
        message=(
            "Report generated. Use POST /api/documents/generate/nlp-report/download "
            "with identical body (JSON) to download."
        ),
    )


@router.post("/generate/nlp-report/download", tags=["Document Intelligence"])
async def download_nlp_report(req: NlpReportRequest, db: Session = Depends(get_db)):
    report, _analysis = _build_report(req)
    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if report.format == "docx"
        else "application/pdf"
    )
    headers = _content_disposition_headers(report.filename)
    bio = io.BytesIO(report.bytes)
    try:
        doc_svc.create_document(
            db=db,
            generator="nlp-report",
            filename=report.filename,
            format=report.format,
            title=req.title or "NLP分析レポート",
            author=req.author or "KANAME [要]",
            size_bytes=report.size_bytes,
            metadata={
                "mode": req.mode,
                "text_length": len(req.text or ""),
                "include_bilingual": req.include_bilingual,
            },
            storage_path=None,
        )
    except Exception:
        pass
    return StreamingResponse(bio, media_type=media_type, headers=headers)


def _resume_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="full_name_kanji", label_jp="氏名（漢字）", label_en="Full Name (Kanji)", field_type="string", required=True, placeholder="山田 太郎"),
        TemplateFieldInfo(name="full_name_kana", label_jp="氏名（ふりがな）", label_en="Full Name (Kana)", field_type="string", required=True, placeholder="やまだ たろう"),
        TemplateFieldInfo(name="issue_date", label_jp="作成日 / 記入日", label_en="Issue Date", field_type="date", required=False),
        TemplateFieldInfo(name="gender", label_jp="性別", label_en="Gender", field_type="select", required=False, default="other",
                          options=[{"value": "male", "label_jp": "男性", "label_en": "Male"}, {"value": "female", "label_jp": "女性", "label_en": "Female"}, {"value": "other", "label_jp": "その他", "label_en": "Other"}]),
        TemplateFieldInfo(name="date_of_birth", label_jp="生年月日", label_en="Date of Birth", field_type="date", required=False, placeholder="1990-01-01"),
        TemplateFieldInfo(name="phone", label_jp="電話番号", label_en="Phone", field_type="string", placeholder="090-xxxx-xxxx"),
        TemplateFieldInfo(name="email", label_jp="メールアドレス", label_en="Email", field_type="string", placeholder="taro@example.com"),
        TemplateFieldInfo(name="postal_code", label_jp="郵便番号", label_en="Postal Code", field_type="string", placeholder="100-0001"),
        TemplateFieldInfo(name="address_prefecture", label_jp="都道府県", label_en="Prefecture", field_type="string", placeholder="東京都"),
        TemplateFieldInfo(name="address_city", label_jp="市区町村", label_en="City / Ward", field_type="string", placeholder="千代田区"),
        TemplateFieldInfo(name="address_detail", label_jp="町名・番地・建物", label_en="Street / Building", field_type="string", placeholder="大手町1-1-1 大手町ビル301"),
        TemplateFieldInfo(name="photo_path", label_jp="顔写真 ファイルパス", label_en="Photo File Path (optional)", field_type="string", required=False,
                          description_jp="顔写真のファイル（PNG/JPG）のパス。空の場合は写真枠のみ表示します。",
                          description_en="File path to face photo (PNG/JPG). If empty, only photo frame is rendered."),
        TemplateFieldInfo(name="marital_status", label_jp="婚姻状況", label_en="Marital Status", field_type="select", default="single",
                          options=[{"value": "single", "label_jp": "未婚", "label_en": "Single"},
                                   {"value": "married", "label_jp": "既婚", "label_en": "Married"},
                                   {"value": "other", "label_jp": "その他", "label_en": "Other"}]),
        TemplateFieldInfo(name="family_members", label_jp="家族構成（扶養含む）", label_en="Family Members", field_type="object_list", required=False,
                          description_jp="配偶者・子などの家族構成。扶養義務の有無も記入してください。",
                          children=[
                              TemplateFieldInfo(name="relationship", label_jp="続柄", label_en="Relationship", field_type="string", required=True, placeholder="配偶者 / 子 / 父 / 母 など"),
                              TemplateFieldInfo(name="name", label_jp="氏名", label_en="Name", field_type="string", required=False),
                              TemplateFieldInfo(name="age", label_jp="年齢", label_en="Age", field_type="number", required=False),
                              TemplateFieldInfo(name="is_dependent", label_jp="扶養対象", label_en="Is Dependent", field_type="boolean", default=False),
                          ]),
        TemplateFieldInfo(name="education_list", label_jp="学歴", label_en="Education", field_type="object_list", required=False,
                          children=[
                              TemplateFieldInfo(name="start_date", label_jp="入学年月", label_en="Start Date", field_type="date"),
                              TemplateFieldInfo(name="end_date", label_jp="卒業年月", label_en="End Date", field_type="date"),
                              TemplateFieldInfo(name="school_name", label_jp="学校名", label_en="School Name", field_type="string", required=True),
                              TemplateFieldInfo(name="department", label_jp="学部・学科", label_en="Department", field_type="string"),
                              TemplateFieldInfo(name="degree", label_jp="学位", label_en="Degree", field_type="string"),
                              TemplateFieldInfo(name="is_graduated", label_jp="卒業済", label_en="Graduated", field_type="boolean", default=True),
                          ]),
        TemplateFieldInfo(name="work_history_list", label_jp="職歴", label_en="Work History", field_type="object_list",
                          children=[
                              TemplateFieldInfo(name="start_date", label_jp="入社年月", label_en="Start", field_type="date"),
                              TemplateFieldInfo(name="end_date", label_jp="退社年月", label_en="End", field_type="date"),
                              TemplateFieldInfo(name="company_name", label_jp="会社名", label_en="Company", field_type="string", required=True),
                              TemplateFieldInfo(name="employment_type", label_jp="雇用形態", label_en="Employment Type", field_type="select", default="full_time",
                                                options=[{"value": "full_time", "label_jp": "正社員", "label_en": "Full-time"},
                                                         {"value": "contract", "label_jp": "契約社員", "label_en": "Contract"},
                                                         {"value": "temp", "label_jp": "派遣社員", "label_en": "Temporary (Dispatch)"},
                                                         {"value": "outsourcing", "label_jp": "業務委託", "label_en": "Outsourcing"},
                                                         {"value": "part_time", "label_jp": "アルバイト・パート", "label_en": "Part-time"}]),
                              TemplateFieldInfo(name="position", label_jp="役職", label_en="Position", field_type="string"),
                              TemplateFieldInfo(name="description", label_jp="業務内容", label_en="Description", field_type="text"),
                              TemplateFieldInfo(name="resign_reason", label_jp="退職理由", label_en="Resignation Reason", field_type="string", required=False),
                              TemplateFieldInfo(name="is_current", label_jp="在職中", label_en="Currently Working", field_type="boolean", default=False),
                          ]),
        TemplateFieldInfo(name="licenses_list", label_jp="免許・資格", label_en="Licenses & Certifications", field_type="object_list",
                          children=[
                              TemplateFieldInfo(name="acquisition_date", label_jp="取得年月", label_en="Date", field_type="date"),
                              TemplateFieldInfo(name="license_name", label_jp="資格名", label_en="License Name", field_type="string", required=True),
                              TemplateFieldInfo(name="issuer", label_jp="発行元", label_en="Issuer", field_type="string"),
                          ]),
        TemplateFieldInfo(name="self_pr", label_jp="自己PR", label_en="Self PR", field_type="text", placeholder="あなたの強み・成果を具体的に"),
        TemplateFieldInfo(name="motivation", label_jp="志望動機", label_en="Motivation", field_type="text", placeholder="志望理由を記入"),
        TemplateFieldInfo(name="hobbies", label_jp="趣味・特技", label_en="Hobbies & Skills", field_type="text"),
        TemplateFieldInfo(name="requests", label_jp="本人希望記入欄", label_en="Requests / Preferred Conditions", field_type="text",
                          placeholder="給料・職種・勤務時間・勤務地 その他、希望する条件を記入"),
    ]


def _career_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="full_name_kanji", label_jp="氏名", label_en="Name", field_type="string", required=True),
        TemplateFieldInfo(name="full_name_kana", label_jp="フリガナ", label_en="Kana", field_type="string"),
        TemplateFieldInfo(name="phone", label_jp="電話番号", label_en="Phone", field_type="string"),
        TemplateFieldInfo(name="email", label_jp="メール", label_en="Email", field_type="string"),
        TemplateFieldInfo(name="career_summary", label_jp="職務要約", label_en="Career Summary", field_type="text",
                          description_jp="これまでの経歴を3～5行で要約してください。", description_en="Brief career summary (3-5 lines)"),
        TemplateFieldInfo(name="roles", label_jp="職務経歴", label_en="Roles", field_type="object_list",
                          children=[
                              TemplateFieldInfo(name="period_from", label_jp="開始年月", label_en="From", field_type="date", required=True),
                              TemplateFieldInfo(name="period_to", label_jp="終了年月", label_en="To", field_type="date"),
                              TemplateFieldInfo(name="company_name", label_jp="会社名", label_en="Company", field_type="string", required=True),
                              TemplateFieldInfo(name="position", label_jp="役職", label_en="Position", field_type="string", required=True),
                              TemplateFieldInfo(name="department", label_jp="部署", label_en="Dept.", field_type="string"),
                              TemplateFieldInfo(name="overview", label_jp="概要", label_en="Overview", field_type="text"),
                          ]),
        TemplateFieldInfo(name="skills", label_jp="保有スキル", label_en="Skills", field_type="list", placeholder="Python, FastAPI, ..."),
        TemplateFieldInfo(name="tech_stack", label_jp="技術スタック", label_en="Tech Stack", field_type="object_list"),
        TemplateFieldInfo(name="education", label_jp="学歴", label_en="Education", field_type="text"),
        TemplateFieldInfo(name="certifications", label_jp="資格", label_en="Certifications", field_type="list"),
    ]


def _invoice_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="invoice_number", label_jp="請求書番号", label_en="Invoice No.", field_type="string", required=True, placeholder="INV-2026-001"),
        TemplateFieldInfo(name="issue_date", label_jp="発行日", label_en="Issue Date", field_type="date"),
        TemplateFieldInfo(name="due_date", label_jp="支払期日", label_en="Due Date", field_type="date"),
        TemplateFieldInfo(name="seller_name", label_jp="発行元（請求者）", label_en="Seller", field_type="string", required=True, placeholder="株式会社○○"),
        TemplateFieldInfo(name="seller_department", label_jp="発行部署", label_en="Seller Dept.", field_type="string"),
        TemplateFieldInfo(name="seller_address", label_jp="発行元住所", label_en="Seller Address", field_type="string"),
        TemplateFieldInfo(name="seller_phone", label_jp="発行元電話", label_en="Seller Phone", field_type="string"),
        TemplateFieldInfo(name="seller_tax_id", label_jp="適格請求書登録番号", label_en="Tax ID (Invoice)", field_type="string", placeholder="T-xxxxxxxxxx"),
        TemplateFieldInfo(name="buyer_name", label_jp="請求先", label_en="Buyer", field_type="string", required=True, placeholder="株式会社△△"),
        TemplateFieldInfo(name="buyer_department", label_jp="請求先部署", label_en="Buyer Dept.", field_type="string"),
        TemplateFieldInfo(name="buyer_address", label_jp="請求先住所", label_en="Buyer Address", field_type="string"),
        TemplateFieldInfo(name="title", label_jp="表題", label_en="Title", field_type="string", default="請求書"),
        TemplateFieldInfo(name="items", label_jp="明細", label_en="Line Items", field_type="object_list", required=True,
                          children=[
                              TemplateFieldInfo(name="description", label_jp="品名", label_en="Description", field_type="string", required=True),
                              TemplateFieldInfo(name="quantity", label_jp="数量", label_en="Qty", field_type="number", default=1.0),
                              TemplateFieldInfo(name="unit_price", label_jp="単価", label_en="Unit Price", field_type="number", required=True),
                              TemplateFieldInfo(name="tax_rate", label_jp="税率", label_en="Tax Rate", field_type="number", default=0.10),
                          ]),
        TemplateFieldInfo(name="notes", label_jp="特記事項", label_en="Notes", field_type="text"),
        TemplateFieldInfo(name="payment_method", label_jp="支払方法", label_en="Payment Method", field_type="select", default="bank_transfer",
                          options=[{"value": "bank_transfer", "label_jp": "銀行振込", "label_en": "Bank Transfer"},
                                   {"value": "credit_card", "label_jp": "クレジットカード", "label_en": "Credit Card"},
                                   {"value": "cash", "label_jp": "現金", "label_en": "Cash"},
                                   {"value": "konbini", "label_jp": "コンビニ", "label_en": "Konbini"}]),
        TemplateFieldInfo(name="bank_name", label_jp="銀行名", label_en="Bank Name", field_type="string"),
        TemplateFieldInfo(name="bank_branch", label_jp="支店名", label_en="Branch", field_type="string"),
        TemplateFieldInfo(name="bank_account_type", label_jp="口座種別", label_en="Account Type", field_type="select",
                          options=[{"value": "ordinary", "label_jp": "普通", "label_en": "Ordinary"}, {"value": "current", "label_jp": "当座", "label_en": "Current"}]),
        TemplateFieldInfo(name="bank_account_number", label_jp="口座番号", label_en="Account No.", field_type="string"),
        TemplateFieldInfo(name="bank_account_holder", label_jp="口座名義", label_en="Account Holder", field_type="string"),
    ]


def _report_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="title", label_jp="報告書名", label_en="Title", field_type="string", required=True, placeholder="〇〇調査報告書"),
        TemplateFieldInfo(name="subtitle", label_jp="副題", label_en="Subtitle", field_type="string"),
        TemplateFieldInfo(name="author", label_jp="作成者", label_en="Author", field_type="string", required=True),
        TemplateFieldInfo(name="author_department", label_jp="作成部署", label_en="Author Dept.", field_type="string"),
        TemplateFieldInfo(name="report_date", label_jp="作成日", label_en="Report Date", field_type="date"),
        TemplateFieldInfo(name="recipient", label_jp="提出先氏名", label_en="Recipient", field_type="string"),
        TemplateFieldInfo(name="recipient_department", label_jp="提出先部署", label_en="Recipient Dept.", field_type="string"),
        TemplateFieldInfo(name="executive_summary", label_jp="概要", label_en="Executive Summary", field_type="text"),
        TemplateFieldInfo(name="background", label_jp="背景", label_en="Background", field_type="text"),
        TemplateFieldInfo(name="objectives", label_jp="目的（リスト）", label_en="Objectives", field_type="list"),
        TemplateFieldInfo(name="methods", label_jp="調査・実施方法", label_en="Methods", field_type="text"),
        TemplateFieldInfo(name="results", label_jp="結果", label_en="Results", field_type="text"),
        TemplateFieldInfo(name="discussion", label_jp="考察", label_en="Discussion", field_type="text"),
        TemplateFieldInfo(name="issues", label_jp="課題（リスト）", label_en="Issues", field_type="list"),
        TemplateFieldInfo(name="conclusion", label_jp="結論", label_en="Conclusion", field_type="text"),
        TemplateFieldInfo(name="recommendations", label_jp="提言・今後の方針", label_en="Recommendations", field_type="list"),
        TemplateFieldInfo(name="references", label_jp="参考文献", label_en="References", field_type="list"),
        TemplateFieldInfo(name="attachments", label_jp="添付資料", label_en="Attachments", field_type="list"),
    ]


def _cover_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="sender_company", label_jp="送付元会社", label_en="Sender Company", field_type="string"),
        TemplateFieldInfo(name="sender_department", label_jp="送付元部署", label_en="Sender Dept.", field_type="string"),
        TemplateFieldInfo(name="sender_name", label_jp="送付元氏名", label_en="Sender Name", field_type="string", required=True),
        TemplateFieldInfo(name="sender_postal_code", label_jp="送付元郵便番号", label_en="Sender Postal", field_type="string"),
        TemplateFieldInfo(name="sender_address", label_jp="送付元住所", label_en="Sender Address", field_type="string"),
        TemplateFieldInfo(name="sender_phone", label_jp="送付元電話", label_en="Sender Phone", field_type="string"),
        TemplateFieldInfo(name="sender_email", label_jp="送付元メール", label_en="Sender Email", field_type="string"),
        TemplateFieldInfo(name="recipient_company", label_jp="送付先会社", label_en="Recipient Company", field_type="string", required=True),
        TemplateFieldInfo(name="recipient_department", label_jp="送付先部署", label_en="Recipient Dept.", field_type="string"),
        TemplateFieldInfo(name="recipient_title", label_jp="送付先役職", label_en="Recipient Title", field_type="string"),
        TemplateFieldInfo(name="recipient_name", label_jp="送付先氏名（御中可）", label_en="Recipient Name", field_type="string", default="御中"),
        TemplateFieldInfo(name="letter_date", label_jp="日付", label_en="Letter Date", field_type="date"),
        TemplateFieldInfo(name="subject", label_jp="件名", label_en="Subject", field_type="string", required=True, placeholder="〇〇書類送付のご案内"),
        TemplateFieldInfo(name="greeting", label_jp="頭語・挨拶", label_en="Greeting", field_type="string", default="お世話になっております。"),
        TemplateFieldInfo(name="body_paragraphs", label_jp="本文（パラグラフ毎）", label_en="Body Paragraphs", field_type="list", required=True),
        TemplateFieldInfo(name="closing", label_jp="結語・末文", label_en="Closing", field_type="string", default="ご確認の程、何卒よろしくお願い申し上げます。"),
        TemplateFieldInfo(name="enclosures_title", label_jp="送付物見出し", label_en="Enclosures Title", field_type="string", default="添付書類一覧"),
        TemplateFieldInfo(name="enclosures", label_jp="送付物一覧", label_en="Enclosures", field_type="list", required=True,
                          placeholder="履歴書・職務経歴書×1,  参考資料×2"),
    ]


def _company_analysis_report_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="report_title", label_jp="報告書タイトル", label_en="Report Title", field_type="string", required=True,
                          default="会社データ分析報告書"),
        TemplateFieldInfo(name="report_date", label_jp="報告日", label_en="Report Date", field_type="date"),
        TemplateFieldInfo(name="period_covered", label_jp="集計期間", label_en="Period Covered", field_type="string", required=False,
                          placeholder="2026年1月 ～ 2026年3月"),
        TemplateFieldInfo(name="company_name", label_jp="会社名", label_en="Company Name", field_type="string", required=False, placeholder="株式会社○○"),
        TemplateFieldInfo(name="department", label_jp="作成部署", label_en="Department", field_type="string"),
        TemplateFieldInfo(name="author_name", label_jp="作成者", label_en="Author", field_type="string",
                          description_jp="報告書の作成者。author と同じ。",
                          description_en="Report author. Aliased from author."),
        TemplateFieldInfo(name="analyst_name", label_jp="分析担当者", label_en="Analyst", field_type="string"),
        TemplateFieldInfo(name="recipient_name", label_jp="提出先", label_en="Recipient", field_type="string"),
        TemplateFieldInfo(name="doc_format", label_jp="出力形式", label_en="Output Format", field_type="select", default="docx",
                          options=[{"value": "docx", "label_jp": "DOCX (Word)", "label_en": "DOCX (Word)"},
                                   {"value": "pdf", "label_jp": "PDF", "label_en": "PDF"}]),
        TemplateFieldInfo(name="iqr_k", label_jp="IQR k-factor (外れ値感度)", label_en="IQR k-factor (outlier sensitivity)",
                          field_type="number", default=1.5,
                          description_jp="小さいほど外れ値検出が厳しくなります。標準は 1.5。",
                          description_en="Lower = stricter outlier detection. Standard default is 1.5."),
        TemplateFieldInfo(name="corr_method", label_jp="相関係数の計算手法", label_en="Correlation method",
                          field_type="select", default="pearson",
                          options=[{"value": "pearson", "label_jp": "ピアソン (Pearson)", "label_en": "Pearson r (parametric)"},
                                   {"value": "spearman", "label_jp": "スピアマン (Spearman) 順位相関", "label_en": "Spearman ρ (rank-based)"},
                                   {"value": "kendall", "label_jp": "ケンドール (Kendall) τ", "label_en": "Kendall τ (ordinal)"}]),
        TemplateFieldInfo(name="include_appendix", label_jp="付録を含める", label_en="Include Appendix",
                          field_type="boolean", default=True),
        TemplateFieldInfo(name="dataset_id", label_jp="データセットID (DB連携時)", label_en="Dataset ID (if DB connected)",
                          field_type="string", required=False,
                          description_jp="DB接続時のみ使用。未指定時は from-dataset エンドポイントでアップロードしたファイルを使用してください。",
                          description_en="Used only when DB is connected. Otherwise use the from-dataset multipart endpoint."),
        TemplateFieldInfo(name="source_filename", label_jp="データソース ファイル名", label_en="Source Filename",
                          field_type="string", required=False, placeholder="sales_q1q2_2026.csv"),
        TemplateFieldInfo(name="override_discussion", label_jp="【上書き】考察・示唆", label_en="Override: Discussion",
                          field_type="text", required=False),
        TemplateFieldInfo(name="override_action_plan", label_jp="【上書き】アクションプラン", label_en="Override: Action Plan (list)",
                          field_type="list", required=False),
        TemplateFieldInfo(name="override_report_title", label_jp="【上書き】表紙タイトル", label_en="Override: Cover Title",
                          field_type="string", required=False),
    ]


def _meeting_minutes_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="title", label_jp="表題", label_en="Title", field_type="string",
                          default="議事録", placeholder="議事録"),
        TemplateFieldInfo(name="meeting_date", label_jp="開催日", label_en="Meeting Date", field_type="date"),
        TemplateFieldInfo(name="start_time", label_jp="開始時刻", label_en="Start Time", field_type="string", placeholder="14:00"),
        TemplateFieldInfo(name="end_time", label_jp="終了時刻", label_en="End Time", field_type="string", placeholder="15:30"),
        TemplateFieldInfo(name="location", label_jp="開催場所", label_en="Location", field_type="string", placeholder="本社 3階 会議室A"),
        TemplateFieldInfo(name="meeting_type", label_jp="会議種別", label_en="Meeting Type", field_type="select", default="regular",
                          options=[{"value": "regular", "label_jp": "定例会議", "label_en": "Regular"},
                                   {"value": "extraordinary", "label_jp": "臨時会議", "label_en": "Extraordinary"},
                                   {"value": "emergency", "label_jp": "緊急会議", "label_en": "Emergency"}]),
        TemplateFieldInfo(name="chairperson", label_jp="議長", label_en="Chairperson", field_type="string"),
        TemplateFieldInfo(name="recorder", label_jp="書記", label_en="Recorder", field_type="string"),
        TemplateFieldInfo(name="attendees", label_jp="出席者", label_en="Attendees", field_type="list"),
        TemplateFieldInfo(name="absentees", label_jp="欠席者", label_en="Absentees", field_type="list"),
        TemplateFieldInfo(name="agenda", label_jp="議題", label_en="Agenda", field_type="list"),
        TemplateFieldInfo(name="discussion", label_jp="議事内容", label_en="Discussion", field_type="text"),
        TemplateFieldInfo(name="decisions", label_jp="決定事項", label_en="Decisions", field_type="list"),
        TemplateFieldInfo(name="action_items", label_jp="アクションアイテム", label_en="Action Items", field_type="object_list",
                          children=[
                              TemplateFieldInfo(name="task", label_jp="タスク", label_en="Task", field_type="string", required=True),
                              TemplateFieldInfo(name="owner", label_jp="担当", label_en="Owner", field_type="string"),
                              TemplateFieldInfo(name="due_date", label_jp="期限", label_en="Due Date", field_type="date"),
                          ]),
        TemplateFieldInfo(name="next_meeting_date", label_jp="次回開催日", label_en="Next Meeting Date", field_type="date"),
    ]


def _custom_document_fields() -> list[TemplateFieldInfo]:
    return [
        TemplateFieldInfo(name="title", label_jp="表題", label_en="Title", field_type="string", required=True,
                          placeholder="〇〇ご提案書"),
        TemplateFieldInfo(name="author", label_jp="作成者", label_en="Author", field_type="string"),
        TemplateFieldInfo(name="document_date", label_jp="文書日付", label_en="Document Date", field_type="date"),
        TemplateFieldInfo(name="recipient", label_jp="宛先", label_en="Recipient", field_type="string", placeholder="株式会社〇〇 御中"),
        TemplateFieldInfo(name="intro", label_jp="前書き・導入文", label_en="Intro Paragraph", field_type="text",
                          description_jp="本文の前に置く導入文（任意）。",
                          description_en="Optional introductory paragraph shown before the sections."),
        TemplateFieldInfo(name="fields", label_jp="カスタム項目（任意）", label_en="Custom Fields (label / value)", field_type="object_list",
                          description_jp="ユーザーが自由に定義できる項目名と値のペア。基本情報テーブルとして出力されます。",
                          description_en="User-defined label/value pairs rendered as a key-value table.",
                          children=[
                              TemplateFieldInfo(name="label", label_jp="項目名", label_en="Label", field_type="string", required=True),
                              TemplateFieldInfo(name="value", label_jp="値", label_en="Value", field_type="string"),
                          ]),
        TemplateFieldInfo(name="sections", label_jp="本文セクション", label_en="Sections (heading / body)", field_type="object_list",
                          description_jp="任意の見出しと本文の組み合わせを任意数定義できます。",
                          description_en="User-defined sections, each with a heading and a body.",
                          children=[
                              TemplateFieldInfo(name="heading", label_jp="見出し", label_en="Heading", field_type="string", required=True),
                              TemplateFieldInfo(name="body", label_jp="本文", label_en="Body", field_type="text"),
                          ]),
        TemplateFieldInfo(name="closing", label_jp="結語・締めの文", label_en="Closing", field_type="text"),
    ]


TEMPLATE_CATALOG: dict[str, DocumentTemplateInfo] = {}


def _build_catalog() -> dict[str, DocumentTemplateInfo]:
    if TEMPLATE_CATALOG:
        return TEMPLATE_CATALOG

    templates = [
        DocumentTemplateInfo(
            template_id="japanese-resume",
            name_jp="履歴書",
            name_en="Japanese Standard Resume (Rirekisho)",
            category="employment",
            description_jp="日本の標準的な履歴書フォーマット（A4縦型）。学歴・職歴・免許・自己PR・志望動機の各セクションと、写真枠を備えます。",
            description_en="Standard Japanese Rirekisho format — A4 portrait layout with lined grids for education, work history, licenses/certifications, self PR, and motivation; includes reserved photo box.",
            typical_use_case_jp="新卒・中途採用への応募、公的機関への提出、アルバイト応募",
            typical_use_case_en="Job applications (entry-level & mid-career), government submissions, part-time hiring.",
            formats=["docx", "pdf"],
            fields=_resume_fields(),
        ),
        DocumentTemplateInfo(
            template_id="career-history",
            name_jp="職務経歴書",
            name_en="Career History / Shokumu-Kekisho",
            category="employment",
            description_jp="職務経歴書フォーマット。キャリアサマリー・職務内容・プロジェクト実績・技術スタック・資格の各セクションを備えます。",
            description_en="Japanese career resume — sections for career summary, roles, project achievements, skill matrix, tech stack, and certifications.",
            typical_use_case_jp="中途採用応募、転職活動、外資/IT企業応募の職務詳細",
            typical_use_case_en="Mid-career job transitions, detailed role history for tech/foreign-capital companies.",
            formats=["docx", "pdf"],
            fields=_career_fields(),
        ),
        DocumentTemplateInfo(
            template_id="invoice",
            name_jp="請求書",
            name_en="Japanese Invoice (Seikyu-Sho)",
            category="finance",
            description_jp="日本の請求書フォーマット（適格請求書発行事業者対応）。明細行・小計・10%消費税・合計・支払方法・銀行口座欄を備えます。",
            description_en="Standard Japanese Teikaku-Seikyu-Sho: line items, subtotal, 10% consumption tax, total, payment method, and bank-transfer fields; compliant with Invoice System (T-number optional).",
            typical_use_case_jp="B2B/B2C請求業務、フリーランス請求、月次締め請求書",
            typical_use_case_en="B2B invoicing, freelance billing, monthly-closed billing.",
            formats=["docx", "pdf"],
            fields=_invoice_fields(),
        ),
        DocumentTemplateInfo(
            template_id="business-report",
            name_jp="一般報告書",
            name_en="General Business Report (Ippan-Hokokusho)",
            category="report",
            description_jp="日本企業向け一般報告書フォーマット。表紙・概要・背景・目的・方法・結果・考察・課題・結論・提言・参考文献の章構成を持ちます。",
            description_en="Standard corporate business report: cover, executive summary, background, objectives, methods, results, discussion, issues, conclusion, recommendations, references, and attachments.",
            typical_use_case_jp="調査報告・プロジェクト中間報告・改善報告・決算報告",
            typical_use_case_en="Survey reports, project milestone reports, improvement reports, financial summaries.",
            formats=["docx", "pdf"],
            fields=_report_fields(),
        ),
        DocumentTemplateInfo(
            template_id="cover-letter",
            name_jp="送付状",
            name_en="Cover Letter / Transmittal (Soufu-Jo)",
            category="correspondence",
            description_jp="日本標準送付状（添付状）フォーマット。送付元/送付先情報、件名、挨拶、本文、送付物一覧表を備えます。",
            description_en="Japanese business transmittal sheet (cover letter): sender/recipient blocks, subject, greeting, body paragraphs, numbered enclosures list.",
            typical_use_case_jp="書類送付時の挨拶状、契約書/見積書同梱、採用応募書類への添付",
            typical_use_case_en="Accompany hard-copy submissions (resumes, contracts, quotes, proposals).",
            formats=["docx", "pdf"],
            fields=_cover_fields(),
        ),
        DocumentTemplateInfo(
            template_id="company-analysis-report",
            name_jp="会社データ分析報告書",
            name_en="Company Dataset Analysis Report (Kaisha Bunseki)",
            category="report",
            description_jp="会社データ（CSV/Excel）を KANAME データインテリジェンスモジュールで直接分析し、12章構成（表紙/目次/KPI/記述統計/カテゴリ分布/相関heatmap/IQR外れ値/欠損値/考察/アクションプラン/付録）の公式報告書をDOCX/PDFで出力。生アップロードでも動作し、最近の分析と同じ計算ロジックを使用します。",
            description_en="Analyzes a company dataset (CSV/Excel) directly via the KANAME data-intelligence module and renders a 12-section standard analysis report (cover, TOC, KPI strip, descriptive stats, categorical Top-10, correlation heatmap, IQR outliers, missing values, auto-discussion, action plan, appendices) in DOCX or PDF. Uses identical math as the recent-analysis dashboard endpoint even without DB.",
            typical_use_case_jp="月次売上分析、部門別KPIレビュー、品質データ監査、投資判断のための定量レポート、経営会議提出資料",
            typical_use_case_en="Monthly sales reviews, KPI dashboards-as-docs, quality audits, investment decision support, formal board-level quantitative reports.",
            formats=["docx", "pdf"],
            fields=_company_analysis_report_fields(),
        ),
        DocumentTemplateInfo(
            template_id="meeting-minutes",
            name_jp="議事録",
            name_en="Meeting Minutes (Giji-Roku)",
            category="business",
            description_jp="日本企業向け議事録フォーマット。開催日時・場所・会議種別・議長・書記・出席者/欠席者・議題・議事内容・決定事項・アクションアイテム（担当/期限）・次回開催日を備えます。",
            description_en="Standard Japanese meeting minutes: date/time, location, meeting type, chairperson, recorder, attendees/absentees, agenda, discussion, decisions, action items (owner/due date), and next meeting date.",
            typical_use_case_jp="定例会議・臨時会議・緊急会議の議事録作成、プロジェクト会議の記録、経営会議の議事録",
            typical_use_case_en="Regular/extraordinary/emergency meeting minutes, project meeting records, board meeting notes.",
            formats=["docx", "pdf"],
            fields=_meeting_minutes_fields(),
        ),
        DocumentTemplateInfo(
            template_id="custom-document",
            name_jp="自由書式文書",
            name_en="Custom Document (Free-Format)",
            category="report",
            description_jp="ユーザーが任意のセクション（見出し＋本文）とカスタム項目（ラベル＋値）を自由に定義できる汎用文書フォーマット。表題・宛先・作成者・日付・前書き・結語を備え、あらゆる社内文書に対応します。",
            description_en="Fully user-defined document: any number of headed sections (heading + body) plus custom key-value fields. Includes title, recipient, author, date, intro, and closing — suitable for any ad-hoc business document.",
            typical_use_case_jp="提案書・企画書・社内通知・申込書・社内規定など、定型テンプレートに当てはまらない任意書式の文書作成",
            typical_use_case_en="Proposals, planning docs, internal notices, application forms, policies — any ad-hoc document not covered by a fixed template.",
            formats=["docx", "pdf"],
            fields=_custom_document_fields(),
        ),
    ]
    for t in templates:
        TEMPLATE_CATALOG[t.template_id] = t
    return TEMPLATE_CATALOG


TEMPLATE_REQUEST_SCHEMAS: dict[str, type[Any]] = {
    "japanese-resume": ResumeRequest,
    "career-history": CareerHistoryRequest,
    "invoice": InvoiceRequest,
    "business-report": BusinessReportRequest,
    "cover-letter": CoverLetterRequest,
    "company-analysis-report": CompanyAnalysisReportRequest,
    "meeting-minutes": MeetingMinutesRequest,
    "custom-document": CustomDocumentRequest,
}


@router.get("/templates", tags=["Document Intelligence"])
async def list_document_templates(
    category: str | None = Query(None, description="Filter by category (employment/finance/business/correspondence/report)"),
):
    catalog = _build_catalog()
    result = list(catalog.values())
    if category:
        result = [t for t in result if t.category == category]
    return {
        "templates": [t.model_dump() for t in result],
        "count": len(result),
        "category_filter": category,
    }


@router.get("/templates/{template_id}", tags=["Document Intelligence"])
async def get_template_details(template_id: str):
    catalog = _build_catalog()
    if template_id not in catalog:
        return {"template": None, "error": "template_id not found"}
    return {"template": catalog[template_id].model_dump()}


@router.post("/outline/draft", tags=["Document Intelligence"])
async def draft_document_outline(req: OutlineDraftRequest):
    """Draft an outline with the configured LLM provider; never labels a fallback as AI."""
    engine, key = _translator._detect_engine()
    if not engine or not key:
        raise HTTPException(
            status_code=503,
            detail="AI outline drafting requires a configured LLM API key.",
        )
    language = "Japanese" if req.language == "ja" else "English"
    prompt = (
        f"Draft a practical document outline in {language}. "
        f"Topic: {req.topic}\nAudience: {req.audience or 'general'}\n"
        f"Use exactly {req.section_count} numbered sections. Give each section a concise heading "
        "and one short purpose sentence. Do not invent facts or write the full document."
    )
    try:
        outline = _translator._call_llm(
            engine,
            key,
            "You are a careful professional document editor. Return only the requested outline.",
            prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Outline provider failed: {exc}")
    if not outline.strip():
        raise HTTPException(status_code=502, detail="Outline provider returned an empty response.")
    return {"outline": outline, "provider": engine, "topic": req.topic, "language": req.language}


@router.post("/batch/generate/download", tags=["Document Intelligence"])
async def generate_document_batch(req: BatchDocumentRequest):
    """Generate up to 20 validated documents and return them with a manifest in a ZIP."""
    catalog = _build_catalog()
    generated: list[tuple[str, bytes, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []
    for index, item in enumerate(req.items, start=1):
        template_id = item.template_id
        if template_id not in TEMPLATE_REQUEST_SCHEMAS or template_id not in catalog:
            errors.append({"item": str(index), "error": f"Unknown template_id: {template_id}"})
            continue
        schema_cls = TEMPLATE_REQUEST_SCHEMAS[template_id]
        try:
            data = schema_cls.model_validate(item.payload).model_dump()
            fmt = str(data.get("doc_format") or "docx").lower()
            if fmt not in ("docx", "pdf"):
                raise ValueError(f"Unsupported format: {fmt}")
            if req.auto_translate:
                data, _ = doc_localize.localize_document_data(schema_cls, data, target_lang="ja")
            report = official_docs.generate_official_document(template_id, fmt, data)
            safe_name = _sanitize_filename(report.filename)
            archive_name = f"{index:02d}_{safe_name}"
            generated.append((archive_name, report.bytes, {
                "template_id": template_id,
                "filename": archive_name,
                "format": fmt,
                "size_bytes": report.size_bytes,
            }))
        except Exception as exc:
            errors.append({"item": str(index), "error": str(exc)})
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})

    archive = io.BytesIO()
    manifest = {"count": len(generated), "documents": [entry[2] for entry in generated]}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload, _metadata in generated:
            bundle.writestr(name, payload)
        bundle.writestr("manifest.json", __import__("json").dumps(manifest, ensure_ascii=False, indent=2))
    archive.seek(0)
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="kaname_documents.zip"'},
    )


@router.post("/merge-pdfs", tags=["Document Intelligence"])
async def merge_bilingual_pdfs(files: list[UploadFile] = File(..., description="PDF files, merged in upload order")):
    if not 2 <= len(files) <= 20:
        raise HTTPException(status_code=422, detail="Upload between 2 and 20 PDF files.")
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        raise HTTPException(status_code=503, detail="PDF merge dependency is unavailable.")

    writer = PdfWriter()
    total_bytes = 0
    try:
        for file in files:
            if not (file.filename or "").lower().endswith(".pdf"):
                raise HTTPException(status_code=415, detail=f"{file.filename} is not a PDF.")
            content = await file.read()
            total_bytes += len(content)
            if total_bytes > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Combined PDF upload exceeds 50 MB.")
            try:
                reader = PdfReader(io.BytesIO(content), strict=False)
                if reader.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported.")
                for page in reader.pages:
                    writer.add_page(page)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Could not read {file.filename}: {exc}")
        output = io.BytesIO()
        writer.write(output)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="kaname_bilingual_merged.pdf"'},
        )
    finally:
        for file in files:
            await file.close()


@router.post("/translate-docx", tags=["Document Intelligence"])
async def translate_docx_document(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form(...),
    domain: str | None = Form(None),
    force_fallback: bool = Form(False),
):
    if target_lang not in ("ja", "en") or source_lang not in ("auto", "ja", "en"):
        raise HTTPException(status_code=422, detail="Languages must be auto/ja/en and target must be ja/en.")
    if not (file.filename or "").lower().endswith(".docx"):
        raise HTTPException(status_code=415, detail="Only .docx documents are supported.")
    payload = await file.read()
    if len(payload) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="DOCX upload exceeds 20 MB.")
    try:
        translated = translate_docx_bytes(
            payload,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            force_fallback=force_fallback,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"DOCX translation failed: {exc}")
    stem = os.path.splitext(_sanitize_filename(file.filename or "document"))[0]
    return StreamingResponse(
        io.BytesIO(translated),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{stem}_{target_lang}.docx"'},
    )


def _sanitize_filename(fn: str) -> str:
    safe = "".join(c for c in fn if c.isalnum() or c in "-._()[] 　") or "document"
    return safe

def _media_type_for(fmt: str):
    return (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if (fmt or "docx").lower() == "docx"
        else "application/pdf"
    )


@router.post("/generate/company-analysis-report/from-dataset", tags=["Document Intelligence"])
async def generate_company_analysis_from_dataset(
    file: UploadFile | None = File(None, description="CSV / Excel dataset file (.csv, .xlsx, .xls)"),
    dataset_id: str | None = Form(None, description="Optional: saved dataset_id to use instead of upload"),
    report_title: str = Form("会社データ分析報告書"),
    report_date: str | None = Form(None),
    period_covered: str | None = Form(None),
    company_name: str = Form(...),
    department: str | None = Form(None),
    author: str = Form("KANAME [要]"),
    author_name: str | None = Form(None),
    analyst_name: str | None = Form(None),
    recipient: str | None = Form(None),
    recipient_name: str | None = Form(None),
    source_filename: str | None = Form(None),
    iqr_k: float = Form(1.5),
    corr_method: str = Form("pearson"),
    executive_summary_override: str | None = Form(None),
    conclusion_override: str | None = Form(None),
    override_discussion: str | None = Form(None),
    override_action_plan: str | None = Form(None, description="Comma-separated list items"),
    override_report_title: str | None = Form(None),
    include_appendix: bool = Form(True),
    doc_format: str = Form("docx"),
    db: Session = Depends(get_db),
):
    fmt = (doc_format or "docx").lower()
    if fmt not in ("docx", "pdf"):
        return JSONResponse(status_code=200, content={"error": f"Unsupported doc_format '{fmt}'. Use docx or pdf."})
    if file is None and not dataset_id:
        return JSONResponse(status_code=200, content={"error": "Either file upload or dataset_id is required."})
    if corr_method not in ("pearson", "spearman", "kendall"):
        return JSONResponse(status_code=200, content={"error": "corr_method must be one of: pearson, spearman, kendall"})
    meta: dict[str, Any] = {
        "report_title": report_title,
        "report_date": report_date,
        "period_covered": period_covered,
        "company_name": company_name,
        "department": department,
        "author": author_name or author,
        "analyst_name": analyst_name,
        "recipient": recipient_name or recipient,
        "source_filename": source_filename,
        "iqr_k": float(iqr_k),
        "corr_method": corr_method,
        "executive_summary_override": executive_summary_override,
        "conclusion_override": conclusion_override,
        "override_discussion": override_discussion,
        "override_action_plan": (
            [s.strip() for s in override_action_plan.split(",") if s.strip()]
            if override_action_plan else []
        ),
        "override_report_title": override_report_title,
        "include_appendix": include_appendix,
        "doc_format": fmt,
    }
    tmp_path: str | None = None
    try:
        if file is not None:
            suffix = ""
            fname = (file.filename or "dataset").lower()
            if fname.endswith(".csv"):
                suffix = ".csv"
            elif fname.endswith(".xlsx"):
                suffix = ".xlsx"
            elif fname.endswith(".xls"):
                suffix = ".xls"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="kaname_car_")
            os.close(fd)
            with open(tmp_path, "wb") as outf:
                content = await file.read()
                outf.write(content)
            if not meta["source_filename"]:
                meta["source_filename"] = file.filename or "dataset"
            data_input: Any = tmp_path
        else:
            try:
                ds = data_svc.get_dataset(db, dataset_id)
            except Exception as e:
                return JSONResponse(status_code=200, content={"error": f"Database unavailable for dataset lookup: {e}"})
            if ds is None:
                return JSONResponse(status_code=200, content={"error": f"dataset_id {dataset_id} not found."})
            content = data_svc.get_dataset_content(db, dataset_id)
            if content is None:
                return JSONResponse(
                    status_code=200,
                    content={"error": "Dataset source is unavailable; re-upload the dataset."},
                )
            filename = ds.get("filename", "dataset.csv")
            suffix = os.path.splitext(filename)[1].lower()
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="kaname_car_")
            os.close(fd)
            with open(tmp_path, "wb") as stored_file:
                stored_file.write(content)
            data_input = tmp_path
            if not meta["source_filename"]:
                meta["source_filename"] = filename
        try:
            payload = official_docs.build_company_analysis_report_from_dataset(
                data_input,
                report_meta=meta,
                iqr_k=float(iqr_k),
                corr_method=corr_method,
                include_raw_appendix=include_appendix,
            )
        except ValueError as e:
            return JSONResponse(status_code=200, content={"error": f"Dataset analysis / build error: {e}"})
        except RuntimeError as e:
            return JSONResponse(status_code=200, content={"error": f"Analyzer unavailable: {e}"})
        except Exception as e:
            return JSONResponse(status_code=200, content={"error": f"Unexpected error building report payload: {e}"})
        final_payload: dict[str, Any] = dict(payload)
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            if k in ("override_discussion", "override_action_plan", "override_report_title",
                     "executive_summary_override", "conclusion_override") or k not in final_payload or not final_payload[k]:
                final_payload[k] = v
        final_payload["doc_format"] = fmt
        try:
            validated = CompanyAnalysisReportRequest.model_validate(final_payload)
        except ValidationError as e:
            errors = [err.get("msg", str(err)) for err in e.errors()]
            return JSONResponse(status_code=200, content={
                "error": f"Built payload failed schema validation: {'; '.join(errors)}",
            })
        final_data = validated.model_dump()
        try:
            report = official_docs.generate_official_document("company-analysis-report", fmt, final_data)
        except ValueError as e:
            return JSONResponse(status_code=200, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=200, content={"error": f"Document engine error: {e}"})
        try:
            metadata = {k: (v if (not isinstance(v, str) or len(v) <= 2000) else v[:2000] + "…[truncated]")
                        for k, v in final_data.items()
                        if not (isinstance(v, list) and len(v) > 50)}
            for k in list(metadata.keys()):
                v = metadata[k]
                if isinstance(v, list) and len(v) > 50:
                    metadata[k] = f"[truncated list len={len(v)}]"
            doc_svc.create_document(
                db=db,
                generator="company-analysis-report",
                filename=report.filename,
                format=report.format,
                title=str(override_report_title or report_title or "会社データ分析報告書"),
                author=str(author_name or author or "KANAME [要]")[:256],
                size_bytes=int(report.size_bytes),
                metadata=metadata,
                storage_path=None,
            )
        except Exception:
            pass
        media = _media_type_for(fmt)
        filename = _sanitize_filename(report.filename)
        ascii_name = "".join(c for c in filename if ord(c) < 128) or "document"
        if "." in filename and not ascii_name.endswith("." + filename.rsplit(".", 1)[-1]):
            ext = filename.rsplit(".", 1)[-1]
            if "".join(c for c in ext if ord(c) < 128) == ext:
                ascii_name = f"{ascii_name}.{ext}"
        encoded = quote(filename)
        headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}; filename=\"{ascii_name}\""}
        return StreamingResponse(io.BytesIO(report.bytes), media_type=media, headers=headers)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@router.post("/generate/company-analysis-report", tags=["Document Intelligence"])
async def generate_company_analysis_report_metadata(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
    auto_translate: bool = Query(
        True,
        description="If true, English free-text fields are translated to Japanese before generation "
                    "(KANAME supports JA<->EN only).",
    ),
):
    try:
        validated = CompanyAnalysisReportRequest.model_validate(payload)
    except ValidationError as e:
        errors = [err.get("msg", str(err)) for err in e.errors()]
        return {
            "document": None, "document_id": None, "generator": "company-analysis-report",
            "error": f"Schema validation failed: {'; '.join(errors)}",
        }
    data = validated.model_dump()
    fmt = str(data.get("doc_format") or "docx").lower()
    if fmt not in ("docx", "pdf"):
        return {"document": None, "document_id": None, "generator": "company-analysis-report",
                "error": f"Unsupported format {fmt}"}

    translation_report = None
    if auto_translate:
        data, translation_report = doc_localize.localize_document_data(
            CompanyAnalysisReportRequest, data, target_lang="ja"
        )

    try:
        report = official_docs.generate_official_document("company-analysis-report", fmt, data)
    except ValueError as e:
        return {"document": None, "document_id": None, "generator": "company-analysis-report", "error": str(e)}
    except Exception as e:
        return {"document": None, "document_id": None, "generator": "company-analysis-report",
                "error": f"Document engine error: {e}"}
    title = data.get("override_report_title") or data.get("report_title") or "会社データ分析報告書"
    author = data.get("author_name") or data.get("author") or "KANAME [要]"
    document_id = None
    try:
        metadata = dict(data)
        for k in list(metadata.keys()):
            v = metadata[k]
            if isinstance(v, list) and len(v) > 50:
                metadata[k] = f"[truncated list len={len(v)}]"
            if isinstance(v, str) and len(v) > 2000:
                metadata[k] = v[:2000] + "…[truncated]"
        document_id = doc_svc.create_document(
            db=db, generator="company-analysis-report", filename=report.filename,
            format=report.format, title=str(title), author=str(author)[:256],
            size_bytes=int(report.size_bytes), metadata=metadata, storage_path=None,
        )
    except Exception:
        document_id = None
    response = {
        "document_id": document_id, "generator": "company-analysis-report",
        "filename": report.filename, "size_bytes": report.size_bytes, "format": report.format,
        "title": str(title), "author": str(author),
        "message": "company-analysis-report generated. POST identical JSON to /generate/company-analysis-report/download to retrieve.",
    }
    if translation_report is not None:
        response["auto_translate"] = translation_report
    return response


@router.post("/generate/company-analysis-report/download", tags=["Document Intelligence"])
async def download_company_analysis_report(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
    auto_translate: bool = Query(
        True,
        description="If true, English free-text fields are translated to Japanese before generation "
                    "(KANAME supports JA<->EN only).",
    ),
):
    try:
        validated = CompanyAnalysisReportRequest.model_validate(payload)
    except ValidationError as e:
        errors = [err.get("msg", str(err)) for err in e.errors()]
        return JSONResponse(status_code=200, content={
            "error": f"Schema validation failed: {'; '.join(errors)}"
        })
    data = validated.model_dump()
    fmt = str(data.get("doc_format") or "docx").lower()
    if fmt not in ("docx", "pdf"):
        return JSONResponse(status_code=200, content={"error": f"Unsupported format {fmt}"})

    translation_report = None
    if auto_translate:
        data, translation_report = doc_localize.localize_document_data(
            CompanyAnalysisReportRequest, data, target_lang="ja"
        )

    try:
        report = official_docs.generate_official_document("company-analysis-report", fmt, data)
    except ValueError as e:
        return JSONResponse(status_code=200, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": f"Document engine error: {e}"})
    title = data.get("override_report_title") or data.get("report_title") or "会社データ分析報告書"
    author = data.get("author_name") or data.get("author") or "KANAME [要]"
    try:
        metadata = dict(data)
        for k in list(metadata.keys()):
            v = metadata[k]
            if isinstance(v, list) and len(v) > 50:
                metadata[k] = f"[truncated list len={len(v)}]"
            if isinstance(v, str) and len(v) > 2000:
                metadata[k] = v[:2000] + "…[truncated]"
        doc_svc.create_document(
            db=db, generator="company-analysis-report", filename=report.filename,
            format=report.format, title=str(title), author=str(author)[:256],
            size_bytes=int(report.size_bytes), metadata=metadata, storage_path=None,
        )
    except Exception:
        pass
    media = _media_type_for(fmt)
    filename = _sanitize_filename(report.filename)
    ascii_name = "".join(c for c in filename if ord(c) < 128) or "document"
    if "." in filename and not ascii_name.endswith("." + filename.rsplit(".", 1)[-1]):
        ext = filename.rsplit(".", 1)[-1]
        if "".join(c for c in ext if ord(c) < 128) == ext:
            ascii_name = f"{ascii_name}.{ext}"
    encoded = quote(filename)
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}; filename=\"{ascii_name}\""}
    if translation_report is not None:
        headers["X-Auto-Translate-Count"] = str(translation_report["translated_count"])
    return StreamingResponse(io.BytesIO(report.bytes), media_type=media, headers=headers)


@router.post("/generate/{template_id}", tags=["Document Intelligence"])
async def generate_official_doc_metadata(
    template_id: str,
    payload: dict[str, Any],
    db: Session = Depends(get_db),
    auto_translate: bool = Query(
        True,
        description="If true, English free-text fields are translated to Japanese before generation "
                    "(KANAME supports JA<->EN only).",
    ),
):
    if template_id not in TEMPLATE_REQUEST_SCHEMAS:
        return {
            "document": None,
            "document_id": None,
            "generator": template_id,
            "error": (
                f"Unknown template_id: {template_id}. "
                f"Available: {list(TEMPLATE_REQUEST_SCHEMAS.keys())}"
            ),
        }
    schema_cls = TEMPLATE_REQUEST_SCHEMAS[template_id]
    try:
        validated = schema_cls.model_validate(payload)
    except ValidationError as e:
        errors = [err.get("msg", str(err)) for err in e.errors()]
        return {
            "document": None,
            "document_id": None,
            "generator": template_id,
            "error": f"Schema validation failed for {template_id}: {'; '.join(errors)}",
        }
    data = validated.model_dump()
    fmt = str(data.get("doc_format") or "docx").lower()
    if fmt not in ("docx", "pdf"):
        return {
            "document": None,
            "document_id": None,
            "generator": template_id,
            "error": f"Unsupported format {fmt} (expected docx or pdf)",
        }

    translation_report = None
    if auto_translate:
        data, translation_report = doc_localize.localize_document_data(schema_cls, data, target_lang="ja")

    try:
        report = official_docs.generate_official_document(template_id, fmt, data)
    except ValueError as e:
        return {"document": None, "generator": template_id, "document_id": None, "error": str(e)}
    except Exception as e:
        return {"document": None, "generator": template_id, "document_id": None, "error": f"Document engine error: {e}"}

    title = data.get("title") or {"japanese-resume": "履歴書", "career-history": "職務経歴書", "invoice": "請求書", "business-report": data.get("title") or "報告書", "cover-letter": "送付状"}.get(template_id, "文書")
    author = data.get("author") or data.get("seller_name") or data.get("sender_name") or data.get("full_name_kanji") or "KANAME [要]"
    document_id = None
    try:
        metadata = dict(data)
        for k in list(metadata.keys()):
            v = metadata[k]
            if isinstance(v, list) and len(v) > 50:
                metadata[k] = f"[truncated list len={len(v)}]"
            if isinstance(v, str) and len(v) > 2000:
                metadata[k] = v[:2000] + "…[truncated]"
        document_id = doc_svc.create_document(
            db=db,
            generator=template_id,
            filename=report.filename,
            format=report.format,
            title=str(title),
            author=str(author)[:256],
            size_bytes=int(report.size_bytes),
            metadata=metadata,
            storage_path=None,
        )
    except Exception:
        document_id = None
    response = {
        "document_id": document_id,
        "generator": template_id,
        "filename": report.filename,
        "size_bytes": report.size_bytes,
        "format": report.format,
        "title": str(title),
        "author": str(author),
        "message": (
            f"{template_id} generated successfully. "
            "POST identical JSON body to /generate/{template_id}/download to retrieve the file bytes."
        ),
    }
    if translation_report is not None:
        response["auto_translate"] = translation_report
    return response


@router.post("/generate/{template_id}/download", tags=["Document Intelligence"])
async def download_official_doc(
    template_id: str,
    payload: dict[str, Any],
    db: Session = Depends(get_db),
    auto_translate: bool = Query(
        True,
        description="If true, English free-text fields are translated to Japanese before generation "
                    "(KANAME supports JA<->EN only).",
    ),
):
    if template_id not in TEMPLATE_REQUEST_SCHEMAS:
        return JSONResponse(
            status_code=200,
            content={
                "document": None,
                "document_id": None,
                "generator": template_id,
                "error": (
                    f"Unknown template_id: {template_id}. "
                    f"Available: {list(TEMPLATE_REQUEST_SCHEMAS.keys())}"
                ),
            },
        )
    schema_cls = TEMPLATE_REQUEST_SCHEMAS[template_id]
    try:
        validated = schema_cls.model_validate(payload)
    except ValidationError as e:
        errors = [err.get("msg", str(err)) for err in e.errors()]
        return JSONResponse(status_code=200, content={
            "error": f"Schema validation failed for {template_id}: {'; '.join(errors)}"
        })
    data = validated.model_dump()
    fmt = str(data.get("doc_format") or "docx").lower()
    if fmt not in ("docx", "pdf"):
        return JSONResponse(status_code=200, content={"error": f"Unsupported format {fmt}"})

    translation_report = None
    if auto_translate:
        data, translation_report = doc_localize.localize_document_data(schema_cls, data, target_lang="ja")

    try:
        report = official_docs.generate_official_document(template_id, fmt, data)
    except ValueError as e:
        return JSONResponse(status_code=200, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": f"Document engine error: {e}"})
    title = data.get("title") or {"japanese-resume": "履歴書", "career-history": "職務経歴書", "invoice": "請求書", "cover-letter": "送付状"}.get(template_id, data.get("title") or "文書")
    author = data.get("author") or data.get("seller_name") or data.get("sender_name") or data.get("full_name_kanji") or "KANAME [要]"
    try:
        metadata = dict(data)
        for k in list(metadata.keys()):
            v = metadata[k]
            if isinstance(v, list) and len(v) > 50:
                metadata[k] = f"[truncated list len={len(v)}]"
            if isinstance(v, str) and len(v) > 2000:
                metadata[k] = v[:2000] + "…[truncated]"
        doc_svc.create_document(
            db=db, generator=template_id, filename=report.filename,
            format=report.format, title=str(title), author=str(author)[:256],
            size_bytes=int(report.size_bytes), metadata=metadata, storage_path=None,
        )
    except Exception:
        pass
    media = _media_type_for(fmt)
    filename = _sanitize_filename(report.filename)
    ascii_name = "".join(c for c in filename if ord(c) < 128) or "document"
    if "." in filename and not ascii_name.endswith("." + filename.rsplit(".", 1)[-1]):
        ext = filename.rsplit(".", 1)[-1]
        if "".join(c for c in ext if ord(c) < 128) == ext:
            ascii_name = f"{ascii_name}.{ext}"
    encoded = quote(filename)
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}; filename=\"{ascii_name}\""}
    if translation_report is not None:
        headers["X-Auto-Translate-Count"] = str(translation_report["translated_count"])
    return StreamingResponse(io.BytesIO(report.bytes), media_type=media, headers=headers)


@router.get("", tags=["Document Intelligence"])
async def list_documents(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    try:
        documents = doc_svc.list_documents(db, limit=limit)
    except Exception as e:
        return {"documents": [], "error": f"Database unavailable: {e}", "count": 0}
    return {"documents": documents, "count": len(documents)}


@router.get("/{document_id}", tags=["Document Intelligence"])
async def get_document_by_id(document_id: str, db: Session = Depends(get_db)):
    try:
        item = doc_svc.get_document(db, document_id)
    except Exception as e:
        return {"document": None, "error": f"Database unavailable: {e}"}
    if item is None:
        return {"document": None, "error": "document_id not found"}
    return {"document": item}


@router.get("/{document_id}/download", tags=["Document Intelligence"])
async def download_document_by_id(document_id: str, db: Session = Depends(get_db)):
    try:
        item = doc_svc.get_document(db, document_id)
    except Exception as e:
        return {"document": None, "error": f"Database unavailable: {e}"}
    if item is None:
        return {"document": None, "error": "document_id not found"}
    storage_path = item.get("storage_path") if isinstance(item, dict) else getattr(item, "storage_path", None)
    if storage_path and os.path.exists(storage_path):
        try:
            with open(storage_path, "rb") as f:
                file_bytes = f.read()
            fmt = (item.get("format") if isinstance(item, dict) else getattr(item, "format", "docx")) or "docx"
            fname = (item.get("filename") if isinstance(item, dict) else getattr(item, "filename", f"document.{fmt}")) or f"document.{fmt}"
            media_type = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if fmt == "docx" else "application/pdf"
            )
            headers = _content_disposition_headers(fname)
            return StreamingResponse(io.BytesIO(file_bytes), media_type=media_type, headers=headers)
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
    return {"error": "File storage_path not set or file not found on disk. File bytes not persisted in DB; regenerate via generate endpoint."}
