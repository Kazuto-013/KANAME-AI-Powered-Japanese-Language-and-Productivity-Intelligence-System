"""initial_schema — nlp_analyses, voice_jobs, dataset_records, document_records

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-23

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "nlp_analyses",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.String(length=64), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=2), nullable=False, server_default="C"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("wakati", sa.Text(), nullable=True),
        sa.Column("kana_only_text", sa.Text(), nullable=True),
        sa.Column("pos_counts_json", sa.Text(), nullable=True),
        sa.Column("noun_list_json", sa.Text(), nullable=True),
        sa.Column("verb_list_json", sa.Text(), nullable=True),
        sa.Column("adjective_list_json", sa.Text(), nullable=True),
        sa.Column("tokens_json", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_id", name="uq_nlp_analyses_analysis_id"),
    )
    op.create_index("ix_nlp_analyses_analysis_id", "nlp_analyses", ["analysis_id"], unique=False)
    op.create_index("ix_nlp_created_at", "nlp_analyses", ["created_at"], unique=False)

    op.create_table(
        "voice_jobs",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("source_filename", sa.String(length=512), nullable=True),
        sa.Column("model_size", sa.String(length=32), nullable=False, server_default="base"),
        sa.Column("detected_language", sa.String(length=8), nullable=True),
        sa.Column("language_probability", sa.Float(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("segment_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("transcription_text", sa.Text(), nullable=True),
        sa.Column("warnings", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", name="uq_voice_jobs_job_id"),
    )
    op.create_index("ix_voice_jobs_job_id", "voice_jobs", ["job_id"], unique=False)
    op.create_index("ix_voice_jobs_created_at", "voice_jobs", ["created_at"], unique=False)

    op.create_table(
        "dataset_records",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("columns", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("columns_names", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", name="uq_dataset_records_dataset_id"),
    )
    op.create_index("ix_dataset_records_dataset_id", "dataset_records", ["dataset_id"], unique=False)
    op.create_index("ix_dataset_records_created_at", "dataset_records", ["created_at"], unique=False)

    op.create_table(
        "document_records",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("generator", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("format", sa.String(length=8), nullable=False, server_default="docx"),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("author", sa.String(length=256), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True, server_default="0"),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("storage_path", sa.String(length=1024), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", name="uq_document_records_document_id"),
    )
    op.create_index("ix_document_records_document_id", "document_records", ["document_id"], unique=False)
    op.create_index("ix_documents_created_at", "document_records", ["created_at"], unique=False)
    op.create_index("ix_documents_generator", "document_records", ["generator"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_documents_generator", table_name="document_records")
    op.drop_index("ix_documents_created_at", table_name="document_records")
    op.drop_index("ix_document_records_document_id", table_name="document_records")
    op.drop_table("document_records")

    op.drop_index("ix_dataset_records_created_at", table_name="dataset_records")
    op.drop_index("ix_dataset_records_dataset_id", table_name="dataset_records")
    op.drop_table("dataset_records")

    op.drop_index("ix_voice_jobs_created_at", table_name="voice_jobs")
    op.drop_index("ix_voice_jobs_job_id", table_name="voice_jobs")
    op.drop_table("voice_jobs")

    op.drop_index("ix_nlp_created_at", table_name="nlp_analyses")
    op.drop_index("ix_nlp_analyses_analysis_id", table_name="nlp_analyses")
    op.drop_table("nlp_analyses")
