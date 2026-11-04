"""advanced generation jobs (Generate › Advanced)

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-04 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "advanced_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("client_request_id", sa.String(length=64), nullable=True),
        sa.Column("connection_id", sa.UUID(), nullable=True),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("variant", sa.String(length=16), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("source", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("lora_version_id", sa.UUID(), nullable=True),
        sa.Column("lora_strength", sa.Float(), nullable=False),
        sa.Column("options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("spec", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("stages", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("reports", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_code", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("result_image_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["connection_id"], ["comfy_connections.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["lora_version_id"], ["lora_versions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["result_image_id"], ["images.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "client_request_id", name="uq_advanced_job_request"),
    )
    op.create_index(
        "ix_advanced_jobs_user_created", "advanced_jobs", ["user_id", "created_at"], unique=False
    )
    op.create_index(op.f("ix_advanced_jobs_user_id"), "advanced_jobs", ["user_id"], unique=False)
    op.create_index(op.f("ix_advanced_jobs_status"), "advanced_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_advanced_jobs_status"), table_name="advanced_jobs")
    op.drop_index(op.f("ix_advanced_jobs_user_id"), table_name="advanced_jobs")
    op.drop_index("ix_advanced_jobs_user_created", table_name="advanced_jobs")
    op.drop_table("advanced_jobs")
