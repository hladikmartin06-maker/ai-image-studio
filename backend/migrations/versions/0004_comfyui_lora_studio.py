"""comfyui, lora studio

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-31 15:14:06.249060
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "comfy_connections",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("base_url", sa.String(length=400), nullable=False),
        sa.Column("auth_type", sa.String(length=16), nullable=False),
        sa.Column("encrypted_secret", sa.LargeBinary(), nullable=True),
        sa.Column("secret_last4", sa.String(length=8), nullable=True),
        sa.Column("models_dir", sa.String(length=500), nullable=True),
        sa.Column("gpu_provider", sa.String(length=32), nullable=True),
        sa.Column("workload_id", sa.String(length=120), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("info", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("deploy", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_comfy_connections_user_id"), "comfy_connections", ["user_id"], unique=False
    )
    op.create_table(
        "comfy_workflows",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("family", sa.String(length=32), nullable=True),
        sa.Column("api_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ui_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("bindings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("node_types", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_comfy_workflows_user_id"), "comfy_workflows", ["user_id"], unique=False
    )
    op.create_table(
        "user_secrets",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=40), nullable=False),
        sa.Column("encrypted_value", sa.LargeBinary(), nullable=False),
        sa.Column("last4", sa.String(length=8), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_user_secret_name"),
    )
    op.create_index(op.f("ix_user_secrets_user_id"), "user_secrets", ["user_id"], unique=False)
    op.create_table(
        "loras",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("trigger_word", sa.String(length=40), nullable=False),
        sa.Column("class_word", sa.String(length=40), nullable=False),
        sa.Column("persona_id", sa.UUID(), nullable=True),
        sa.Column("architecture", sa.String(length=32), nullable=False),
        sa.Column("base_variant", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("caption_template", sa.String(length=300), nullable=False),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("adult_confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("preferred_version_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["persona_id"], ["personas.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "trigger_word", name="uq_lora_trigger"),
    )
    op.create_index(op.f("ix_loras_persona_id"), "loras", ["persona_id"], unique=False)
    op.create_index(op.f("ix_loras_user_id"), "loras", ["user_id"], unique=False)
    op.create_index("ix_loras_user_updated", "loras", ["user_id", "updated_at"], unique=False)
    op.create_table(
        "lora_dataset_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lora_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("reference_image_id", sa.UUID(), nullable=False),
        sa.Column("caption", sa.Text(), nullable=False),
        sa.Column("caption_source", sa.String(length=16), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("excluded", sa.Boolean(), nullable=False),
        sa.Column("crop", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("dhash", sa.String(length=16), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["lora_id"], ["loras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["reference_image_id"], ["reference_images.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lora_id", "reference_image_id", name="uq_lora_dataset_image"),
    )
    op.create_index(
        op.f("ix_lora_dataset_items_lora_id"), "lora_dataset_items", ["lora_id"], unique=False
    )
    op.create_index(
        op.f("ix_lora_dataset_items_reference_image_id"),
        "lora_dataset_items",
        ["reference_image_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_lora_dataset_items_user_id"), "lora_dataset_items", ["user_id"], unique=False
    )
    op.create_table(
        "lora_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lora_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("client_request_id", sa.String(length=64), nullable=True),
        sa.Column("preset", sa.String(length=32), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("trainer_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stage_detail", sa.String(length=300), nullable=False),
        sa.Column("gpu_provider", sa.String(length=32), nullable=True),
        sa.Column("gpu_type", sa.String(length=80), nullable=False),
        sa.Column("gpu_count", sa.Integer(), nullable=False),
        sa.Column("workload_id", sa.String(length=120), nullable=True),
        sa.Column("agent_url", sa.String(length=400), nullable=True),
        sa.Column("encrypted_agent_token", sa.LargeBinary(), nullable=True),
        sa.Column("cost_per_hour", sa.Float(), nullable=True),
        sa.Column("auto_cleanup", sa.Boolean(), nullable=False),
        sa.Column("progress", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column("file_sha256", sa.String(length=64), nullable=True),
        sa.Column("checkpoints", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("deployments", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("dataset_manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("error_code", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gpu_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gpu_stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["lora_id"], ["loras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lora_id", "number", name="uq_lora_version_number"),
        sa.UniqueConstraint("user_id", "client_request_id", name="uq_lora_version_request"),
    )
    op.create_index(op.f("ix_lora_versions_lora_id"), "lora_versions", ["lora_id"], unique=False)
    op.create_index(op.f("ix_lora_versions_status"), "lora_versions", ["status"], unique=False)
    op.create_index(op.f("ix_lora_versions_user_id"), "lora_versions", ["user_id"], unique=False)
    op.add_column(
        "generations",
        sa.Column(
            "comfy_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column("generations", sa.Column("lora_version_id", sa.UUID(), nullable=True))
    op.create_index(
        op.f("ix_generations_lora_version_id"), "generations", ["lora_version_id"], unique=False
    )
    op.create_foreign_key(
        "fk_generations_lora_version_id",
        "generations",
        "lora_versions",
        ["lora_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.add_column(
        "provider_tasks",
        sa.Column(
            "progress",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("provider_tasks", "progress")
    op.drop_constraint("fk_generations_lora_version_id", "generations", type_="foreignkey")
    op.drop_index(op.f("ix_generations_lora_version_id"), table_name="generations")
    op.drop_column("generations", "lora_version_id")
    op.drop_column("generations", "comfy_meta")
    op.drop_index(op.f("ix_lora_versions_user_id"), table_name="lora_versions")
    op.drop_index(op.f("ix_lora_versions_status"), table_name="lora_versions")
    op.drop_index(op.f("ix_lora_versions_lora_id"), table_name="lora_versions")
    op.drop_table("lora_versions")
    op.drop_index(op.f("ix_lora_dataset_items_user_id"), table_name="lora_dataset_items")
    op.drop_index(op.f("ix_lora_dataset_items_reference_image_id"), table_name="lora_dataset_items")
    op.drop_index(op.f("ix_lora_dataset_items_lora_id"), table_name="lora_dataset_items")
    op.drop_table("lora_dataset_items")
    op.drop_index("ix_loras_user_updated", table_name="loras")
    op.drop_index(op.f("ix_loras_user_id"), table_name="loras")
    op.drop_index(op.f("ix_loras_persona_id"), table_name="loras")
    op.drop_table("loras")
    op.drop_index(op.f("ix_user_secrets_user_id"), table_name="user_secrets")
    op.drop_table("user_secrets")
    op.drop_index(op.f("ix_comfy_workflows_user_id"), table_name="comfy_workflows")
    op.drop_table("comfy_workflows")
    op.drop_index(op.f("ix_comfy_connections_user_id"), table_name="comfy_connections")
    op.drop_table("comfy_connections")
