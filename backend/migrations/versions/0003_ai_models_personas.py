"""ai models (personas)

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-31 14:25:23.715987
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "personas",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("appearance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("cover_reference_id", sa.UUID(), nullable=True),
        sa.Column("adult_confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["cover_reference_id"], ["reference_images.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_personas_user_id"), "personas", ["user_id"], unique=False)
    op.create_index("ix_personas_user_updated", "personas", ["user_id", "updated_at"], unique=False)
    op.create_table(
        "persona_images",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("persona_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("reference_image_id", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["persona_id"], ["personas.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["reference_image_id"], ["reference_images.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("persona_id", "reference_image_id", name="uq_persona_image"),
    )
    op.create_index(
        op.f("ix_persona_images_persona_id"), "persona_images", ["persona_id"], unique=False
    )
    op.create_index(
        op.f("ix_persona_images_reference_image_id"),
        "persona_images",
        ["reference_image_id"],
        unique=False,
    )
    op.create_index(op.f("ix_persona_images_user_id"), "persona_images", ["user_id"], unique=False)
    op.add_column("generations", sa.Column("persona_id", sa.UUID(), nullable=True))
    op.add_column(
        "generations",
        sa.Column(
            "persona_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column("generations", sa.Column("final_prompt", sa.Text(), nullable=True))
    op.create_index(op.f("ix_generations_persona_id"), "generations", ["persona_id"], unique=False)
    op.create_foreign_key(
        "fk_generations_persona_id",
        "generations",
        "personas",
        ["persona_id"],
        ["id"],
        ondelete="SET NULL",
    )
    # Existing uploads have no time, so they are renewed once on their next use.
    op.add_column(
        "reference_images",
        sa.Column(
            "provider_uploaded_at",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column("reference_images", sa.Column("sha256", sa.String(length=64), nullable=True))
    op.create_index(
        "ix_reference_images_user_sha", "reference_images", ["user_id", "sha256"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_reference_images_user_sha", table_name="reference_images")
    op.drop_column("reference_images", "sha256")
    op.drop_column("reference_images", "provider_uploaded_at")
    op.drop_constraint("fk_generations_persona_id", "generations", type_="foreignkey")
    op.drop_index(op.f("ix_generations_persona_id"), table_name="generations")
    op.drop_column("generations", "final_prompt")
    op.drop_column("generations", "persona_meta")
    op.drop_column("generations", "persona_id")
    op.drop_index(op.f("ix_persona_images_user_id"), table_name="persona_images")
    op.drop_index(op.f("ix_persona_images_reference_image_id"), table_name="persona_images")
    op.drop_index(op.f("ix_persona_images_persona_id"), table_name="persona_images")
    op.drop_table("persona_images")
    op.drop_index("ix_personas_user_updated", table_name="personas")
    op.drop_index(op.f("ix_personas_user_id"), table_name="personas")
    op.drop_table("personas")
