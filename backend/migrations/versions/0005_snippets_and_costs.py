"""prompt snippets, confirmed price of hosted runs

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-02 20:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("generations", sa.Column("cost_estimate", sa.Float(), nullable=True))
    op.create_table(
        "prompt_snippets",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("persona_id", sa.UUID(), nullable=True),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["persona_id"], ["personas.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_prompt_snippets_persona_id"), "prompt_snippets", ["persona_id"], unique=False
    )
    op.create_index(
        op.f("ix_prompt_snippets_user_id"), "prompt_snippets", ["user_id"], unique=False
    )
    op.create_index(
        "ix_prompt_snippets_user_name", "prompt_snippets", ["user_id", "name"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_prompt_snippets_user_name", table_name="prompt_snippets")
    op.drop_index(op.f("ix_prompt_snippets_user_id"), table_name="prompt_snippets")
    op.drop_index(op.f("ix_prompt_snippets_persona_id"), table_name="prompt_snippets")
    op.drop_table("prompt_snippets")
    op.drop_column("generations", "cost_estimate")
