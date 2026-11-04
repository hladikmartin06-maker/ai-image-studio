"""hosted runs made as photos for a LoRA's dataset

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-04 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("generations", sa.Column("dataset_lora_id", sa.UUID(), nullable=True))
    op.create_index(
        op.f("ix_generations_dataset_lora_id"), "generations", ["dataset_lora_id"], unique=False
    )
    op.create_foreign_key(
        "fk_generations_dataset_lora_id_loras",
        "generations",
        "loras",
        ["dataset_lora_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_generations_dataset_lora_id_loras", "generations", type_="foreignkey")
    op.drop_index(op.f("ix_generations_dataset_lora_id"), table_name="generations")
    op.drop_column("generations", "dataset_lora_id")
