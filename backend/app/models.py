"""Database tables. Every user-owned row carries user_id and every query filters on it."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> uuid.UUID:
    return uuid.uuid4()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # SHA-256 of the cookie token; the token itself is never stored.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    user_agent: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Integration(Base):
    """A user's credentials for one provider. The API key is stored encrypted (AES-GCM)."""

    __tablename__ = "integrations"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_integration_provider"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    encrypted_key: Mapped[bytes] = mapped_column(LargeBinary)
    key_last4: Mapped[str] = mapped_column(String(8))
    # connected | error
    status: Mapped[str] = mapped_column(String(16), default="connected")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ReferenceImage(Base):
    __tablename__ = "reference_images"
    __table_args__ = (Index("ix_reference_images_user_sha", "user_id", "sha256"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    file_path: Mapped[str] = mapped_column(String(500))
    thumb_path: Mapped[str] = mapped_column(String(500))
    original_name: Mapped[str] = mapped_column(String(255), default="")
    mime_type: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    # URLs returned by each provider's upload API, so a file is uploaded once per provider.
    provider_urls: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    # When each of those uploads happened (ISO time per provider); old uploads are renewed.
    provider_uploaded_at: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    # SHA-256 of the file: the same image uploaded again reuses this row (no second copy).
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Generation(Base):
    """One generation request as the user submitted it (the history entry)."""

    __tablename__ = "generations"
    __table_args__ = (
        Index("ix_generations_user_created", "user_id", "created_at"),
        UniqueConstraint("user_id", "client_request_id", name="uq_generation_client_request"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Sent by the browser with every submit so a double click never creates two jobs.
    client_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_key: Mapped[str] = mapped_column(String(40))
    provider: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(20))  # text-to-image | image-edit
    provider_model_id: Mapped[str] = mapped_column(String(120))
    prompt: Mapped[str] = mapped_column(Text)
    # The settings chosen in the UI (aspect ratio, resolution, seed, ...).
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # Exactly what was sent to the provider for one request (never contains credentials).
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    reference_image_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    image_count: Mapped[int] = mapped_column(Integer, default=1)
    # queued | processing | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_generation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # The saved AI model (character) used for this request, if any. The name and the
    # references it contributed are kept in persona_meta so history survives a deletion.
    persona_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("personas.id", ondelete="SET NULL"), index=True
    )
    persona_meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # The prompt exactly as sent when the AI model's description was added to it.
    final_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    tasks: Mapped[list["ProviderTask"]] = relationship(
        back_populates="generation", cascade="all, delete-orphan", order_by="ProviderTask.index"
    )
    images: Mapped[list["Image"]] = relationship(
        back_populates="generation", order_by="Image.index"
    )


class ProviderTask(Base):
    """One request at the provider. A generation of 4 images may be 4 tasks (or 1 with n=4)."""

    __tablename__ = "provider_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    generation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("generations.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(Integer, default=0)
    images_requested: Mapped[int] = mapped_column(Integer, default=1)
    # pending (not sent) | submitting | processing | completed | failed
    # "submitting" survives a crash only when the provider may have accepted the request;
    # such a task is never sent again (it could be charged twice) and is failed instead.
    status: Mapped[str] = mapped_column(String(16), default="pending")
    provider_task_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    outputs: Mapped[list[str]] = mapped_column(JSONB, default=list)
    nsfw_flags: Mapped[list[bool]] = mapped_column(JSONB, default=list)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    generation: Mapped[Generation] = relationship(back_populates="tasks")


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        Index("ix_images_user_created", "user_id", "created_at"),
        Index("ix_images_user_favorite", "user_id", "is_favorite"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    generation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("generations.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str] = mapped_column(String(500))
    thumb_path: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    # True when the provider reported the image as mature content.
    nsfw: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # False when "save automatically" is off and the user has not saved it yet.
    in_library: Mapped[bool] = mapped_column(Boolean, default=True)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    favorited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    generation: Mapped[Generation] = relationship(back_populates="images")


# --- AI models (saved fictional characters) ---------------------------------------------------


class Persona(Base):
    """A reusable fictional adult character: visual description and reference images."""

    __tablename__ = "personas"
    __table_args__ = (Index("ix_personas_user_updated", "user_id", "updated_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(80))
    # Added to the prompt (with the appearance fields) when the user keeps it switched on.
    description: Mapped[str] = mapped_column(Text, default="")
    # Private notes, never sent anywhere.
    notes: Mapped[str] = mapped_column(Text, default="")
    # Visual characteristics: age, gender, hair, face, body, style, ... (see services/personas.py).
    appearance: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # The primary reference image (sent first).
    cover_reference_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reference_images.id", ondelete="SET NULL")
    )
    adult_confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    images: Mapped[list["PersonaImage"]] = relationship(
        back_populates="persona", cascade="all, delete-orphan", order_by="PersonaImage.position"
    )


class PersonaImage(Base):
    """One reference image in an AI model's collection (the file is a ReferenceImage, so a
    duplicated model shares files and provider uploads instead of copying them)."""

    __tablename__ = "persona_images"
    __table_args__ = (
        UniqueConstraint("persona_id", "reference_image_id", name="uq_persona_image"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    persona_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("personas.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    reference_image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reference_images.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    persona: Mapped[Persona] = relationship(back_populates="images")
    reference: Mapped[ReferenceImage] = relationship()


# --- AI assistant ---------------------------------------------------------------------------


class AIIntegration(Base):
    """A user's key for one chat AI provider (separate from image and GPU providers)."""

    __tablename__ = "ai_integrations"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_ai_integration_provider"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    encrypted_key: Mapped[bytes] = mapped_column(LargeBinary)
    key_last4: Mapped[str] = mapped_column(String(8))
    # Only for the generic OpenAI-compatible provider.
    base_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="connected")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Model ids reported by the provider's models endpoint (cached; refreshed on test).
    models: Mapped[list[str]] = mapped_column(JSONB, default=list)
    # The user's choice per cost tier: {"low": id, "balanced": id, "high": id}.
    tier_models: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ChatConversation(Base):
    __tablename__ = "chat_conversations"
    __table_args__ = (Index("ix_chat_conversations_user_updated", "user_id", "updated_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), default="")
    # Rolling summary of older turns (context management) and how far it reaches.
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The studio context text last sent in this conversation (sent again only when changed).
    last_context: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_conversations.id", ondelete="CASCADE")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text, default="")
    # Exactly what was sent to the model for this turn (user text + changed context), so
    # later turns replay an identical prefix and the provider's prompt cache keeps working.
    api_content: Mapped[str] = mapped_column(Text, default="")
    # complete | streaming | stopped | error
    status: Mapped[str] = mapped_column(String(16), default="complete")
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # api | local | cache — where the answer came from (local and cache cost nothing).
    source: Mapped[str] = mapped_column(String(16), default="api")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    usage_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AIUsageEvent(Base):
    """Every provider call, including ones without a visible message (summaries)."""

    __tablename__ = "ai_usage_events"
    __table_args__ = (Index("ix_ai_usage_user_created", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(120))
    tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    purpose: Mapped[str] = mapped_column(
        String(16), default="chat"
    )  # chat | summary | local | cache
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --- GPU compute ----------------------------------------------------------------------------


class GPUIntegration(Base):
    """Credentials for a GPU compute provider (Modal, RunPod). Stored encrypted as JSON."""

    __tablename__ = "gpu_integrations"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_gpu_integration_provider"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    encrypted_credentials: Mapped[bytes] = mapped_column(LargeBinary)
    key_last4: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), default="connected")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class GPUWorkload(Base):
    """A workload started from the studio (history and usage; the provider is the truth)."""

    __tablename__ = "gpu_workloads"
    __table_args__ = (UniqueConstraint("provider", "external_id", name="uq_gpu_workload_external"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(120))
    kind: Mapped[str] = mapped_column(String(16))  # pod | sandbox
    name: Mapped[str] = mapped_column(String(120), default="")
    gpu_type: Mapped[str] = mapped_column(String(80), default="")
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    image: Mapped[str] = mapped_column(String(300), default="")
    status: Mapped[str] = mapped_column(String(24), default="starting")
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
