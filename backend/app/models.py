"""Database tables. Every user-owned row carries user_id and every query filters on it."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
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
    # ComfyUI runs only: connection, workflow, model files, LoRAs, sampler values.
    comfy_meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # The LoRA version used (ComfyUI runs), for "images made with this LoRA".
    lora_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("lora_versions.id", ondelete="SET NULL"), index=True
    )
    # Hosted runs made as photos for a LoRA's dataset (LoRA › Dataset › "Create a dataset").
    dataset_lora_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loras.id", ondelete="SET NULL"), index=True
    )
    # Hosted runs: the price the user confirmed before it started (USD, the provider's own
    # estimate), for the spending overview; None when no price was known.
    cost_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
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
    # Live state reported by the backend (ComfyUI: queue position, sampler step).
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
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


# --- Secrets for other services -----------------------------------------------------------


class UserSecret(Base):
    """A credential for a service that is not a provider integration (e.g. a Hugging Face
    token for gated model downloads on the user's GPU). Stored encrypted (AES-GCM)."""

    __tablename__ = "user_secrets"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_secret_name"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(40))
    encrypted_value: Mapped[bytes] = mapped_column(LargeBinary)
    last4: Mapped[str] = mapped_column(String(8))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# --- ComfyUI --------------------------------------------------------------------------------


class ComfyConnection(Base):
    """A ComfyUI server: local, remote, or deployed by the studio on RunPod / Modal (then it
    is reached through the studio agent and `gpu_provider` / `workload_id` are set)."""

    __tablename__ = "comfy_connections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(80))
    # local | remote | runpod | modal
    kind: Mapped[str] = mapped_column(String(16))
    base_url: Mapped[str] = mapped_column(String(400), default="")
    # none | bearer | basic | agent
    auth_type: Mapped[str] = mapped_column(String(16), default="none")
    encrypted_secret: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    secret_last4: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # Local ComfyUI only: its folder, so LoRA files can be copied into models/loras.
    models_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    gpu_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    workload_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # new | ready | error | deploying | stopped
    status: Mapped[str] = mapped_column(String(16), default="new")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Last inspection: version, devices, detected model files, nodes (refreshed on test).
    info: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # Deployment settings and progress (studio-deployed connections).
    deploy: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ComfyWorkflow(Base):
    """An imported ComfyUI workflow. `api_json` (API format) is what runs; the UI-format
    file is kept for export only. `bindings` say which node inputs the studio fills."""

    __tablename__ = "comfy_workflows"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    # The model family it is configured for ("krea2") or None for any.
    family: Mapped[str | None] = mapped_column(String(32), nullable=True)
    api_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    ui_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    bindings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    node_types: Mapped[list[str]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# --- LoRA studio ----------------------------------------------------------------------------


class Lora(Base):
    """A character LoRA project: dataset, settings and its trained versions."""

    __tablename__ = "loras"
    __table_args__ = (
        UniqueConstraint("user_id", "trigger_word", name="uq_lora_trigger"),
        Index("ix_loras_user_updated", "user_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(80))
    trigger_word: Mapped[str] = mapped_column(String(40))
    # Class word used with the trigger ("woman", "man", ...) and for output preservation.
    class_word: Mapped[str] = mapped_column(String(40), default="")
    persona_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("personas.id", ondelete="SET NULL"), index=True
    )
    # Architecture family (e.g. "krea2") and the base variant it was trained on.
    architecture: Mapped[str] = mapped_column(String(32))
    base_variant: Mapped[str] = mapped_column(String(32))
    # trained | imported
    source: Mapped[str] = mapped_column(String(16), default="trained")
    caption_template: Mapped[str] = mapped_column(String(300), default="{trigger}, {caption}")
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list)
    notes: Mapped[str] = mapped_column(Text, default="")
    adult_confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    preferred_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class LoraDatasetItem(Base):
    """One training image (the original file is a ReferenceImage and is never modified;
    crop is applied to a copy when the dataset is packaged)."""

    __tablename__ = "lora_dataset_items"
    __table_args__ = (
        UniqueConstraint("lora_id", "reference_image_id", name="uq_lora_dataset_image"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    lora_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loras.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    reference_image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reference_images.id", ondelete="CASCADE"), index=True
    )
    caption: Mapped[str] = mapped_column(Text, default="")
    # manual | ai | empty
    caption_source: Mapped[str] = mapped_column(String(16), default="empty")
    # train | validation (validation images are kept out of training)
    split: Mapped[str] = mapped_column(String(16), default="train")
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    # Relative crop {x, y, w, h} in 0..1, or None for the whole image.
    crop: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # 64-bit difference hash (hex) for near-duplicate detection.
    dhash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    reference: Mapped[ReferenceImage] = relationship()


class LoraVersion(Base):
    """One training run (or an imported file) of a LoRA: config, GPU job, artifacts."""

    __tablename__ = "lora_versions"
    __table_args__ = (
        UniqueConstraint("lora_id", "number", name="uq_lora_version_number"),
        UniqueConstraint("user_id", "client_request_id", name="uq_lora_version_request"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    lora_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loras.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    client_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preset: Mapped[str] = mapped_column(String(32), default="")
    # The exact training settings (studio form) and the ai-toolkit config sent.
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    trainer_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # queued | preparing | uploading | running | downloading | completed | failed |
    # cancelled | interrupted   (imported files are "completed")
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage_detail: Mapped[str] = mapped_column(String(300), default="")
    gpu_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gpu_type: Mapped[str] = mapped_column(String(80), default="")
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    workload_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    agent_url: Mapped[str | None] = mapped_column(String(400), nullable=True)
    encrypted_agent_token: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    cost_per_hour: Mapped[float | None] = mapped_column(Float, nullable=True)
    auto_cleanup: Mapped[bool] = mapped_column(Boolean, default=True)
    # {step, total, loss: [[step, loss], ...], samples: [{step, file}], gpu: {...}}
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # Final LoRA file and kept checkpoints (relative storage paths).
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoints: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    # ComfyUI connections this version was copied to: {connection_id: {filename, at}}.
    deployments: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # Dataset snapshot used for this run (item ids, captions, crop) for reproducibility.
    dataset_manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    gpu_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    gpu_stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# --- Prompt snippets ----------------------------------------------------------------------------


class PromptSnippet(Base):
    """A saved piece of prompt text (lighting, outfit, place, …) inserted with one click; for
    every AI model, or only for one."""

    __tablename__ = "prompt_snippets"
    __table_args__ = (Index("ix_prompt_snippets_user_name", "user_id", "name"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    persona_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("personas.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(80))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
