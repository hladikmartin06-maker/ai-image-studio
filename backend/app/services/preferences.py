"""Per-user preferences (Settings page), stored as one JSON document per user."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserSettings


class Preferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # General
    theme: Literal["dark", "light", "system"] = "dark"
    language: Literal["en", "sk"] = "en"
    default_model: Literal["nano-banana-pro", "seedream", "wan-2.7", "wan-2.7-pro"] = (
        "nano-banana-pro"
    )
    # The provider used for new generations ("Default provider" / "active provider").
    active_provider: Literal["wavespeed", "atlas"] | None = None
    default_image_count: int = Field(default=1, ge=1, le=4)
    default_aspect_ratio: Literal["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16"] = "1:1"

    # Appearance
    sidebar: Literal["expanded", "compact"] = "expanded"
    grid_density: Literal["compact", "comfortable", "large"] = "comfortable"
    reduce_motion: bool = False
    show_prompts_on_cards: bool = True
    blur_nsfw: bool = True

    # Generation
    # Preferred size tier; models without it use their own default. None = model default.
    default_resolution: Literal["1K", "2K", "4K"] | None = None
    default_output_format: Literal["png", "jpeg"] = "png"
    default_thinking_mode: bool = True
    remember_prompt: bool = True
    clear_prompt_after_generate: bool = False
    auto_save: bool = True
    # Failed generations are removed from the history after this many days (0 = keep).
    keep_failed_days: Literal[0, 7, 30, 90] = 0

    # Storage
    download_name: Literal["prompt", "id", "date"] = "prompt"


class PreferencesPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: Literal["dark", "light", "system"] | None = None
    language: Literal["en", "sk"] | None = None
    default_model: Literal["nano-banana-pro", "seedream", "wan-2.7", "wan-2.7-pro"] | None = None
    active_provider: Literal["wavespeed", "atlas"] | None = None
    default_image_count: int | None = Field(default=None, ge=1, le=4)
    default_aspect_ratio: Literal["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16"] | None = None
    sidebar: Literal["expanded", "compact"] | None = None
    grid_density: Literal["compact", "comfortable", "large"] | None = None
    reduce_motion: bool | None = None
    show_prompts_on_cards: bool | None = None
    blur_nsfw: bool | None = None
    default_resolution: Literal["1K", "2K", "4K", "auto"] | None = None
    default_output_format: Literal["png", "jpeg"] | None = None
    default_thinking_mode: bool | None = None
    remember_prompt: bool | None = None
    clear_prompt_after_generate: bool | None = None
    auto_save: bool | None = None
    keep_failed_days: Literal[0, 7, 30, 90] | None = None
    download_name: Literal["prompt", "id", "date"] | None = None


async def load(db: AsyncSession, user_id: uuid.UUID) -> Preferences:
    row = await db.get(UserSettings, user_id)
    return Preferences.model_validate(row.data if row else {})


async def save(db: AsyncSession, user_id: uuid.UUID, patch: PreferencesPatch) -> Preferences:
    row = await db.get(UserSettings, user_id)
    current = dict(row.data) if row else {}
    changes = patch.model_dump(exclude_unset=True)
    if changes.get("default_resolution") == "auto":
        changes["default_resolution"] = None
    current.update(changes)
    prefs = Preferences.model_validate(current)
    if row is None:
        db.add(UserSettings(user_id=user_id, data=prefs.model_dump()))
    else:
        row.data = prefs.model_dump()
    await db.commit()
    return prefs
