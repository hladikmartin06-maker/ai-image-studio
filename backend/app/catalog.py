"""The four generation models and how each provider exposes them.

Everything here comes from the providers' public model pages and API references
(see `docs/PROVIDERS.md` for the sources and for what is confirmed vs. conservative).
A control is offered only when the provider documents the parameter for that exact
endpoint; nothing else is ever sent. Each provider endpoint ("variant") is described
on its own, because the same model has different model ids and parameters on
WaveSpeed and on Atlas Cloud, and for text-to-image vs. editing with reference images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderKey = Literal["wavespeed", "atlas"]
ModelKey = Literal[
    "nano-banana-pro", "seedream", "wan-2.7", "wan-2.7-pro", "wan-3.0", "wan-3.0-prime"
]
Mode = Literal["text-to-image", "image-edit", "text-to-video", "image-to-video"]
# The modes that send reference images (an edit, or a video from a photo).
REFERENCE_MODES = ("image-edit", "image-to-video")

PROVIDERS: dict[str, dict[str, Any]] = {
    "wavespeed": {
        "key": "wavespeed",
        "name": "WaveSpeed",
        "description": "WaveSpeedAI inference API. Pay-as-you-go balance, fast image models.",
        "website": "https://wavespeed.ai",
        "keys_url": "https://wavespeed.ai/accesskey",
        "docs_url": "https://wavespeed.ai/docs",
        "supports_balance": True,
        # Prices come from its API (list price per model, estimate per request).
        "supports_prices": True,
        "pricing_url": "https://wavespeed.ai/docs/how-pricing-works",
    },
    "atlas": {
        "key": "atlas",
        "name": "Atlas Cloud",
        "description": "Atlas Cloud unified media API with one endpoint for many image models.",
        "website": "https://www.atlascloud.ai",
        "keys_url": "https://www.atlascloud.ai/console/api-keys",
        "docs_url": "https://www.atlascloud.ai/docs",
        "supports_balance": False,
        # No price API: the user checks its pricing page.
        "supports_prices": False,
        "pricing_url": "https://www.atlascloud.ai/docs/models/price",
    },
}

MAX_PROMPT_CHARS = 4000
# Image count is reached with several requests when an endpoint has no count parameter.
MAX_IMAGES_PER_GENERATION = 4
SEED_MAX = 2_147_483_647

# --- Size tables ----------------------------------------------------------------------
# Endpoints that take `size` as "WIDTH*HEIGHT" get one of these presets.
# 2K presets are ByteDance's documented Seedream presets (minimum 2560*1440 pixels);
# 1K presets are ~1 MP, multiples of 16; 4K presets stay within 4096 per side.
PIXEL_PRESETS: dict[str, dict[str, tuple[int, int]]] = {
    "1K": {
        "1:1": (1024, 1024),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "3:2": (1248, 832),
        "2:3": (832, 1248),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "21:9": (1344, 576),
    },
    "2K": {
        "1:1": (2048, 2048),
        "4:3": (2304, 1728),
        "3:4": (1728, 2304),
        "3:2": (2496, 1664),
        "2:3": (1664, 2496),
        "16:9": (2560, 1440),
        "9:16": (1440, 2560),
        "21:9": (3024, 1296),
    },
    "4K": {
        "1:1": (4096, 4096),
        "4:3": (4096, 3072),
        "3:4": (3072, 4096),
        "3:2": (3840, 2560),
        "2:3": (2560, 3840),
        "16:9": (4096, 2304),
        "9:16": (2304, 4096),
        "21:9": (3584, 1536),
    },
}

STANDARD_RATIOS = ("1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16")
NANO_BANANA_RATIOS = ("1:1", "3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "16:9", "9:16", "21:9")
SEEDREAM_RATIOS = STANDARD_RATIOS + ("21:9",)
WAN_RATIOS = STANDARD_RATIOS + ("21:9",)

# "match" = leave size/aspect out so the provider keeps the reference image's shape.
MATCH_INPUT = "match"


@dataclass(frozen=True)
class Variant:
    provider: ProviderKey
    mode: Mode
    model_id: str
    docs_url: str
    # How the image size is sent:
    #   aspect+resolution: `aspect_ratio` + `resolution` fields (Nano Banana Pro)
    #   pixels: `size` = "W*H" from PIXEL_PRESETS[resolution][aspect_ratio]
    #   tier: `size` = the resolution tier itself ("1K", "2K", "4K"); no aspect control
    size_mode: Literal["aspect+resolution", "pixels", "tier"]
    aspect_ratios: tuple[str, ...] = ()
    resolutions: tuple[str, ...] = ()
    default_resolution: str = ""
    default_aspect_ratio: str = "1:1"
    # Editing endpoints may keep the input's shape when size is omitted.
    allow_match_input: bool = False
    output_formats: tuple[str, ...] = ()
    seed: bool = False
    seed_min: int = 0
    thinking_mode: bool = False
    negative_prompt: bool = False
    # Name of the documented "number of images" field, when there is one.
    count_param: str | None = None
    min_reference_images: int = 0
    max_reference_images: int = 0
    notes: tuple[str, ...] = ()
    # Video endpoints: duration range in seconds, and the documented switches.
    durations: tuple[int, int] | None = None
    default_duration: int = 5
    audio: bool = False  # generate_audio
    prompt_expansion: bool = False  # enable_prompt_expansion

    @property
    def video(self) -> bool:
        return self.mode in ("text-to-video", "image-to-video")

    def controls(self) -> list[str]:
        out: list[str] = []
        if self.aspect_ratios:
            out.append("aspect_ratio")
        if self.resolutions:
            out.append("resolution")
        if self.output_formats:
            out.append("output_format")
        if self.seed:
            out.append("seed")
        if self.thinking_mode:
            out.append("thinking_mode")
        if self.negative_prompt:
            out.append("negative_prompt")
        if self.durations:
            out.append("duration")
        if self.audio:
            out.append("generate_audio")
        if self.prompt_expansion:
            out.append("enable_prompt_expansion")
        return out


@dataclass(frozen=True)
class ModelInfo:
    key: ModelKey
    name: str
    family: str
    description: str
    content: Literal["sfw", "sfw+nsfw"]
    variants: tuple[Variant, ...] = field(default_factory=tuple)
    # What it makes: images, or videos (Wan 3.0).
    media: Literal["image", "video"] = "image"

    def mode_for(self, has_references: bool) -> str:
        if self.media == "video":
            return "image-to-video" if has_references else "text-to-video"
        return "image-edit" if has_references else "text-to-image"

    def variant(self, provider: str, mode: str) -> Variant | None:
        for v in self.variants:
            if v.provider == provider and v.mode == mode:
                return v
        return None

    def providers(self) -> list[str]:
        return sorted({v.provider for v in self.variants})


WS = "https://wavespeed.ai/docs/docs-api"
AT = "https://www.atlascloud.ai/models"

# Wan 3.0 video (WaveSpeed). Verified 2026-11-02 from WaveSpeed's API docs pages
# docs-api/alibaba/alibaba-wan-3.0[-prime]-{text,image}-to-video: prompt (required),
# resolution 480p/720p/1080p (default 720p), aspect_ratio 16:9/9:16/1:1/4:3/3:4 (text: default
# 16:9; image: optional, else it follows the photo), duration 2-30 s (default 5),
# enable_prompt_expansion (default false), generate_audio (default true), seed (-1 = random);
# image to video: image (first frame, required), last_image (last frame, optional).
WAN3_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4")
WAN3_RESOLUTIONS = ("480p", "720p", "1080p")


def _wan3(model_id: str, slug: str) -> tuple[Variant, Variant]:
    common: dict[str, Any] = dict(
        provider="wavespeed",
        size_mode="aspect+resolution",
        aspect_ratios=WAN3_RATIOS,
        resolutions=WAN3_RESOLUTIONS,
        default_resolution="720p",
        default_aspect_ratio="16:9",
        seed=True,
        seed_min=-1,
        durations=(2, 30),
        default_duration=5,
        audio=True,
        prompt_expansion=True,
    )
    return (
        Variant(
            mode="text-to-video",
            model_id=f"{model_id}/text-to-video",
            docs_url=f"{WS}/alibaba/alibaba-{slug}-text-to-video",
            **common,
        ),
        Variant(
            mode="image-to-video",
            model_id=f"{model_id}/image-to-video",
            docs_url=f"{WS}/alibaba/alibaba-{slug}-image-to-video",
            allow_match_input=True,
            # The first reference is the first frame; a second one the last frame.
            min_reference_images=1,
            max_reference_images=2,
            **common,
        ),
    )


MODELS: dict[str, ModelInfo] = {
    "nano-banana-pro": ModelInfo(
        key="nano-banana-pro",
        name="Nano Banana Pro",
        family="Google Gemini 3 Pro Image",
        description="Google's studio-grade model: precise prompt following, legible text, "
        "consistent subjects and edits from reference photos.",
        content="sfw",
        variants=(
            Variant(
                provider="wavespeed",
                mode="text-to-image",
                model_id="google/nano-banana-pro/text-to-image",
                docs_url=f"{WS}/google/google-nano-banana-pro-text-to-image",
                size_mode="aspect+resolution",
                aspect_ratios=NANO_BANANA_RATIOS,
                resolutions=("1k", "2k", "4k"),
                default_resolution="1k",
                output_formats=("png", "jpeg"),
            ),
            Variant(
                provider="wavespeed",
                mode="image-edit",
                model_id="google/nano-banana-pro/edit",
                docs_url=f"{WS}/google/google-nano-banana-pro-edit",
                size_mode="aspect+resolution",
                aspect_ratios=NANO_BANANA_RATIOS,
                resolutions=("1k", "2k", "4k"),
                default_resolution="1k",
                allow_match_input=True,
                output_formats=("png", "jpeg"),
                min_reference_images=1,
                max_reference_images=4,
            ),
            Variant(
                provider="atlas",
                mode="text-to-image",
                model_id="google/nano-banana-pro/text-to-image",
                docs_url=f"{AT}/google/nano-banana-pro/text-to-image",
                size_mode="aspect+resolution",
                aspect_ratios=NANO_BANANA_RATIOS,
                resolutions=("1k", "2k", "4k"),
                default_resolution="1k",
                output_formats=("png", "jpeg"),
            ),
            Variant(
                provider="atlas",
                mode="image-edit",
                model_id="google/nano-banana-pro/edit",
                docs_url=f"{AT}/google/nano-banana-pro/edit",
                size_mode="aspect+resolution",
                aspect_ratios=NANO_BANANA_RATIOS,
                resolutions=("1k", "2k", "4k"),
                default_resolution="1k",
                allow_match_input=True,
                min_reference_images=1,
                max_reference_images=4,
            ),
        ),
    ),
    "seedream": ModelInfo(
        key="seedream",
        name="Seedream",
        family="ByteDance Seedream 4.5",
        description="ByteDance's high-resolution model for posters, products and brand "
        "visuals in 2K and 4K, with multi-reference editing.",
        content="sfw",
        variants=(
            Variant(
                provider="wavespeed",
                mode="text-to-image",
                model_id="bytedance/seedream-v4.5",
                docs_url=f"{WS}/bytedance/bytedance-seedream-v4.5",
                size_mode="pixels",
                aspect_ratios=SEEDREAM_RATIOS,
                resolutions=("2K", "4K"),
                default_resolution="2K",
            ),
            Variant(
                provider="wavespeed",
                mode="image-edit",
                model_id="bytedance/seedream-v4.5/edit",
                docs_url=f"{WS}/bytedance/bytedance-seedream-v4.5-edit",
                # The edit endpoint documents 1024 to 4096 pixels per side: 2K presets only.
                size_mode="pixels",
                aspect_ratios=SEEDREAM_RATIOS,
                resolutions=("2K",),
                default_resolution="2K",
                allow_match_input=True,
                min_reference_images=1,
                max_reference_images=10,
            ),
            Variant(
                provider="atlas",
                mode="text-to-image",
                model_id="bytedance/seedream-v4.5",
                docs_url="https://www.atlascloud.ai/docs/en/more-models/bytedance/seedream-v4.5/generateImage",
                size_mode="pixels",
                aspect_ratios=SEEDREAM_RATIOS,
                resolutions=("2K",),
                default_resolution="2K",
                notes=("Atlas lists fixed presets; only ByteDance's 2K presets are offered.",),
            ),
            Variant(
                provider="atlas",
                mode="image-edit",
                model_id="bytedance/seedream-v4.5/edit",
                docs_url=f"{AT}/bytedance/seedream-v4.5/edit",
                # Atlas documents no size field for this endpoint: the output follows the input.
                size_mode="pixels",
                min_reference_images=1,
                max_reference_images=4,
            ),
        ),
    ),
    "wan-2.7": ModelInfo(
        key="wan-2.7",
        name="Wan 2.7",
        family="Alibaba Wan 2.7 Image",
        description="Alibaba's fast open model with thinking mode. Allows mature (NSFW) "
        "content for adults.",
        content="sfw+nsfw",
        variants=(
            Variant(
                provider="wavespeed",
                mode="text-to-image",
                model_id="alibaba/wan-2.7/text-to-image",
                docs_url=f"{WS}/alibaba/alibaba-wan-2.7-text-to-image",
                size_mode="pixels",
                aspect_ratios=WAN_RATIOS,
                resolutions=("1K", "2K"),
                default_resolution="1K",
                seed=True,
                seed_min=-1,
                thinking_mode=True,
            ),
            Variant(
                provider="wavespeed",
                mode="image-edit",
                model_id="alibaba/wan-2.7/image-edit",
                docs_url=f"{WS}/alibaba/alibaba-wan-2.7-image-edit",
                size_mode="pixels",
                aspect_ratios=WAN_RATIOS,
                resolutions=("1K", "2K"),
                default_resolution="1K",
                allow_match_input=True,
                seed=True,
                seed_min=-1,
                min_reference_images=1,
                max_reference_images=3,
            ),
            Variant(
                provider="atlas",
                mode="text-to-image",
                model_id="alibaba/wan-2.7/text-to-image",
                docs_url=f"{AT}/alibaba/wan-2.7/text-to-image",
                size_mode="tier",
                resolutions=("1K", "2K"),
                default_resolution="2K",
                seed=True,
                thinking_mode=True,
                notes=("Atlas sets the size by resolution tier only (no aspect ratio).",),
            ),
            Variant(
                provider="atlas",
                mode="image-edit",
                model_id="alibaba/wan-2.7/image-edit",
                docs_url=f"{AT}/alibaba/wan-2.7/image-edit",
                size_mode="tier",
                resolutions=("1K", "2K"),
                default_resolution="2K",
                seed=True,
                min_reference_images=1,
                max_reference_images=3,
            ),
        ),
    ),
    "wan-2.7-pro": ModelInfo(
        key="wan-2.7-pro",
        name="Wan 2.7 Pro",
        family="Alibaba Wan 2.7 Image Pro",
        description="The professional Wan tier: higher fidelity and up to 4K output. "
        "Allows mature (NSFW) content for adults.",
        content="sfw+nsfw",
        variants=(
            Variant(
                provider="wavespeed",
                mode="text-to-image",
                model_id="alibaba/wan-2.7/text-to-image-pro",
                docs_url=f"{WS}/alibaba/alibaba-wan-2.7-text-to-image-pro",
                size_mode="pixels",
                aspect_ratios=WAN_RATIOS,
                resolutions=("1K", "2K", "4K"),
                default_resolution="2K",
                seed=True,
                seed_min=-1,
                thinking_mode=True,
            ),
            Variant(
                provider="wavespeed",
                mode="image-edit",
                model_id="alibaba/wan-2.7/image-edit-pro",
                docs_url=f"{WS}/alibaba/alibaba-wan-2.7-image-edit-pro",
                size_mode="pixels",
                aspect_ratios=WAN_RATIOS,
                resolutions=("1K", "2K"),
                default_resolution="2K",
                allow_match_input=True,
                seed=True,
                seed_min=-1,
                min_reference_images=1,
                max_reference_images=3,
            ),
            Variant(
                provider="atlas",
                mode="text-to-image",
                model_id="alibaba/wan-2.7-pro/text-to-image",
                docs_url=f"{AT}/alibaba/wan-2.7-pro/text-to-image",
                size_mode="tier",
                resolutions=("1K", "2K", "4K"),
                default_resolution="2K",
                seed=True,
                thinking_mode=True,
                count_param="n",
                notes=("Atlas sets the size by resolution tier only (no aspect ratio).",),
            ),
            Variant(
                provider="atlas",
                mode="image-edit",
                model_id="alibaba/wan-2.7-pro/image-edit",
                docs_url=f"{AT}/alibaba/wan-2.7-pro/image-edit",
                size_mode="tier",
                resolutions=("1K", "2K"),
                default_resolution="2K",
                seed=True,
                min_reference_images=1,
                max_reference_images=3,
            ),
        ),
    ),
    "wan-3.0": ModelInfo(
        key="wan-3.0",
        name="Wan 3.0",
        family="Alibaba Wan 3.0 (video)",
        description="Alibaba's video model: 2–30 s clips from a prompt or from a photo (first "
        "and last frame), with sound.",
        content="sfw",
        media="video",
        variants=_wan3("alibaba/wan-3.0", "wan-3.0"),
    ),
    "wan-3.0-prime": ModelInfo(
        key="wan-3.0-prime",
        name="Wan 3.0 Prime",
        family="Alibaba Wan 3.0 Prime (video)",
        description="The higher-quality Wan 3.0 video model, at a higher price per second.",
        content="sfw",
        media="video",
        variants=_wan3("alibaba/wan-3.0-prime", "wan-3.0-prime"),
    ),
}


class CatalogError(ValueError):
    """A request that the selected model/provider cannot run. The message is user-facing."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def get_model(key: str) -> ModelInfo:
    model = MODELS.get(key)
    if model is None:
        raise CatalogError("unsupported_model", f"Unknown model “{key}”.")
    return model


def resolve_variant(model_key: str, provider: str, has_references: bool) -> Variant:
    model = get_model(model_key)
    if provider not in PROVIDERS:
        raise CatalogError("unsupported_provider", f"Unknown provider “{provider}”.")
    mode = model.mode_for(has_references)
    variant = model.variant(provider, mode)
    if variant is None:
        what = "reference images" if has_references else mode.replace("-", " ")
        raise CatalogError(
            "unsupported_model",
            f"{model.name} with {what} is not available through {PROVIDERS[provider]['name']}.",
        )
    return variant


def normalize_settings(variant: Variant, raw: dict[str, Any]) -> dict[str, Any]:
    """Validates the UI settings against one endpoint and fills in defaults.

    Settings the endpoint does not document are dropped, so they can never reach the API.
    """
    out: dict[str, Any] = {}

    if variant.aspect_ratios:
        ratio = raw.get("aspect_ratio") or (
            MATCH_INPUT if variant.allow_match_input else variant.default_aspect_ratio
        )
        allowed = variant.aspect_ratios + ((MATCH_INPUT,) if variant.allow_match_input else ())
        if ratio not in allowed:
            # A ratio from another model (e.g. 4:5 after switching models) falls back quietly
            # only when it was never shown for this model; the UI always sends a shown value.
            raise CatalogError("invalid_parameter", f"Aspect ratio {ratio} is not supported.")
        out["aspect_ratio"] = ratio

    if variant.resolutions:
        resolution = raw.get("resolution") or variant.default_resolution
        if resolution not in variant.resolutions:
            resolution_lower = {r.lower(): r for r in variant.resolutions}.get(
                str(resolution).lower()
            )
            if resolution_lower is None:
                raise CatalogError(
                    "invalid_parameter", f"Resolution {resolution} is not supported."
                )
            resolution = resolution_lower
        out["resolution"] = resolution

    # With "match input", a pixel-preset endpoint receives no size at all, so a resolution
    # would only be misleading in the history.
    if variant.size_mode == "pixels" and out.get("aspect_ratio") == MATCH_INPUT:
        out.pop("resolution", None)

    if variant.output_formats:
        fmt = raw.get("output_format") or variant.output_formats[0]
        if fmt not in variant.output_formats:
            raise CatalogError("invalid_parameter", f"Output format {fmt} is not supported.")
        out["output_format"] = fmt

    if variant.seed:
        seed = raw.get("seed")
        if seed not in (None, ""):
            try:
                seed_int = int(seed)
            except (TypeError, ValueError) as exc:
                raise CatalogError("invalid_parameter", "Seed must be a whole number.") from exc
            if not (variant.seed_min <= seed_int <= SEED_MAX):
                raise CatalogError(
                    "invalid_parameter", f"Seed must be between {variant.seed_min} and {SEED_MAX}."
                )
            out["seed"] = seed_int

    if variant.thinking_mode:
        value = raw.get("thinking_mode")
        out["thinking_mode"] = True if value is None else bool(value)

    if variant.durations:
        low, high = variant.durations
        value = raw.get("duration")
        try:
            seconds = variant.default_duration if value in (None, "") else int(value)
        except (TypeError, ValueError) as exc:
            raise CatalogError("invalid_parameter", "Duration must be whole seconds.") from exc
        if not low <= seconds <= high:
            raise CatalogError("invalid_parameter", f"Duration must be {low} to {high} seconds.")
        out["duration"] = seconds
    if variant.audio:
        value = raw.get("generate_audio")
        out["generate_audio"] = True if value is None else bool(value)
    if variant.prompt_expansion:
        out["enable_prompt_expansion"] = bool(raw.get("enable_prompt_expansion", False))

    return out


def size_label(variant: Variant, settings: dict[str, Any]) -> str | None:
    """Human readable target size, e.g. "2048×2048", "2K · 16:9", "4K" or None (follows input)."""
    ratio = settings.get("aspect_ratio")
    resolution = settings.get("resolution")
    if variant.size_mode == "pixels":
        if resolution and ratio and ratio != MATCH_INPUT:
            w, h = PIXEL_PRESETS[resolution][ratio]
            return f"{w}×{h}"
        return None
    parts = [resolution.upper()] if resolution else []
    if ratio and ratio != MATCH_INPUT:
        parts.append(ratio)
    return " · ".join(parts) or None


def build_payload(
    variant: Variant,
    *,
    prompt: str,
    settings: dict[str, Any],
    image_urls: list[str],
    count: int,
) -> dict[str, Any]:
    """The provider request body for one task, from normalized settings only."""
    payload: dict[str, Any] = {"prompt": prompt}
    ratio = settings.get("aspect_ratio")
    resolution = settings.get("resolution")

    if variant.size_mode == "aspect+resolution":
        if ratio and ratio != MATCH_INPUT:
            payload["aspect_ratio"] = ratio
        if resolution:
            payload["resolution"] = resolution
    elif variant.size_mode == "pixels":
        if resolution and ratio and ratio != MATCH_INPUT:
            w, h = PIXEL_PRESETS[resolution][ratio]
            payload["size"] = f"{w}*{h}"
    elif variant.size_mode == "tier" and resolution:
        payload["size"] = resolution

    if "output_format" in settings:
        payload["output_format"] = settings["output_format"]
    if "seed" in settings:
        payload["seed"] = settings["seed"]
    if "thinking_mode" in settings:
        payload["thinking_mode"] = settings["thinking_mode"]
    for key in ("duration", "generate_audio", "enable_prompt_expansion"):
        if key in settings:
            payload[key] = settings[key]
    if variant.video:
        # Image to video: the first photo is the first frame, a second one the last frame.
        if image_urls:
            payload["image"] = image_urls[0]
        if len(image_urls) > 1:
            payload["last_image"] = image_urls[1]
    elif image_urls:
        payload["images"] = image_urls
    if variant.count_param and count > 1:
        payload[variant.count_param] = count
    return payload


def plan_tasks(variant: Variant, count: int) -> list[int]:
    """How many images each provider request asks for."""
    if variant.count_param:
        return [count]
    return [1] * count


def catalog_json() -> dict[str, Any]:
    def variant_json(v: Variant) -> dict[str, Any]:
        return {
            "provider": v.provider,
            "mode": v.mode,
            "model_id": v.model_id,
            "docs_url": v.docs_url,
            "controls": v.controls(),
            "size_mode": v.size_mode,
            "aspect_ratios": list(v.aspect_ratios),
            "default_aspect_ratio": (MATCH_INPUT if v.allow_match_input else "1:1")
            if v.aspect_ratios
            else None,
            "allow_match_input": v.allow_match_input,
            "resolutions": list(v.resolutions),
            "default_resolution": v.default_resolution or None,
            "output_formats": list(v.output_formats),
            "seed_min": v.seed_min if v.seed else None,
            "seed_max": SEED_MAX if v.seed else None,
            "max_reference_images": v.max_reference_images,
            "min_reference_images": v.min_reference_images,
            "max_images": MAX_IMAGES_PER_GENERATION,
            "batched": v.count_param is not None,
            "notes": list(v.notes),
            "durations": list(v.durations) if v.durations else None,
            "default_duration": v.default_duration if v.durations else None,
            "pixel_presets": {
                r: {a: list(PIXEL_PRESETS[r][a]) for a in v.aspect_ratios}
                for r in v.resolutions
                if v.size_mode == "pixels" and r in PIXEL_PRESETS
            },
        }

    return {
        "providers": list(PROVIDERS.values()),
        "models": [
            {
                "key": m.key,
                "name": m.name,
                "family": m.family,
                "description": m.description,
                "content": m.content,
                "media": m.media,
                "providers": m.providers(),
                "variants": [variant_json(v) for v in m.variants],
            }
            for m in MODELS.values()
        ],
        "limits": {
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "max_images": MAX_IMAGES_PER_GENERATION,
        },
    }
