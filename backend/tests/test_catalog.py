"""The catalog sends only documented parameters to each provider endpoint."""

import pytest

from app.catalog import (
    MATCH_INPUT,
    MODELS,
    PIXEL_PRESETS,
    CatalogError,
    build_payload,
    normalize_settings,
    resolve_variant,
    size_label,
)
from app.providers.base import error_from_response, normalize_status
from app.security import decrypt_secret, encrypt_secret, hash_password, verify_password

ALLOWED_KEYS = {
    "prompt",
    "aspect_ratio",
    "resolution",
    "output_format",
    "size",
    "seed",
    "thinking_mode",
    "images",
    "n",
}


IMAGE_MODELS = [k for k, m in MODELS.items() if m.media == "image"]


def test_the_models():
    assert list(MODELS) == [
        "nano-banana-pro",
        "seedream",
        "wan-2.7",
        "wan-2.7-pro",
        "wan-3.0",
        "wan-3.0-prime",
    ]
    assert IMAGE_MODELS == ["nano-banana-pro", "seedream", "wan-2.7", "wan-2.7-pro"]
    assert {m.content for m in MODELS.values()} == {"sfw", "sfw+nsfw"}
    assert MODELS["wan-2.7"].content == MODELS["wan-2.7-pro"].content == "sfw+nsfw"
    # Wan 3.0 makes videos, on WaveSpeed only (verified there); SFW.
    for key in ("wan-3.0", "wan-3.0-prime"):
        assert MODELS[key].media == "video" and MODELS[key].content == "sfw"
        assert MODELS[key].providers() == ["wavespeed"]


@pytest.mark.parametrize("model", ["wan-3.0", "wan-3.0-prime"])
def test_wan3_video_payloads(model):
    """Only the documented Wan 3.0 fields; a photo is the first frame, a second the last."""
    allowed = {
        "prompt",
        "resolution",
        "aspect_ratio",
        "duration",
        "enable_prompt_expansion",
        "generate_audio",
        "seed",
        "image",
        "last_image",
    }
    t2v = resolve_variant(model, "wavespeed", False)
    assert t2v.model_id == f"alibaba/{model}/text-to-video" and t2v.mode == "text-to-video"
    settings = normalize_settings(t2v, {"steps": 4, "negative_prompt": "x"})
    assert settings == {
        "aspect_ratio": "16:9",
        "resolution": "720p",
        "duration": 5,
        "generate_audio": True,
        "enable_prompt_expansion": False,
    }
    payload = build_payload(t2v, prompt="p", settings=settings, image_urls=[], count=1)
    assert set(payload) <= allowed and "image" not in payload
    i2v = resolve_variant(model, "wavespeed", True)
    assert i2v.model_id == f"alibaba/{model}/image-to-video" and i2v.max_reference_images == 2
    settings = normalize_settings(
        i2v, {"resolution": "1080p", "duration": 30, "generate_audio": False, "seed": -1}
    )
    payload = build_payload(i2v, prompt="p", settings=settings, image_urls=["a", "b"], count=1)
    assert set(payload) <= allowed
    assert payload["image"] == "a" and payload["last_image"] == "b"
    # Without a ratio the video follows the photo.
    assert "aspect_ratio" not in payload and payload["duration"] == 30 and payload["seed"] == -1
    assert payload["generate_audio"] is False
    for bad in ({"duration": 1}, {"duration": 31}, {"resolution": "4K"}, {"aspect_ratio": "21:9"}):
        with pytest.raises(CatalogError):
            normalize_settings(t2v, bad)
    with pytest.raises(CatalogError):
        resolve_variant(model, "atlas", False)


@pytest.mark.parametrize("model", IMAGE_MODELS)
@pytest.mark.parametrize("provider", ["wavespeed", "atlas"])
@pytest.mark.parametrize("refs", [False, True])
def test_every_variant_builds_a_documented_payload(model, provider, refs):
    variant = resolve_variant(model, provider, refs)
    settings = normalize_settings(variant, {"seed": 7, "steps": 50, "guidance_scale": 3})
    payload = build_payload(
        variant, prompt="p", settings=settings, image_urls=["u"] if refs else [], count=2
    )
    assert set(payload) <= ALLOWED_KEYS
    assert "steps" not in payload and "guidance_scale" not in payload
    if not variant.seed:
        assert "seed" not in payload
    if variant.count_param is None:
        assert "n" not in payload


def test_pixel_presets_are_within_documented_ranges():
    for tier, table in PIXEL_PRESETS.items():
        for ratio, (w, h) in table.items():
            assert 512 <= w <= 4096 and 512 <= h <= 4096, (tier, ratio)
    for w, h in PIXEL_PRESETS["2K"].values():
        assert w * h >= 2560 * 1440  # Seedream's documented minimum


def test_match_input_omits_size():
    variant = resolve_variant("wan-2.7-pro", "wavespeed", True)
    settings = normalize_settings(variant, {"aspect_ratio": MATCH_INPUT, "resolution": "2K"})
    payload = build_payload(variant, prompt="p", settings=settings, image_urls=["u"], count=1)
    assert "size" not in payload
    assert "resolution" not in settings  # not recorded when it is not sent
    assert size_label(variant, settings) is None
    nano = resolve_variant("nano-banana-pro", "atlas", True)
    nano_settings = normalize_settings(nano, {"aspect_ratio": MATCH_INPUT, "resolution": "2k"})
    assert build_payload(nano, prompt="p", settings=nano_settings, image_urls=["u"], count=1) == {
        "prompt": "p",
        "resolution": "2k",
        "images": ["u"],
    }
    assert size_label(nano, nano_settings) == "2K"


def test_invalid_values_are_rejected():
    variant = resolve_variant("nano-banana-pro", "wavespeed", False)
    with pytest.raises(CatalogError):
        normalize_settings(variant, {"resolution": "8k"})
    wan = resolve_variant("wan-2.7", "atlas", False)
    with pytest.raises(CatalogError):
        normalize_settings(wan, {"seed": -5})


def test_status_and_error_normalization():
    import httpx

    assert normalize_status("succeeded") == "completed"
    assert normalize_status("created") == "processing"
    assert normalize_status("failed") == "failed"
    r = httpx.Response(400, json={"message": "Insufficient credits"})
    assert error_from_response("wavespeed", r).code == "insufficient_credits"
    assert error_from_response("atlas", httpx.Response(401)).code == "invalid_credentials"
    assert error_from_response("atlas", httpx.Response(429)).code == "rate_limited"


def test_crypto_and_passwords():
    blob = encrypt_secret("secret-value", associated="u:p")
    assert b"secret-value" not in blob
    assert decrypt_secret(blob, associated="u:p") == "secret-value"
    with pytest.raises(Exception):  # noqa: B017 - bound to its row
        decrypt_secret(blob, associated="other:p")
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong", stored)
