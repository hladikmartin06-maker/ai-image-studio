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


def test_exactly_four_models():
    assert list(MODELS) == ["nano-banana-pro", "seedream", "wan-2.7", "wan-2.7-pro"]
    assert {m.content for m in MODELS.values()} == {"sfw", "sfw+nsfw"}
    assert MODELS["wan-2.7"].content == MODELS["wan-2.7-pro"].content == "sfw+nsfw"


@pytest.mark.parametrize("model", list(MODELS))
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
