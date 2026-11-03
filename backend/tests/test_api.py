"""End-to-end API tests against the fake providers."""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Generation, Image, Integration, ProviderTask, User
from app.services.worker import worker
from tests.conftest import Api, make_user, wait_done
from tests.fakes import TEST_VIDEO, VALID_KEY, png_bytes


async def test_health(user: Api):
    r = await user.get("/api/health")
    assert r.json()["app"] == "ai-image-studio"
    assert r.json()["database"] == "ok"


async def test_requires_sign_in_and_client_header(user: Api):
    anon = Api(user.client.__class__(transport=user.client._transport, base_url="http://test"))
    assert (await anon.get("/api/images")).status_code == 401
    # Missing CSRF header on a state-changing request.
    r = await user.client.post("/api/generations", json={})
    assert r.status_code == 403
    await anon.client.aclose()


async def test_login_logout(fake):
    api = await make_user()
    me = (await api.get("/api/auth/me")).json()["user"]
    await api.post("/api/auth/logout")
    assert (await api.get("/api/auth/me")).status_code == 401
    bad = await api.post("/api/auth/login", json={"email": me["email"], "password": "nope-nope"})
    assert bad.status_code == 401
    ok = await api.post("/api/auth/login", json={"email": me["email"], "password": "password123"})
    assert ok.status_code == 200
    assert (await api.get("/api/auth/me")).status_code == 200
    await api.client.aclose()


async def test_auto_sign_in_opens_the_first_account(user: Api):
    anon = Api(user.client.__class__(transport=user.client._transport, base_url="http://test"))
    state = (await anon.get("/api/auth/state")).json()
    assert state["auto_login"] is True and state["user"] is None
    # Needs the client header like every state-changing request (other sites cannot use it).
    assert (await anon.client.post("/api/auth/auto", json={})).status_code == 403
    r = await anon.post("/api/auth/auto", json={"language": "sk"})
    assert r.status_code == 200
    async with SessionLocal() as db:
        first = await db.scalar(select(User).order_by(User.created_at, User.id).limit(1))
    assert r.json()["user"]["id"] == str(first.id)
    assert (await anon.get("/api/auth/me")).json()["user"]["id"] == str(first.id)

    # AUTO_LOGIN=false brings the sign-in screen back.
    settings = get_settings()
    settings.auto_login = False
    try:
        other = Api(user.client.__class__(transport=user.client._transport, base_url="http://test"))
        off = await other.post("/api/auth/auto", json={})
        assert off.status_code == 403 and off.json()["error"]["code"] == "auto_login_off"
        assert (await other.get("/api/auth/state")).json()["auto_login"] is False
        await other.client.aclose()
    finally:
        settings.auto_login = True
    await anon.client.aclose()


async def test_integration_key_is_encrypted_and_masked(user: Api, fake):
    r = await user.put("/api/integrations/wavespeed", json={"api_key": VALID_KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "connected"
    assert body["masked_key"].endswith(VALID_KEY[-4:])
    assert VALID_KEY not in r.text
    listing = await user.get("/api/integrations")
    assert VALID_KEY not in listing.text
    assert listing.json()["active_provider"] == "wavespeed"
    async with SessionLocal() as db:
        rows = (await db.scalars(select(Integration))).all()
        assert all(VALID_KEY.encode() not in row.encrypted_key for row in rows)
    balance = await user.get("/api/integrations/wavespeed/balance")
    assert balance.json()["balance"] == 12.5
    test = await user.post("/api/integrations/wavespeed/test")
    assert test.json()["ok"] is True


async def test_invalid_key_is_rejected(user: Api, fake):
    r = await user.put("/api/integrations/atlas", json={"api_key": "wrong-key-0000"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_credentials"
    listing = (await user.get("/api/integrations")).json()
    atlas = next(i for i in listing["integrations"] if i["key"] == "atlas")
    assert atlas["status"] == "not_connected"


async def test_generation_requires_connection(user: Api):
    r = await user.post(
        "/api/generations", json={"model": "nano-banana-pro", "prompt": "a red fox"}
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "not_connected"


async def test_nano_banana_on_wavespeed(connected: Api, fake):
    r = await connected.post(
        "/api/generations",
        json={
            "model": "nano-banana-pro",
            "provider": "wavespeed",
            "prompt": "a red fox in snow",
            "count": 2,
            "settings": {"aspect_ratio": "16:9", "resolution": "2k", "seed": 5, "steps": 30},
        },
    )
    assert r.status_code == 201, r.text
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "completed"
    assert len(gen["images"]) == 2
    # Two requests (no documented count parameter), only documented fields sent.
    assert len(fake.submits) == 2
    sent = fake.submits[0]
    assert sent["model"] == "google/nano-banana-pro/text-to-image"
    assert sent["body"] == {
        "prompt": "a red fox in snow",
        "aspect_ratio": "16:9",
        "resolution": "2k",
        "output_format": "png",
    }
    image = gen["images"][0]
    original = await connected.get(image["download_url"])
    assert original.status_code == 200
    assert original.content == png_bytes(96, 72)  # byte for byte (as the fake CDN serves)
    assert "attachment" in original.headers["content-disposition"]
    thumb = await connected.get(image["thumb_url"])
    assert thumb.headers["content-type"] == "image/webp"


async def test_wan_pro_on_atlas_uses_n(connected: Api, fake):
    r = await connected.post(
        "/api/generations",
        json={
            "model": "wan-2.7-pro",
            "provider": "atlas",
            "prompt": "studio portrait, soft light",
            "count": 3,
            "settings": {"resolution": "4K", "seed": 42, "thinking_mode": False},
        },
    )
    assert r.status_code == 201, r.text
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "completed"
    assert len(gen["images"]) == 3
    assert len(fake.submits) == 1
    assert fake.submits[0]["body"] == {
        "model": "alibaba/wan-2.7-pro/text-to-image",
        "prompt": "studio portrait, soft light",
        "size": "4K",
        "seed": 42,
        "thinking_mode": False,
        "n": 3,
    }


async def test_seedream_edit_uploads_references_once(connected: Api, fake):
    files = {"file": ("ref.png", png_bytes(300, 200), "image/png")}
    up = await connected.post("/api/reference-images", files=files)
    assert up.status_code == 201, up.text
    ref_id = up.json()["id"]
    body = {
        "model": "seedream",
        "provider": "wavespeed",
        "prompt": "make it a watercolor",
        "count": 2,
        "reference_image_ids": [ref_id],
        "settings": {"aspect_ratio": "3:2", "resolution": "2K"},
    }
    r = await connected.post("/api/generations", json=body)
    assert r.status_code == 201, r.text
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "completed"
    assert gen["mode"] == "image-edit"
    assert len(fake.uploads) == 1
    assert fake.submits[0]["model"] == "bytedance/seedream-v4.5/edit"
    assert fake.submits[0]["body"]["size"] == "2496*1664"
    assert fake.submits[0]["body"]["images"] == fake.uploads
    # Regenerate reuses the uploaded URL instead of uploading again.
    again = await connected.post(f"/api/generations/{gen['id']}/regenerate", json={"count": 1})
    assert again.status_code == 201
    regen = await wait_done(connected, again.json()["id"])
    assert regen["status"] == "completed"
    assert regen["source_generation_id"] == gen["id"]
    assert len(fake.uploads) == 1


async def test_unsupported_settings_and_limits(connected: Api, fake):
    r = await connected.post(
        "/api/generations",
        json={
            "model": "seedream",
            "provider": "wavespeed",
            "prompt": "x",
            "settings": {"aspect_ratio": "5:4"},
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
    r = await connected.post(
        "/api/generations", json={"model": "dall-e", "provider": "wavespeed", "prompt": "x"}
    )
    assert r.status_code == 400
    r = await connected.post(
        "/api/generations",
        json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "x", "count": 9},
    )
    assert r.status_code == 422


async def test_content_rules(connected: Api, fake):
    sfw = await connected.post(
        "/api/generations",
        json={"model": "nano-banana-pro", "provider": "wavespeed", "prompt": "a nude figure"},
    )
    assert sfw.status_code == 422
    minors = await connected.post(
        "/api/generations",
        json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "nude teen"},
    )
    assert minors.status_code == 422
    body = {"model": "wan-2.7", "provider": "wavespeed", "prompt": "artistic nude, adult woman"}
    # Wan allows 18+ content, but only with its "18+ content" switch on.
    off = await connected.post("/api/generations", json=body)
    assert off.status_code == 422 and off.json()["error"]["code"] == "adult_off"
    # The switch never opens SFW-only models.
    on_sfw = await connected.post(
        "/api/generations",
        json={**body, "model": "nano-banana-pro", "prompt": "a portrait", "adult": True},
    )
    assert on_sfw.status_code == 400
    adult = await connected.post("/api/generations", json={**body, "adult": True})
    assert adult.status_code == 201
    gen = await wait_done(connected, adult.json()["id"])
    assert fake.submits[-1]["model"] == "alibaba/wan-2.7/text-to-image"
    # Kept with the request, never sent to the provider; every image is marked 18+.
    assert gen["settings"]["adult"] is True
    assert "adult" not in fake.submits[-1]["body"] and "adult" not in gen["request_payload"]
    assert gen["images"] and all(img["nsfw"] is True for img in gen["images"])
    # Repeating it keeps the switch on.
    again = await connected.post(f"/api/generations/{gen['id']}/regenerate", json={"count": 1})
    assert again.status_code == 201, again.text
    assert (await wait_done(connected, again.json()["id"]))["settings"]["adult"] is True
    # A safe prompt with the switch off stays unmarked (the provider's own flag counts).
    safe = await connected.post("/api/generations", json={**body, "prompt": "a lighthouse at dusk"})
    safe_gen = await wait_done(connected, safe.json()["id"])
    assert "adult" not in safe_gen["settings"]
    assert all(img["nsfw"] is not True for img in safe_gen["images"])


async def test_provider_errors_are_normalized(connected: Api, fake):
    fake.submit_status = 402
    fake.submit_body = {"code": 402, "message": "Insufficient balance"}
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "a cat"}
    )
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "failed"
    assert gen["error_code"] == "insufficient_credits"

    fake.submit_status = 200
    fake.fail_message = "Generation failed: internal"
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "atlas", "prompt": "a cat"}
    )
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "failed"
    assert gen["error_code"] == "generation_failed"

    fake.fail_message = None
    fake.submit_status = 429
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "atlas", "prompt": "a cat"}
    )
    gen = await wait_done(connected, r.json()["id"])
    assert gen["error_code"] == "rate_limited"


async def test_duplicate_submit_creates_one_job(connected: Api, fake):
    body = {
        "model": "wan-2.7",
        "provider": "wavespeed",
        "prompt": "a lighthouse",
        "client_request_id": "abc-123",
    }
    first = await connected.post("/api/generations", json=body)
    second = await connected.post("/api/generations", json=body)
    assert first.json()["id"] == second.json()["id"]
    await wait_done(connected, first.json()["id"])
    assert len(fake.submits) == 1


async def test_users_are_isolated(connected: Api, fake):
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "a tree"}
    )
    gen = await wait_done(connected, r.json()["id"])
    image_id = gen["images"][0]["id"]
    other = await make_user("Other")
    assert (await other.get(f"/api/generations/{gen['id']}")).status_code == 404
    assert (await other.get(f"/api/images/{image_id}")).status_code == 404
    assert (await other.get(f"/api/images/{image_id}/file")).status_code == 404
    assert (await other.delete(f"/api/images/{image_id}")).status_code == 404
    assert (await other.get("/api/images")).json()["total"] == 0
    listing = (await other.get("/api/integrations")).json()
    assert all(i["status"] == "not_connected" for i in listing["integrations"])
    await other.client.aclose()


async def test_library_favorites_search_and_bulk(connected: Api, fake):
    for prompt in ("blue ocean waves", "green forest path"):
        r = await connected.post(
            "/api/generations",
            json={"model": "seedream", "provider": "wavespeed", "prompt": prompt},
        )
        await wait_done(connected, r.json()["id"])
    lib = (await connected.get("/api/images")).json()
    assert lib["total"] == 2
    ocean = (await connected.get("/api/images", params={"q": "ocean"})).json()
    assert ocean["total"] == 1
    by_model = (await connected.get("/api/images", params={"q": "seedream"})).json()
    assert by_model["total"] == 2
    ids = [i["id"] for i in lib["items"]]
    await connected.post("/api/images/bulk", json={"ids": ids, "action": "favorite"})
    favs = (await connected.get("/api/images", params={"favorite": True})).json()
    assert favs["total"] == 2
    sorted_favs = (await connected.get("/api/images", params={"sort": "favorited"})).json()
    assert sorted_favs["total"] == 2
    one = await connected.patch(f"/api/images/{ids[0]}", json={"is_favorite": False})
    assert one.json()["is_favorite"] is False
    archive = await connected.get("/api/images/archive", params={"ids": ",".join(ids)})
    assert archive.headers["content-type"] == "application/zip"
    await connected.post("/api/images/bulk", json={"ids": ids, "action": "delete"})
    assert (await connected.get("/api/images")).json()["total"] == 0
    stats = (await connected.get("/api/stats", params={"tz": "Europe/Bratislava"})).json()
    assert stats["generations"]["completed"] == 2
    assert stats["images_total"] == 0


async def test_paging(connected: Api, fake):
    r = await connected.post(
        "/api/generations",
        json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "pages", "count": 4},
    )
    await wait_done(connected, r.json()["id"])
    first = (await connected.get("/api/images", params={"limit": 3})).json()
    assert len(first["items"]) == 3 and first["next_cursor"]
    second = (
        await connected.get("/api/images", params={"limit": 3, "cursor": first["next_cursor"]})
    ).json()
    assert len(second["items"]) == 1 and second["next_cursor"] is None
    assert not {i["id"] for i in first["items"]} & {i["id"] for i in second["items"]}


async def test_auto_save_off_keeps_images_out_of_library(connected: Api, fake):
    await connected.patch("/api/settings", json={"auto_save": False})
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "draft"}
    )
    gen = await wait_done(connected, r.json()["id"])
    assert gen["images"][0]["in_library"] is False
    assert (await connected.get("/api/images")).json()["total"] == 0
    unsaved = (await connected.get("/api/images", params={"scope": "unsaved"})).json()
    assert unsaved["total"] == 1


async def test_recovery_never_resends_an_uncertain_request(connected: Api, fake):
    me = (await connected.get("/api/auth/me")).json()["user"]
    async with SessionLocal() as db:
        gen = Generation(
            user_id=uuid.UUID(me["id"]),
            model_key="wan-2.7",
            provider="wavespeed",
            mode="text-to-image",
            provider_model_id="alibaba/wan-2.7/text-to-image",
            prompt="crashed",
            settings={"aspect_ratio": "1:1", "resolution": "1K", "thinking_mode": True},
            status="processing",
            image_count=2,
            tasks=[
                ProviderTask(index=0, status="submitting"),
                ProviderTask(index=1, status="processing", provider_task_id="wavespeed-999"),
            ],
        )
        db.add(gen)
        await db.commit()
        gen_id = gen.id
    fake.predictions["wavespeed-999"] = {"model": "x", "count": 1, "polls": 0}
    await worker.recover()
    done = await wait_done(connected, str(gen_id))
    assert fake.submits == []  # nothing was sent again
    assert done["status"] == "completed"
    assert len(done["images"]) == 1
    statuses = sorted(t["error_code"] or "" for t in done["tasks"])
    assert statuses == ["", "interrupted"]


async def test_cancel_only_while_queued(connected: Api, fake):
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "late"}
    )
    await wait_done(connected, r.json()["id"])
    c = await connected.post(f"/api/generations/{r.json()['id']}/cancel")
    assert c.status_code == 409


async def test_photos_up_to_20_mb(connected: Api, fake):
    """Photos up to 20 MB are accepted and stored as they are; a hosted provider gets at most
    10 MB (a high-quality JPEG of a larger photo); above 20 MB is refused."""
    import io

    from PIL import Image as PILImage

    noise = PILImage.frombytes("RGB", (2400, 2000), os.urandom(2400 * 2000 * 3))
    out = io.BytesIO()
    noise.save(out, format="PNG", compress_level=0)
    big = out.getvalue()
    assert 10 * 1024 * 1024 < len(big) < 20 * 1024 * 1024
    up = await connected.post(
        "/api/reference-images", files={"file": ("big.png", big, "image/png")}
    )
    assert up.status_code == 201, up.text
    assert up.json()["size_bytes"] == len(big) and up.json()["width"] == 2400
    r = await connected.post(
        "/api/generations",
        json={
            "model": "seedream",
            "provider": "wavespeed",
            "prompt": "make it a watercolor",
            "reference_image_ids": [up.json()["id"]],
        },
    )
    assert (await wait_done(connected, r.json()["id"]))["status"] == "completed"
    assert fake.upload_sizes and fake.upload_sizes[-1] <= 10 * 1024 * 1024 + 4096
    too_big = big + b"\0" * (21 * 1024 * 1024 - len(big))
    refused = await connected.post(
        "/api/reference-images", files={"file": ("huge.png", too_big, "image/png")}
    )
    assert refused.status_code == 413 and "20 MB" in refused.json()["error"]["message"]


async def test_reference_upload_validation(user: Api):
    bad = await user.post(
        "/api/reference-images", files={"file": ("x.png", b"not an image", "image/png")}
    )
    assert bad.status_code == 415
    small = await user.post(
        "/api/reference-images", files={"file": ("x.png", png_bytes(20, 20), "image/png")}
    )
    assert small.status_code == 400


async def test_settings_roundtrip(user: Api):
    r = await user.patch("/api/settings", json={"theme": "light", "language": "sk"})
    assert r.json()["theme"] == "light"
    assert (await user.get("/api/settings")).json()["language"] == "sk"
    bad = await user.patch("/api/settings", json={"theme": "neon"})
    assert bad.status_code == 422


async def test_prompt_history(connected: Api, fake):
    for prompt in ("one", "two", "one"):
        r = await connected.post(
            "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": prompt}
        )
        await wait_done(connected, r.json()["id"])
    items = (await connected.get("/api/prompts/history")).json()["items"]
    assert [i["prompt"] for i in items] == ["one", "two"]


async def test_prices_and_estimate_before_a_run(connected: Api, fake):
    """WaveSpeed's price API gives the list price per model and an estimate per request;
    Atlas Cloud has none (price unknown, link to its pricing page). Nothing is generated."""
    listing = (await connected.get("/api/prices")).json()
    ws = listing["providers"]["wavespeed"]
    assert ws["supported"] is True and ws["error"] is None
    wan = next(p for p in ws["prices"] if p["model"] == "wan-2.7" and p["mode"] == "text-to-image")
    assert wan["price"] == fake.list_price
    atlas = listing["providers"]["atlas"]
    assert atlas["supported"] is False and atlas["prices"] == [] and atlas["pricing_url"]

    body = {"model": "wan-2.7", "provider": "wavespeed", "prompt": "a lighthouse", "count": 2}
    est = (await connected.post("/api/generations/estimate", json=body)).json()
    assert est["source"] == "estimate" and est["currency"] == "USD"
    assert est["total"] == round(fake.unit_price * 2, 4)
    assert est["per_image"] == fake.unit_price and est["images"] == 2
    # The real inputs are priced (here the size), and nothing was submitted.
    assert fake.pricing_calls[-1]["model_id"] == "alibaba/wan-2.7/text-to-image"
    assert "size" in fake.pricing_calls[-1]["inputs"]
    assert fake.submits == []

    # Without an estimate, the list price per image is used.
    fake.pricing_fails = True
    est = (await connected.post("/api/generations/estimate", json=body)).json()
    assert est["source"] == "list_price" and est["total"] == round(fake.list_price * 2, 4)
    fake.pricing_fails = False

    # Atlas: no price, but the request is still checked.
    est = await connected.post("/api/generations/estimate", json={**body, "provider": "atlas"})
    assert est.status_code == 200
    assert est.json()["total"] is None and est.json()["supported"] is False
    # The estimate runs every check of a real submit.
    refused = await connected.post(
        "/api/generations/estimate",
        json={**body, "model": "nano-banana-pro", "prompt": "a nude figure"},
    )
    assert refused.status_code == 422
    adult_off = await connected.post(
        "/api/generations/estimate", json={**body, "prompt": "artistic nude, adult woman"}
    )
    assert adult_off.json()["error"]["code"] == "adult_off"

    # Repeating a run is priced the same way.
    gen = await wait_done(
        connected, (await connected.post("/api/generations", json=body)).json()["id"]
    )
    again = await connected.post(f"/api/generations/{gen['id']}/estimate", json={"count": 1})
    assert again.status_code == 200, again.text
    assert again.json()["total"] == fake.unit_price and again.json()["images"] == 1
    # Only the run's own two requests (one image each) were submitted, never an estimate.
    assert len(fake.submits) == 2


async def test_wan3_videos(connected: Api, fake):
    """Wan 3.0 makes videos: from a prompt, or from a photo (first frame) and a second photo
    (last frame). The video is stored as delivered and listed with the images."""
    body = {
        "model": "wan-3.0",
        "provider": "wavespeed",
        "prompt": "a lighthouse in a storm, slow drone shot",
        "settings": {
            "aspect_ratio": "9:16",
            "resolution": "1080p",
            "duration": 12,
            "generate_audio": False,
        },
    }
    est = (await connected.post("/api/generations/estimate", json=body)).json()
    assert est["total"] is not None and fake.pricing_calls[-1]["inputs"]["duration"] == 12
    r = await connected.post("/api/generations", json=body)
    assert r.status_code == 201, r.text
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "completed" and gen["mode"] == "text-to-video", gen
    sent = fake.submits[-1]
    assert sent["model"] == "alibaba/wan-3.0/text-to-video"
    assert sent["body"] == {
        "prompt": body["prompt"],
        "aspect_ratio": "9:16",
        "resolution": "1080p",
        "duration": 12,
        "generate_audio": False,
        "enable_prompt_expansion": False,
    }
    video = gen["images"][0]
    assert video["media"] == "video" and video["mime_type"] == "video/mp4"
    assert (video["width"], video["height"]) == (320, 180)  # read from the MP4 itself
    file = await connected.get(video["url"])
    assert file.status_code == 200 and file.content == TEST_VIDEO
    assert file.headers["content-type"] == "video/mp4"
    assert (await connected.get(video["thumb_url"])).headers["content-type"] == "video/mp4"
    listed = (await connected.get("/api/images", params={"media": "video"})).json()
    assert [i["id"] for i in listed["items"]] == [video["id"]]
    photos = (await connected.get("/api/images", params={"media": "image"})).json()
    assert video["id"] not in [i["id"] for i in photos["items"]]

    # From photos: the first is the first frame, the second the last frame.
    refs = [
        (
            await connected.post(
                "/api/reference-images",
                files={"file": (f"{n}.png", png_bytes(300 + n, 400), "image/png")},
            )
        ).json()["id"]
        for n in (1, 2)
    ]
    r = await connected.post(
        "/api/generations",
        json={**body, "model": "wan-3.0-prime", "reference_image_ids": refs, "settings": {}},
    )
    assert r.status_code == 201, r.text
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "completed" and gen["mode"] == "image-to-video"
    sent = fake.submits[-1]
    assert sent["model"] == "alibaba/wan-3.0-prime/image-to-video"
    assert (
        sent["body"]["image"] == fake.uploads[-2] and sent["body"]["last_image"] == fake.uploads[-1]
    )
    assert "aspect_ratio" not in sent["body"] and "images" not in sent["body"]
    assert gen["request_payload"]["image"] == fake.uploads[-2]

    # SFW only, WaveSpeed only, 2-30 seconds.
    refused = await connected.post(
        "/api/generations", json={**body, "prompt": "nude woman on a beach"}
    )
    assert refused.status_code == 422
    atlas = await connected.post("/api/generations", json={**body, "provider": "atlas"})
    assert atlas.status_code == 400
    long = await connected.post("/api/generations", json={**body, "settings": {"duration": 31}})
    assert long.status_code == 400

    # A video is never used as a photo (references, AI models, LoRA photos, Krea 2).
    from app.services import references

    async with SessionLocal() as db:
        row = await db.get(Image, uuid.UUID(video["id"]))
        with pytest.raises(references.ReferenceRejected) as err:
            await references.from_library_file(db, row.user_id, row.file_path, "v.mp4")
    assert err.value.code == "video_not_photo"
    # Deleting it removes the file.
    async with SessionLocal() as db:
        path = (await db.get(Image, uuid.UUID(video["id"]))).file_path
    assert (await connected.delete(f"/api/images/{video['id']}")).status_code == 200
    from app.services import storage

    assert not storage.absolute(path).exists()


def test_mp4_size_and_expected_size():
    from app.services import storage

    assert storage.mp4_size(TEST_VIDEO) == (320, 180)
    assert storage.video_kind(TEST_VIDEO) == ("video/mp4", "mp4")
    assert storage.video_kind(png_bytes(10, 10)) is None
    assert storage.mp4_size(b"\0\0\0\x08moov") is None
    assert storage.expected_size({"resolution": "1080p", "aspect_ratio": "9:16"}) == (1080, 1920)
    assert storage.expected_size({"resolution": "480p", "aspect_ratio": "16:9"}) == (854, 480)


async def test_animate_uses_a_generated_image_as_first_frame(connected: Api, fake):
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "a cat"}
    )
    image = (await wait_done(connected, r.json()["id"]))["images"][0]
    ref = await connected.post("/api/reference-images/from-image", json={"image_id": image["id"]})
    assert ref.status_code == 201, ref.text
    video = await connected.post(
        "/api/generations",
        json={
            "model": "wan-3.0",
            "provider": "wavespeed",
            "prompt": "the cat yawns",
            "reference_image_ids": [ref.json()["id"]],
        },
    )
    assert (await wait_done(connected, video.json()["id"]))["mode"] == "image-to-video"
    other = await make_user("animate-other")
    assert (
        await other.post("/api/reference-images/from-image", json={"image_id": image["id"]})
    ).status_code == 404
    await other.client.aclose()


async def test_export_for_social_media(connected: Api, fake):
    """Cropped to the format, re-encoded without metadata, optional watermark; no videos."""
    import io
    import zipfile

    from PIL import Image as PILImage

    r = await connected.post(
        "/api/generations",
        json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "a cat", "count": 2},
    )
    ids = [i["id"] for i in (await wait_done(connected, r.json()["id"]))["images"]]
    out = await connected.get(
        "/api/images/export",
        params={"ids": ",".join(ids), "format": "story", "watermark": "@lena"},
    )
    assert out.status_code == 200 and out.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(out.content)) as zf:
        names = zf.namelist()
        assert len(names) == 2 and all(n.endswith("-story.jpg") for n in names)
        with PILImage.open(io.BytesIO(zf.read(names[0]))) as img:
            assert img.size == (1080, 1920) and img.format == "JPEG"
            assert not img.getexif() and "exif" not in img.info
    square = await connected.get("/api/images/export", params={"ids": ids[0], "format": "square"})
    with zipfile.ZipFile(io.BytesIO(square.content)) as zf:
        first = zf.read(zf.namelist()[0])
    with PILImage.open(io.BytesIO(first)) as img:
        assert img.size == (1080, 1080)
    bad = await connected.get("/api/images/export", params={"ids": ids[0], "format": "tall"})
    assert bad.status_code == 422
    other = await make_user("export-other")
    assert (await other.get("/api/images/export", params={"ids": ids[0]})).status_code == 404
    await other.client.aclose()
