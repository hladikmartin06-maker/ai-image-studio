"""AI models (saved characters): profiles, reference collections and use in generations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Persona, ReferenceImage
from app.services import personas, references, storage
from app.services.safety import mentions_minors
from tests.conftest import Api, make_user, wait_done
from tests.fakes import png_bytes

LENA = {
    "name": "Lena",
    "description": "confident, warm smile",
    "notes": "travel series",
    "appearance": {
        "age": 27,
        "gender": "woman",
        "hair_color": "auburn",
        "hairstyle": "long loose waves",
        "eyes": "green",
        "face": "freckles, high cheekbones",
        "body": "slim, athletic",
    },
    "adult_confirmed": True,
}


async def _persona(api: Api, **overrides) -> dict:
    r = await api.post("/api/personas", json={**LENA, **overrides})
    assert r.status_code == 201, r.text
    return r.json()


async def _add_images(api: Api, persona_id: str, n: int) -> dict:
    body: dict = {}
    for i in range(n):
        files = {"file": (f"p{i}.png", png_bytes(100 + i, 80, (i * 30, 40, 90)), "image/png")}
        r = await api.post(f"/api/personas/{persona_id}/images", files=files)
        assert r.status_code == 201, r.text
        body = r.json()["persona"]
    return body


def test_character_text_and_identity():
    p = Persona(name="Lena", description="calm look.", appearance=LENA["appearance"])
    text = personas.character_text(p)
    assert text.startswith("Character: adult woman, 27 years old; calm look; ")
    assert "hairstyle: long loose waves" in text and "facial features: freckles" in text
    assert personas.character_text(Persona(name="x", description="", appearance={})) == ""
    assert "the reference image:" in personas.identity_text(1)
    assert personas.identity_text(0) == ""
    composed = personas.compose_prompt("on a beach", p, 2)
    assert composed.startswith("on a beach\n\nCharacter: ")
    assert composed.endswith("and body proportions.")


def test_profile_words_for_minors():
    assert mentions_minors("a teen girl")
    assert mentions_minors("16 year old")
    assert not mentions_minors("baby blue eyes, kid gloves")
    assert not mentions_minors("adult woman")


async def test_create_requires_adult_and_blocks_minor_terms(user: Api):
    r = await user.post("/api/personas", json={**LENA, "adult_confirmed": False})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "adult_confirmation_required"
    young = {**LENA, "appearance": {**LENA["appearance"], "age": 16}}
    assert (await user.post("/api/personas", json=young)).status_code == 422
    teen = await user.post("/api/personas", json={**LENA, "description": "a teenager"})
    assert teen.status_code == 422
    assert teen.json()["error"]["code"] == "content_blocked"
    ok = await _persona(user, appearance={**LENA["appearance"], "eyes": "baby blue"})
    patched = await user.patch(f"/api/personas/{ok['id']}", json={"name": "schoolgirl Lena"})
    assert patched.status_code == 422


async def test_profile_and_collection(user: Api):
    lena = await _persona(user)
    assert lena["prompt_text"].startswith("Character: adult woman, 27 years old")
    assert lena["images"] == [] and lena["primary_image"] is None
    lena = await _add_images(user, lena["id"], 3)
    ids = [i["id"] for i in lena["images"]]
    assert lena["image_count"] == 3 and lena["primary_image"]["id"] == ids[0]
    # The library shows these photos and can download them as files.
    url = lena["images"][0]["url"]
    shown = await user.get(url)
    assert shown.status_code == 200 and "content-disposition" not in shown.headers
    saved = await user.get(f"{url}?download=1")
    assert saved.headers["content-disposition"].startswith("attachment")

    # The same file again is not stored twice.
    files = {"file": ("again.png", png_bytes(100, 80, (0, 40, 90)), "image/png")}
    again = await user.post(f"/api/personas/{lena['id']}/images", files=files)
    assert again.json()["added"] == 0 and again.json()["persona"]["image_count"] == 3
    async with SessionLocal() as db:
        assert len((await db.scalars(select(ReferenceImage.id))).all()) >= 3

    # Primary image and order.
    r = await user.patch(
        f"/api/personas/{lena['id']}",
        json={"cover_reference_id": ids[2], "image_order": [ids[1], ids[0]]},
    )
    body = r.json()
    assert [i["id"] for i in body["images"]] == [ids[2], ids[1], ids[0]]
    assert body["images"][0]["is_primary"] is True
    foreign = await user.patch(
        f"/api/personas/{lena['id']}", json={"cover_reference_id": str(uuid.uuid4())}
    )
    assert foreign.status_code == 400

    # Edit, search, list.
    edited = await user.patch(
        f"/api/personas/{lena['id']}",
        json={"description": "serene", "appearance": {"age": 30, "hairstyle": "bob"}},
    )
    assert edited.json()["appearance"]["age"] == 30
    assert edited.json()["appearance"]["gender"] == ""
    await _persona(user, name="Mara")
    listing = (await user.get("/api/personas")).json()
    assert listing["total"] == 2
    assert (await user.get("/api/personas", params={"q": "mar"})).json()["total"] == 1

    # Removing an image deletes its file (nobody else uses it).
    removed = await user.delete(f"/api/personas/{lena['id']}/images/{ids[1]}")
    assert removed.status_code == 200
    assert removed.json()["image_count"] == 2
    async with SessionLocal() as db:
        assert await db.get(ReferenceImage, uuid.UUID(ids[1])) is None
    # The reference of an AI model cannot be deleted from the generic uploads route.
    blocked = await user.delete(f"/api/reference-images/{ids[0]}")
    assert blocked.status_code == 409


async def test_collection_holds_more_than_24_images(user: Api):
    big = await _persona(user, name="Big collection")
    body: dict = {}
    for i in range(26):
        color = (i * 9, 20, 200 - i * 5)
        files = {"file": (f"m{i}.png", png_bytes(80 + i, 70, color), "image/png")}
        r = await user.post(f"/api/personas/{big['id']}/images", files=files)
        assert r.status_code == 201, r.text
        body = r.json()["persona"]
    assert body["image_count"] == 26
    assert body["max_images"] == personas.MAX_PERSONA_IMAGES == 200


async def test_generation_attaches_references_within_the_endpoint_limit(connected: Api, fake):
    lena = await _add_images(connected, (await _persona(connected))["id"], 5)
    order = [i["id"] for i in lena["images"]]
    body = {
        "model": "wan-2.7",
        "provider": "wavespeed",
        "prompt": "walking in Lisbon at sunset",
        "persona": {"id": lena["id"]},
    }
    r = await connected.post("/api/generations", json=body)
    assert r.status_code == 201, r.text
    gen = await wait_done(connected, r.json()["id"])
    assert gen["status"] == "completed"
    assert gen["mode"] == "image-edit"
    # Wan 2.7 edit on WaveSpeed takes up to 3: the primary image first, then the order.
    assert gen["reference_image_ids"] == order[:3]
    assert gen["persona"]["name"] == "Lena"
    assert gen["persona"]["references_skipped"] == 2
    assert len(fake.uploads) == 3
    sent = fake.submits[0]
    assert sent["model"] == "alibaba/wan-2.7/image-edit"
    assert sent["body"]["prompt"].startswith(
        "walking in Lisbon at sunset\n\nCharacter: adult woman"
    )
    assert "Keep the same person as in the reference images" in sent["body"]["prompt"]
    assert gen["prompt"] == "walking in Lisbon at sunset"
    assert gen["final_prompt"] == sent["body"]["prompt"]

    # A second run reuses the uploads; Seedream on WaveSpeed takes all five.
    r = await connected.post("/api/generations", json={**body, "model": "seedream", "settings": {}})
    gen2 = await wait_done(connected, r.json()["id"])
    assert gen2["status"] == "completed", gen2
    assert len(gen2["reference_image_ids"]) == 5
    assert len(fake.uploads) == 5

    # Explicit selection over the limit is refused, not cut silently.
    pick_all = {"id": lena["id"], "reference_image_ids": order}
    too_many = await connected.post("/api/generations", json={**body, "persona": pick_all})
    assert too_many.status_code == 400
    # With the user's own reference, fewer of the AI model's images fit.
    files = {"file": ("outfit.png", png_bytes(300, 200), "image/png")}
    own = (await connected.post("/api/reference-images", files=files)).json()
    r = await connected.post("/api/generations", json={**body, "reference_image_ids": [own["id"]]})
    gen3 = await wait_done(connected, r.json()["id"])
    assert gen3["reference_image_ids"] == [*order[:2], own["id"]]

    # The library and history can be filtered by AI model.
    images = (await connected.get("/api/images", params={"persona": lena["id"]})).json()
    assert images["total"] == 3
    assert images["items"][0]["persona_name"] == "Lena"
    history = (await connected.get("/api/generations", params={"persona": lena["id"]})).json()
    assert history["total"] == 3
    listed = (await connected.get(f"/api/personas/{lena['id']}")).json()
    assert listed["generation_count"] == 3 and listed["last_used_at"]


async def test_references_and_description_switches(connected: Api, fake):
    lena = await _add_images(connected, (await _persona(connected))["id"], 2)
    base = {"model": "nano-banana-pro", "provider": "atlas", "prompt": "portrait, studio light"}
    r = await connected.post(
        "/api/generations",
        json={**base, "persona": {"id": lena["id"], "use_references": False}},
    )
    gen = await wait_done(connected, r.json()["id"])
    assert gen["mode"] == "text-to-image" and gen["reference_image_ids"] == []
    assert fake.uploads == []
    prompt = fake.submits[-1]["body"]["prompt"]
    assert "Character: adult woman" in prompt and "Keep the same person" not in prompt

    r = await connected.post(
        "/api/generations",
        json={**base, "persona": {"id": lena["id"], "use_description": False}},
    )
    gen = await wait_done(connected, r.json()["id"])
    assert gen["mode"] == "image-edit" and len(gen["reference_image_ids"]) == 2
    assert fake.submits[-1]["body"]["prompt"] == "portrait, studio light"
    assert gen["final_prompt"] is None

    # A profile without references and text adds nothing.
    empty = await _persona(connected, name="Blank", description="", appearance={})
    r = await connected.post("/api/generations", json={**base, "persona": {"id": empty["id"]}})
    gen = await wait_done(connected, r.json()["id"])
    assert fake.submits[-1]["body"]["prompt"] == "portrait, studio light"
    assert gen["persona"]["name"] == "Blank"


async def test_description_follows_the_content_rules(connected: Api, fake):
    bold = await _persona(connected, description="sexy, lingerie")
    sfw = await connected.post(
        "/api/generations",
        json={
            "model": "nano-banana-pro",
            "provider": "wavespeed",
            "prompt": "a portrait",
            "persona": {"id": bold["id"]},
        },
    )
    assert sfw.status_code == 422
    assert "AI model “Lena”" in sfw.json()["error"]["message"]
    body = {
        "model": "wan-2.7-pro",
        "provider": "wavespeed",
        "prompt": "a portrait",
        "persona": {"id": bold["id"]},
    }
    off = await connected.post("/api/generations", json=body)
    assert off.status_code == 422 and off.json()["error"]["code"] == "adult_off"
    assert "AI model “Lena”" in off.json()["error"]["message"]
    adult = await connected.post("/api/generations", json={**body, "adult": True})
    assert adult.status_code == 201
    gen = await wait_done(connected, adult.json()["id"])
    assert gen["settings"]["adult"] is True
    assert all(img["nsfw"] is True for img in gen["images"])


async def test_regenerate_repeats_the_request_as_sent(connected: Api, fake):
    lena = await _add_images(connected, (await _persona(connected))["id"], 1)
    r = await connected.post(
        "/api/generations",
        json={
            "model": "wan-2.7",
            "provider": "atlas",
            "prompt": "reading in a cafe",
            "persona": {"id": lena["id"]},
        },
    )
    gen = await wait_done(connected, r.json()["id"])
    await connected.patch(f"/api/personas/{lena['id']}", json={"description": "changed"})
    again = await connected.post(f"/api/generations/{gen['id']}/regenerate", json={})
    regen = await wait_done(connected, again.json()["id"])
    assert regen["final_prompt"] == gen["final_prompt"]
    assert regen["persona"]["id"] == lena["id"]
    assert fake.submits[-1]["body"]["prompt"] == fake.submits[0]["body"]["prompt"]
    assert len(fake.uploads) == 1


async def test_duplicate_delete_and_library_images(connected: Api, fake):
    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "a tree"}
    )
    gen = await wait_done(connected, r.json()["id"])
    image_id = gen["images"][0]["id"]
    lena = await _add_images(connected, (await _persona(connected))["id"], 1)
    added = await connected.post(
        f"/api/personas/{lena['id']}/images/add", json={"library_image_ids": [image_id]}
    )
    assert added.status_code == 200 and added.json()["added"] == 1
    again = await connected.post(
        f"/api/personas/{lena['id']}/images/add", json={"library_image_ids": [image_id]}
    )
    assert again.json()["added"] == 0
    lena = again.json()["persona"]
    ref_ids = [i["id"] for i in lena["images"]]

    copy = await connected.post(f"/api/personas/{lena['id']}/duplicate")
    assert copy.status_code == 201
    copy = copy.json()
    assert copy["name"] == "Lena (copy)"
    assert [i["id"] for i in copy["images"]] == ref_ids  # shared files, nothing copied

    # Deleting the original keeps the files the copy still uses.
    assert (await connected.delete(f"/api/personas/{lena['id']}")).status_code == 200
    async with SessionLocal() as db:
        rows = (
            await db.scalars(
                select(ReferenceImage).where(ReferenceImage.id.in_([uuid.UUID(i) for i in ref_ids]))
            )
        ).all()
        assert len(rows) == 2
        paths = [r.file_path for r in rows]
    assert all(storage.absolute(p).exists() for p in paths)
    # Deleting the copy removes them; the generated image stays in the library.
    assert (await connected.delete(f"/api/personas/{copy['id']}")).status_code == 200
    assert not any(storage.absolute(p).exists() for p in paths)
    assert (await connected.get(f"/api/images/{image_id}")).status_code == 200


async def test_cleanup_keeps_ai_model_references(connected: Api, fake):
    lena = await _add_images(connected, (await _persona(connected))["id"], 2)
    files = {"file": ("loose.png", png_bytes(120, 90, (9, 9, 9)), "image/png")}
    loose = (await connected.post("/api/reference-images", files=files)).json()
    info = (await connected.get("/api/storage")).json()
    assert info["reference_images"] == 3 and info["persona_reference_images"] == 2
    r = await connected.post("/api/storage/cleanup", json={"action": "unused_references"})
    assert r.json()["removed"] == 1
    assert (await connected.get(f"/api/reference-images/{loose['id']}")).status_code == 404
    assert (await connected.get(f"/api/personas/{lena['id']}")).json()["image_count"] == 2


async def test_ai_models_are_private(connected: Api, fake):
    lena = await _add_images(connected, (await _persona(connected))["id"], 1)
    other = await make_user("Other")
    assert (await other.get(f"/api/personas/{lena['id']}")).status_code == 404
    assert (await other.get("/api/personas")).json()["total"] == 0
    assert (await other.patch(f"/api/personas/{lena['id']}", json={"name": "x"})).status_code == 404
    assert (await other.delete(f"/api/personas/{lena['id']}")).status_code == 404
    assert (await other.post(f"/api/personas/{lena['id']}/duplicate")).status_code == 404
    ref_id = lena["images"][0]["id"]
    assert (await other.get(f"/api/reference-images/{ref_id}/thumb")).status_code == 404
    await other.client.aclose()
    # Someone else's AI model cannot be used in a generation.
    stranger = await make_user("Stranger")
    for provider in ("wavespeed",):
        await stranger.put(f"/api/integrations/{provider}", json={"api_key": "test-key-1234567890"})
    r = await stranger.post(
        "/api/generations",
        json={
            "model": "wan-2.7",
            "provider": "wavespeed",
            "prompt": "x",
            "persona": {"id": lena["id"]},
        },
    )
    assert r.status_code == 404
    await stranger.client.aclose()


async def test_uploads_are_renewed_after_a_day(user: Api):
    files = {"file": ("r.png", png_bytes(200, 100), "image/png")}
    ref_id = (await user.post("/api/reference-images", files=files)).json()["id"]
    async with SessionLocal() as db:
        ref = await db.get(ReferenceImage, uuid.UUID(ref_id))
        assert ref is not None
        assert references.reusable_url(ref, "wavespeed") is None
        references.remember_upload(ref, "wavespeed", "https://cdn.fake/x.png")
        assert references.reusable_url(ref, "wavespeed") == "https://cdn.fake/x.png"
        later = datetime.now(UTC) + references.UPLOAD_REUSE + timedelta(minutes=1)
        assert references.reusable_url(ref, "wavespeed", later) is None
        assert references.reusable_url(ref, "atlas") is None


def test_assistant_context_mentions_the_ai_model():
    from app.ai.context import StudioContext, format_context

    ctx = StudioContext(
        page="generate",
        ai_model="Lena",
        ai_model_text="Character: adult woman.",
        ai_model_references=2,
    )
    text = format_context(ctx, [])
    assert 'AI model (saved character): "Lena", 2 of its reference images attached' in text
    assert "added to the prompt: Character: adult woman." in text
