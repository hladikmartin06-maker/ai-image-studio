"""Prompt snippets: per user, for all AI models or one, checked like AI model profiles."""

from __future__ import annotations

from tests.conftest import Api, make_user
from tests.test_personas import LENA


async def test_snippets(user: Api):
    r = await user.post("/api/personas", json=LENA)
    assert r.status_code == 201, r.text
    lena = r.json()
    a = await user.post(
        "/api/snippets", json={"name": "Golden hour", "text": "warm golden hour light"}
    )
    assert a.status_code == 201, a.text
    b = await user.post(
        "/api/snippets",
        json={"name": "Lena outfit", "text": "black blazer, gold hoops", "persona_id": lena["id"]},
    )
    assert b.status_code == 201 and b.json()["persona_id"] == lena["id"]
    items = (await user.get("/api/snippets")).json()["items"]
    assert [i["name"] for i in items] == ["Golden hour", "Lena outfit"]
    edited = await user.patch(f"/api/snippets/{a.json()['id']}", json={"text": "soft dusk light"})
    assert edited.json()["text"] == "soft dusk light"
    refused = await user.post("/api/snippets", json={"name": "x", "text": "a teen girl"})
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "content_blocked"
    other = await make_user("snippets-other")
    assert (await other.get("/api/snippets")).json()["items"] == []
    assert (await other.delete(f"/api/snippets/{a.json()['id']}")).status_code == 404
    assert (
        await other.post("/api/snippets", json={"name": "x", "text": "y", "persona_id": lena["id"]})
    ).status_code == 404
    await other.client.aclose()
    # Deleting the AI model deletes its snippets.
    assert (await user.delete(f"/api/personas/{lena['id']}")).status_code == 200
    assert [i["name"] for i in (await user.get("/api/snippets")).json()["items"]] == ["Golden hour"]
    assert (await user.delete(f"/api/snippets/{a.json()['id']}")).status_code == 200
