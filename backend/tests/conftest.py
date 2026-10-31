"""Test setup: a separate PostgreSQL database, a temporary storage folder and fake providers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

TEST_DB = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://localhost:5432/ai_image_studio_test"
)
os.environ["DATABASE_URL"] = TEST_DB
os.environ["APP_SECRET_KEY"] = "t" * 64
os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="studio-test-")
os.environ["WAVESPEED_BASE_URL"] = "https://api.wavespeed.fake"
os.environ["ATLAS_BASE_URL"] = "https://api.atlas.fake"

import httpx  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from app import providers  # noqa: E402
from app.main import app  # noqa: E402
from app.providers.atlas import AtlasAdapter  # noqa: E402
from app.providers.wavespeed import WaveSpeedAdapter  # noqa: E402
from app.services.worker import worker  # noqa: E402
from tests.fakes import VALID_KEY, FakeProviders  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent


def _reset_database() -> None:
    sync_url = TEST_DB
    engine = create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    engine.dispose()
    subprocess.run(["alembic", "upgrade", "head"], cwd=BACKEND, check=True, env=os.environ.copy())


@pytest.fixture(scope="session", autouse=True)
async def _database() -> AsyncIterator[None]:
    _reset_database()
    worker.poll_initial = 0.01
    worker.poll_max = 0.02
    await worker.start()
    yield
    await worker.stop()


@pytest.fixture
def fake() -> FakeProviders:
    f = FakeProviders()
    transport = f.transport()
    providers.ADAPTERS["wavespeed"] = WaveSpeedAdapter(transport=transport)
    providers.ADAPTERS["atlas"] = AtlasAdapter(transport=transport)
    worker.download_transport = transport
    return f


class Api:
    """A signed-in browser: keeps the session cookie and sends the client header."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def request(self, method: str, url: str, **kwargs):
        headers = {"x-studio-client": "web", **kwargs.pop("headers", {})}
        return await self.client.request(method, url, headers=headers, **kwargs)

    async def get(self, url: str, **kw):
        return await self.request("GET", url, **kw)

    async def post(self, url: str, **kw):
        return await self.request("POST", url, **kw)

    async def put(self, url: str, **kw):
        return await self.request("PUT", url, **kw)

    async def patch(self, url: str, **kw):
        return await self.request("PATCH", url, **kw)

    async def delete(self, url: str, **kw):
        return await self.request("DELETE", url, **kw)


_counter = iter(range(1, 10_000))


async def make_user(name: str = "Tester") -> Api:
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    api = Api(client)
    email = f"user{next(_counter)}@example.com"
    r = await api.post(
        "/api/auth/register", json={"name": name, "email": email, "password": "password123"}
    )
    assert r.status_code == 201, r.text
    return api


@pytest.fixture
async def user(fake) -> AsyncIterator[Api]:
    api = await make_user()
    yield api
    await api.client.aclose()


@pytest.fixture
async def connected(user: Api) -> Api:
    for provider in ("wavespeed", "atlas"):
        r = await user.put(f"/api/integrations/{provider}", json={"api_key": VALID_KEY})
        assert r.status_code == 200, r.text
    return user


async def wait_done(api: Api, gen_id: str) -> dict:
    await worker.drain()
    r = await api.get(f"/api/generations/{gen_id}")
    assert r.status_code == 200
    return r.json()
