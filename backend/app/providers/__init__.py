"""Provider adapters. Tests replace entries of `ADAPTERS` with fakes."""

from app.providers.atlas import AtlasAdapter
from app.providers.base import ProviderAdapter, ProviderError
from app.providers.wavespeed import WaveSpeedAdapter

ADAPTERS: dict[str, ProviderAdapter] = {
    "wavespeed": WaveSpeedAdapter(),
    "atlas": AtlasAdapter(),
}


def get_adapter(provider: str) -> ProviderAdapter:
    try:
        return ADAPTERS[provider]
    except KeyError as exc:
        raise ProviderError("unsupported_model", provider, "Unknown provider") from exc
