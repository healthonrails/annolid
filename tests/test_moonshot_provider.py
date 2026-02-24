from annolid.core.agent.providers.registry import find_by_model
from annolid.core.agent.providers.litellm_provider import LiteLLMProvider


def test_moonshot_registry_resolution():
    spec = find_by_model("kimi-k2.5")
    assert spec is not None
    assert spec.name == "moonshot"


def test_moonshotai_registry_resolution():
    spec = find_by_model("moonshotai/kimi-k2.5")
    assert spec is not None
    assert spec.name == "nvidia"


def test_moonshot_model_resolution():
    provider = LiteLLMProvider(default_model="kimi-k2.5")
    # Should resolve to native moonshot/
    resolved = provider._resolve_model("kimi-k2.5")
    assert resolved == "moonshot/kimi-k2.5"


def test_nvidia_moonshot_resolution():
    provider = LiteLLMProvider(default_model="kimi-k2.5")
    # Should resolve to nvidia_nim/ via explicit prefix and preserve moonshotai
    resolved = provider._resolve_model("nvidia/moonshotai/kimi-k2.5")
    assert resolved == "nvidia_nim/moonshotai/kimi-k2.5"

    # Should resolve to nvidia_nim/ via keyword moonshotai
    resolved2 = provider._resolve_model("moonshotai/kimi-k2.5")
    assert resolved2 == "nvidia_nim/moonshotai/kimi-k2.5"
