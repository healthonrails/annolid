from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    keywords: Tuple[str, ...]
    env_key: str
    default_api_base: str = ""
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""
    is_gateway: bool = False
    is_local: bool = False
    litellm_prefix: str = ""
    skip_prefixes: Tuple[str, ...] = ()
    env_extras: Tuple[Tuple[str, str], ...] = ()
    strip_model_prefix: bool = False
    model_overrides: Tuple[Tuple[str, dict[str, Any]], ...] = ()


PROVIDERS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        env_key="OPENROUTER_API_KEY",
        default_api_base="https://openrouter.ai/api/v1",
        detect_by_key_prefix="sk-or-",
        detect_by_base_keyword="openrouter",
        is_gateway=True,
        litellm_prefix="openrouter",
    ),
    ProviderSpec(
        name="aihubmix",
        keywords=("aihubmix",),
        env_key="OPENAI_API_KEY",
        default_api_base="https://aihubmix.com/v1",
        detect_by_base_keyword="aihubmix",
        is_gateway=True,
        litellm_prefix="openai",
        strip_model_prefix=True,
    ),
    ProviderSpec(
        name="vllm",
        keywords=("vllm",),
        env_key="HOSTED_VLLM_API_KEY",
        is_local=True,
        litellm_prefix="hosted_vllm",
    ),
    ProviderSpec(
        name="ollama",
        keywords=("ollama", "llama", "qwen"),
        env_key="OPENAI_API_KEY",
        default_api_base="http://localhost:11434/v1",
        detect_by_base_keyword="localhost:11434",
        is_local=True,
    ),
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt"),
        env_key="OPENAI_API_KEY",
        default_api_base="https://api.openai.com/v1",
    ),
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
    ),
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        env_key="DEEPSEEK_API_KEY",
        litellm_prefix="deepseek",
        skip_prefixes=("deepseek/",),
    ),
    ProviderSpec(
        name="gemini",
        keywords=("gemini",),
        env_key="GEMINI_API_KEY",
        litellm_prefix="gemini",
        skip_prefixes=("gemini/",),
    ),
    ProviderSpec(
        name="dashscope",
        keywords=("qwen", "dashscope"),
        env_key="DASHSCOPE_API_KEY",
        litellm_prefix="dashscope",
        skip_prefixes=("dashscope/", "openrouter/"),
    ),
    ProviderSpec(
        name="zhipu",
        keywords=("zhipu", "glm", "zai"),
        env_key="ZAI_API_KEY",
        litellm_prefix="zai",
        skip_prefixes=("zhipu/", "zai/", "openrouter/", "hosted_vllm/"),
        env_extras=(("ZHIPUAI_API_KEY", "{api_key}"),),
    ),
    ProviderSpec(
        name="nvidia",
        keywords=("nvidia", "moonshotai"),
        env_key="NVIDIA_API_KEY",
        default_api_base="https://integrate.api.nvidia.com/v1",
        detect_by_key_prefix="nvapi-",
        detect_by_base_keyword="nvidia",
        litellm_prefix="nvidia_nim",
        strip_model_prefix=True,
        skip_prefixes=("nvidia_nim/", "nvidia/"),
    ),
    ProviderSpec(
        name="moonshot",
        keywords=("moonshot", "kimi"),
        env_key="MOONSHOT_API_KEY",
        litellm_prefix="moonshot",
        strip_model_prefix=True,
        skip_prefixes=("moonshot/", "openrouter/"),
        model_overrides=(("kimi-k2.5", {"temperature": 1.0}),),
    ),
    ProviderSpec(
        name="minimax",
        keywords=("minimax",),
        env_key="MINIMAX_API_KEY",
        default_api_base="https://api.minimax.io/v1",
        litellm_prefix="minimax",
        skip_prefixes=("minimax/", "openrouter/"),
    ),
    ProviderSpec(
        name="groq",
        keywords=("groq",),
        env_key="GROQ_API_KEY",
        litellm_prefix="groq",
        skip_prefixes=("groq/",),
    ),
)


def find_by_name(name: str) -> Optional[ProviderSpec]:
    normalized = str(name or "").strip().lower()
    for spec in PROVIDERS:
        if spec.name == normalized:
            return spec
    return None


def find_by_model(model: str) -> Optional[ProviderSpec]:
    model_lower = str(model or "").lower()

    # Check for explicit provider/prefix first (e.g. nvidia/ or moonshot/)
    if "/" in model_lower:
        prefix = model_lower.split("/", 1)[0]
        for spec in PROVIDERS:
            if spec.name == prefix or spec.litellm_prefix == prefix:
                return spec

    for spec in PROVIDERS:
        if spec.is_gateway or spec.is_local:
            continue
        if any(keyword in model_lower for keyword in spec.keywords):
            return spec
    return None


def find_gateway(
    *,
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[ProviderSpec]:
    if provider_name:
        by_name = find_by_name(provider_name)
        if by_name and (by_name.is_gateway or by_name.is_local):
            return by_name
    if api_key:
        for spec in PROVIDERS:
            if spec.detect_by_key_prefix and str(api_key).startswith(
                spec.detect_by_key_prefix
            ):
                return spec
    if api_base:
        base = str(api_base).lower()
        for spec in PROVIDERS:
            if spec.detect_by_base_keyword and spec.detect_by_base_keyword in base:
                return spec
    return None
