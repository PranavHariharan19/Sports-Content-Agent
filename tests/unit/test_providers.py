import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from llm.base import LLMAuthError, LLMProviderError, LLMRateLimitError
from llm.gemini_provider import GeminiProvider
from llm.groq_provider import GroqProvider
from llm.ollama_provider import OllamaProvider
from llm.openrouter_provider import OpenRouterProvider
from schemas.llm import LLMRequest


def test_gemini_provider_availability():
    """Verify GeminiProvider availability check reflects API key presence."""
    p_no_key = GeminiProvider(api_key=None)
    assert p_no_key.is_available() is False

    p_with_key = GeminiProvider(api_key="valid-dummy-key")
    assert p_with_key.is_available() is True


@pytest.mark.asyncio
async def test_gemini_provider_unconfigured_raises_auth_error():
    """Verify calling an unconfigured GeminiProvider raises LLMAuthError."""
    provider = GeminiProvider(api_key=None)
    req = LLMRequest(prompt="Test prompt")
    with pytest.raises(LLMAuthError):
        await provider.generate(req)


@pytest.mark.asyncio
async def test_openai_compatible_provider_success():
    """Verify OpenAICompatibleProvider sends correct request format and parses response."""
    provider = GroqProvider(api_key="dummy_groq_key")
    req = LLMRequest(
        prompt="Tell me about F1 qualifying",
        system_prompt="You are a factual sports journalist",
        temperature=0.0,
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Qualifying sets the grid for Sunday's race."}}
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        res = await provider.generate(req)

        assert res.text == "Qualifying sets the grid for Sunday's race."
        assert res.provider == "groq"
        assert res.usage.total_tokens == 20
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["messages"][0]["role"] == "system"
        assert call_kwargs["json"]["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_openai_compatible_provider_rate_limit():
    """Verify HTTP 429 is mapped to LLMRateLimitError."""
    provider = OpenRouterProvider(api_key="dummy_openrouter_key")
    req = LLMRequest(prompt="Test prompt")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit reached"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(LLMRateLimitError):
            await provider.generate(req)


@pytest.mark.asyncio
async def test_openai_compatible_provider_auth_error():
    """Verify HTTP 401 is mapped to LLMAuthError."""
    provider = GroqProvider(api_key="invalid_key")
    req = LLMRequest(prompt="Test prompt")

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(LLMAuthError):
            await provider.generate(req)
