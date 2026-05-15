from __future__ import annotations

from tangram.client import _to_openai_message
from tangram.config import ModelConfig


def test_model_config_has_no_default_thinking_budget(monkeypatch):
    monkeypatch.delenv("TANGRAM_PROVIDER", raising=False)
    monkeypatch.delenv("TANGRAM_MODEL", raising=False)
    config = ModelConfig()
    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-4-5"
    assert config.thinking_budget_tokens is None
    assert config.thinking is None


def test_openai_provider_defaults_to_openai_model(monkeypatch):
    monkeypatch.delenv("TANGRAM_MODEL", raising=False)
    config = ModelConfig(provider="openai")
    assert config.model == "gpt-5.2"
    assert config.reasoning is None


def test_openai_message_conversion_uses_data_urls_for_images():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc123",
                },
            },
        ],
    }

    converted = _to_openai_message(message)

    assert converted["role"] == "user"
    assert converted["content"][0] == {"type": "input_text", "text": "Look at this."}
    assert converted["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,abc123",
        "detail": "auto",
    }


def test_openai_assistant_history_converts_to_plain_text():
    converted = _to_openai_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Previous turn."}],
        }
    )

    assert converted == {"role": "assistant", "content": "Previous turn."}
