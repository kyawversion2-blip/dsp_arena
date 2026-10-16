import asyncio
import json

import pytest

import ai_model_gateway as gateway


def test_provider_status_never_returns_key_material(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-not-for-output")
    status = gateway.provider_status()
    openai = next(p for p in status["providers"] if p["id"] == "openai")
    assert openai["configured"] is True
    assert "sk-test-secret-not-for-output" not in json.dumps(status)
    assert openai["key_env"] == "OPENAI_API_KEY"


def test_sanitize_rows_drops_nonfinite_and_unknown_fields():
    clean = gateway.sanitize_rows([
        {"group": "a", "impact_freq_hz": 1200, "laplace_sigma_fit": float("nan"), "free_text": "ignore me"},
        {"group": "Z", "hfsr": 0.5},
    ])
    assert clean == [
        {"row": 1, "group": "A", "impact_freq_hz": 1200.0},
        {"row": 2, "hfsr": 0.5},
    ]


def test_sanitize_rows_rejects_empty_and_too_many_rows():
    with pytest.raises(gateway.AIProviderError) as empty:
        gateway.sanitize_rows([])
    assert empty.value.status_code == 422
    with pytest.raises(gateway.AIProviderError) as many:
        gateway.sanitize_rows([{"impact_freq_hz": 1000}] * (gateway.MAX_ROWS + 1))
    assert many.value.status_code == 413


def test_unconfigured_provider_fails_without_network(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(gateway.AIProviderError, match="DEEPSEEK_API_KEY") as exc:
        asyncio.run(gateway.run_dsp_ai("deepseek", "deepseek-chat", "feature_analysis", [{"impact_freq_hz": 1000}]))
    assert exc.value.status_code == 503


def test_classification_hides_holdout_labels_and_reports_small_holdout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    def fake_call(provider, model, system_prompt, user_prompt):
        captured["provider"] = provider
        captured["prompt"] = json.loads(user_prompt)
        return json.dumps({"predictions": [
            {"target_id": "T1", "predicted_group": "A", "confidence": 0.8, "reason": "near reference"},
            {"target_id": "T2", "predicted_group": "B", "confidence": 0.7, "reason": "near reference"},
        ], "limitations": "tiny holdout"})

    monkeypatch.setattr(gateway, "_call_provider", fake_call)
    rows = [
        {"group": "A", "impact_freq_hz": 1000, "laplace_sigma_fit": 10},
        {"group": "A", "impact_freq_hz": 1010, "laplace_sigma_fit": 11},
        {"group": "B", "impact_freq_hz": 3000, "laplace_sigma_fit": 30},
        {"group": "B", "impact_freq_hz": 3010, "laplace_sigma_fit": 31},
    ]
    result = asyncio.run(gateway.run_dsp_ai("openai", "gpt-4.1-mini", "classify", rows))
    prompt = captured["prompt"]
    targets = prompt["target_rows_without_labels"]
    assert [r["target_id"] for r in targets] == ["T1", "T2"]
    assert all("group" not in r and "row" not in r for r in targets)
    assert all("row" not in r for r in prompt["labeled_reference_rows"])
    assert prompt["group_feature_summary_from_training_rows_only"]["groups"]["A"]["n"] == 1
    assert prompt["group_feature_summary_from_training_rows_only"]["groups"]["B"]["n"] == 1
    assert result["result"]["holdout_evaluation"] == {
        "n_evaluated": 2, "n_holdout": 2, "accuracy": 1.0,
        "note": "Accuracy applies only to this small in-dataset holdout, not to future/generalization performance.",
    }


def test_classification_requires_two_labeled_groups(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(gateway.AIProviderError, match="at least two groups"):
        asyncio.run(gateway.run_dsp_ai("openai", "gpt-4.1-mini", "classify", [
            {"group": "A", "impact_freq_hz": 1000},
            {"group": "A", "impact_freq_hz": 1100},
        ]))


def test_local_dotenv_loader_does_not_override_process_env_or_read_permissive_file(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    # Verify the key is never overwritten by local config.
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    original = gateway.Path
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\nDEEPSEEK_API_KEY=file-deepseek\n")
    env_file.chmod(0o600)
    class FakePath:
        def __new__(cls, *args, **kwargs):
            if args and str(args[0]).endswith("ai_model_gateway.py"):
                return original(tmp_path / "ai_model_gateway.py")
            return original(*args, **kwargs)
    # Test the parser behavior directly using an isolated temporary module copy path.
    monkeypatch.setattr(gateway, "Path", FakePath)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    gateway._load_private_dotenv()
    assert os.environ["OPENAI_API_KEY"] == "process-key"
    assert os.environ["DEEPSEEK_API_KEY"] == "file-deepseek"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file.chmod(0o644)
    gateway._load_private_dotenv()
    assert "DEEPSEEK_API_KEY" not in os.environ


@pytest.mark.parametrize(
    "provider,model,key_env,response,expected_url,expected_header",
    [
        ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY", {"choices": [{"message": {"content": "deepseek ok"}}]}, "https://api.deepseek.com/chat/completions", "Authorization"),
        ("openai", "gpt-4.1-mini", "OPENAI_API_KEY", {"choices": [{"message": {"content": "openai ok"}}]}, "https://api.openai.com/v1/chat/completions", "Authorization"),
        ("claude", "claude-sonnet-4-5", "ANTHROPIC_API_KEY", {"content": [{"type": "text", "text": "claude ok"}]}, "https://api.anthropic.com/v1/messages", "x-api-key"),
        ("gemini", "gemini-2.5-flash", "GEMINI_API_KEY", {"candidates": [{"content": {"parts": [{"text": "gemini ok"}]}}]}, "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", "x-goog-api-key"),
    ],
)
def test_provider_adapters_use_server_keys_and_expected_api_shapes(monkeypatch, provider, model, key_env, response, expected_url, expected_header):
    # Adapter tests must never invoke a real Gemini CLI session even if the developer is logged in.
    monkeypatch.setattr(gateway, "_gemini_cli_logged_in", lambda: False)
    monkeypatch.setenv(key_env, "test-secret-key")
    seen = {}

    def fake_request(url, headers, payload, timeout=30):
        seen.update(url=url, headers=headers, payload=payload)
        return response

    monkeypatch.setattr(gateway, "_request_json", fake_request)
    text = gateway._call_provider(provider, model, "system", "user")
    assert text.endswith("ok")
    assert seen["url"] == expected_url
    assert seen["headers"][expected_header]
    assert "test-secret-key" not in seen["url"]
    if provider == "gemini":
        assert model in seen["url"]
    else:
        assert seen["payload"]["model"] == model


def test_gemini_cli_adapter_uses_cached_google_login_and_never_api_key(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.setattr(gateway, "_gemini_cli_path", lambda: "/usr/local/bin/gemini")
    monkeypatch.setattr(gateway, "Path", Path)
    monkeypatch.setattr(gateway, "_gemini_cli_logged_in", lambda: True)
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used")
    seen = {}

    class Result:
        returncode = 0
        stdout = json.dumps({"response": "CLI Google login works"})
        stderr = ""

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["env"] = kwargs["env"]
        return Result()

    monkeypatch.setattr(gateway.subprocess, "run", fake_run)
    result = gateway._call_gemini_cli("gemini-2.5-flash", "system", "DSP features")
    assert result == "CLI Google login works"
    assert "--output-format" in seen["args"]
    assert seen["env"].get("GEMINI_API_KEY") is None
    assert "DSP features" in seen["args"][seen["args"].index("--prompt") + 1]


def test_gemini_status_reports_cli_google_login_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gateway, "_gemini_cli_logged_in", lambda: True)
    gemini = next(p for p in gateway.provider_status()["providers"] if p["id"] == "gemini")
    assert gemini["configured"] is True
    assert gemini["auth_mode"] == "Gemini CLI · Google login"
    assert gemini["key_env"] is None
