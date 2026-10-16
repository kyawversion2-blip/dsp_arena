"""Secure, minimal multi-provider gateway for DSP Arena.

API keys are read only from server environment variables. Never pass keys from the
browser or return secret values to clients. No dataset is sent unless the user invokes
an AI action from the UI.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import ssl
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

PROVIDERS: dict[str, dict[str, Any]] = {
    "deepseek": {"env": "DEEPSEEK_API_KEY", "default_model": "deepseek-chat", "models": ["deepseek-chat", "deepseek-reasoner"], "label": "DeepSeek"},
    "openai": {"env": "OPENAI_API_KEY", "default_model": "gpt-4.1-mini", "models": ["gpt-4.1-mini", "gpt-4.1", "o4-mini"], "label": "OpenAI"},
    "claude": {"env": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-4-5", "models": ["claude-sonnet-4-5", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"], "label": "Claude"},
    "gemini": {"env": "GEMINI_API_KEY", "default_model": "gemini-2.5-flash", "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], "label": "Gemini"},
}
FEATURE_FIELDS = ("impact_freq_hz", "laplace_sigma_fit", "fft_centroid_hz", "hfsr", "autocorr_peak", "impact", "duration", "peak_time", "mel_mean_db")
MAX_ROWS = 200
MAX_REQUEST_BYTES = 180_000


def _load_private_dotenv() -> None:
    """Load simple KEY=VALUE entries from project .env without overriding process env."""
    path = Path(__file__).resolve().with_name(".env")
    try:
        if not path.is_file():
            return
        # Refuse overly permissive POSIX files; environment exports remain supported.
        if os.name == "posix" and path.stat().st_mode & 0o077:
            # Do not read secrets from a group/world-readable config file.
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if name not in {cfg["env"] for cfg in PROVIDERS.values()} or name in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if value:
                os.environ[name] = value
    except OSError:
        # Missing/inaccessible local config is equivalent to unconfigured providers.
        return


_load_private_dotenv()

class AIProviderError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _gemini_cli_path() -> str | None:
    """Find the official Gemini CLI without invoking a shell."""
    configured = os.environ.get("DSP_GEMINI_CLI_PATH", "").strip()
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("gemini")
    if found:
        return found
    for candidate in (
        Path.home() / ".npm-global" / "bin" / "gemini",
        Path("/opt/homebrew/bin/gemini"),
        Path("/usr/local/bin/gemini"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _gemini_cli_logged_in() -> bool:
    """Check for a local Gemini CLI OAuth cache; never read or return token values."""
    return bool(_gemini_cli_path() and (Path.home() / ".gemini" / "oauth_creds.json").is_file())


def provider_status() -> dict[str, Any]:
    """Expose provider labels, model choices, and configuration booleans only."""
    providers = []
    for pid, cfg in PROVIDERS.items():
        cli_login = pid == "gemini" and _gemini_cli_logged_in()
        api_key = bool(os.environ.get(cfg["env"], "").strip())
        providers.append({
            "id": pid,
            "label": cfg["label"],
            "configured": bool(cli_login or api_key),
            "auth_mode": "Gemini CLI · Google login" if cli_login else ("API key" if api_key else "not connected"),
            "key_env": cfg["env"] if (pid != "gemini" or not cli_login) else None,
            "default_model": cfg["default_model"],
            "models": cfg["models"],
        })
    return {"providers": providers}


def _call_gemini_cli(model: str, system_prompt: str, user_prompt: str) -> str:
    """Use Gemini CLI's cached Google login in documented non-interactive mode."""
    cli = _gemini_cli_path()
    if not cli:
        raise AIProviderError("Gemini CLI is not installed. Install @google/gemini-cli, then sign in with Google.", 503)
    if not (Path.home() / ".gemini" / "oauth_creds.json").is_file():
        raise AIProviderError("Gemini CLI Google login is missing. Run `gemini` and choose Sign in with Google.", 503)
    if model not in PROVIDERS["gemini"]["models"]:
        raise AIProviderError("Unsupported Gemini model selection.", 422)
    # Prefer cached Google login rather than accidentally routing this request through an API key.
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST / DSP DATA:\n{user_prompt}"
    try:
        proc = subprocess.run(
            [cli, "--model", model, "--prompt", prompt, "--output-format", "json"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120, check=False, env=env, cwd=str(Path(__file__).resolve().parent),
        )
    except subprocess.TimeoutExpired:
        raise AIProviderError("Gemini CLI timed out after 120 seconds. Try a smaller dataset or retry.", 504) from None
    except OSError:
        raise AIProviderError("Could not start Gemini CLI. Verify the CLI installation and permissions.", 503) from None
    if proc.returncode != 0:
        # Do not echo CLI stderr; it may contain account or environment diagnostics.
        raise AIProviderError("Gemini CLI request failed. Check that Google login is active and the account quota is available.", 502)
    try:
        data = json.loads(proc.stdout)
        text = data.get("response") if isinstance(data, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    except (json.JSONDecodeError, TypeError):
        pass
    raise AIProviderError("Gemini CLI returned an unexpected response format.", 502)


def _safe_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def sanitize_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise AIProviderError("Load a CSV dataset before running AI analysis.", 422)
    if len(rows) > MAX_ROWS:
        raise AIProviderError(f"AI requests are limited to {MAX_ROWS} rows per run. Reduce the dataset first.", 413)
    clean: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        item: dict[str, Any] = {"row": index + 1}
        group = str(row.get("group", "")).strip().upper()
        if group in {"A", "B", "C", "D"}:
            item["group"] = group
        for field in FEATURE_FIELDS:
            n = _safe_number(row.get(field))
            if n is not None:
                # Prevent accidental transmission of arbitrary text and absurd numeric values.
                if abs(n) <= 1e12:
                    item[field] = round(n, 7)
        if any(field in item for field in FEATURE_FIELDS):
            clean.append(item)
    if not clean:
        raise AIProviderError("No finite numeric DSP features found. Check the CSV columns.", 422)
    return clean


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row.get("group", "UNLABELED")].append(row)
    summary: dict[str, Any] = {"row_count": len(rows), "groups": {}}
    for group, items in sorted(by_group.items()):
        feat: dict[str, Any] = {}
        for field in FEATURE_FIELDS:
            vals = [r[field] for r in items if field in r]
            if vals:
                feat[field] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 7), "min": min(vals), "max": max(vals)}
        summary["groups"][group] = {"n": len(items), "features": feat}
    return summary


def _request_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=timeout) as resp:
            raw = resp.read(1_000_001)
            if len(raw) > 1_000_000:
                raise AIProviderError("Provider response exceeded the size limit.", 502)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise AIProviderError("Provider returned an invalid response.", 502)
            return data
    except urllib.error.HTTPError as exc:
        code = int(exc.code)
        if code in (401, 403):
            raise AIProviderError("Provider rejected the API key or its permissions. Check the server environment key.", 502) from None
        if code == 429:
            raise AIProviderError("Provider rate limit or quota reached. Try later or check billing.", 429) from None
        if code in (400, 404, 422):
            raise AIProviderError("Provider rejected the request/model. Verify the selected model is available to this key.", 502) from None
        raise AIProviderError(f"AI provider request failed (HTTP {code}). Try again later.", 502) from None
    except (urllib.error.URLError, TimeoutError, ssl.SSLError):
        raise AIProviderError("Could not reach the AI provider or the request timed out.", 504) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise AIProviderError("Provider returned malformed JSON.", 502) from None


def _extract_text(provider: str, data: dict[str, Any]) -> str:
    try:
        if provider in ("openai", "deepseek"):
            text = data["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "\n".join(str(p.get("text", "")) for p in text if isinstance(p, dict))
        elif provider == "claude":
            text = "\n".join(str(block.get("text", "")) for block in data.get("content", []) if isinstance(block, dict) and block.get("type") == "text")
        else:
            text = "\n".join(str(p.get("text", "")) for c in data.get("candidates", []) for p in c.get("content", {}).get("parts", []) if isinstance(p, dict))
        if isinstance(text, str) and text.strip():
            return text.strip()
    except (KeyError, IndexError, TypeError):
        pass
    raise AIProviderError("Provider response did not contain generated text.", 502)


def _call_provider(provider: str, model: str, system_prompt: str, user_prompt: str) -> str:
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["env"], "").strip()
    if model not in cfg["models"]:
        raise AIProviderError("Unsupported model selection for this provider.", 422)
    # Gemini prefers the official CLI's cached Google account session; API key remains an optional fallback.
    if provider == "gemini" and _gemini_cli_logged_in():
        return _call_gemini_cli(model, system_prompt, user_prompt)
    if not key:
        if provider == "gemini":
            raise AIProviderError("Gemini is not connected. Install Gemini CLI and sign in with Google, or configure GEMINI_API_KEY as an optional alternative.", 503)
        raise AIProviderError(f"{cfg['label']} API key is not configured on the server ({cfg['env']}).", 503)

    if provider in ("openai", "deepseek"):
        base = "https://api.openai.com/v1/chat/completions" if provider == "openai" else "https://api.deepseek.com/chat/completions"
        payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1, "max_tokens": 1800}
        if provider == "openai" and model.startswith("o"):
            payload.pop("temperature", None)
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        data = _request_json(base, {"Authorization": f"Bearer {key}"}, payload)
    elif provider == "claude":
        payload = {"model": model, "system": system_prompt, "messages": [{"role": "user", "content": user_prompt}], "max_tokens": 1800, "temperature": 0.1}
        data = _request_json("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}, payload)
    else:
        payload = {"system_instruction": {"parts": [{"text": system_prompt}]}, "contents": [{"role": "user", "parts": [{"text": user_prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1800}}
        # Send the key in a header, not the URL, to reduce accidental URL-log exposure.
        data = _request_json(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": key}, payload)
    return _extract_text(provider, data)


async def run_dsp_ai(provider: str, model: str | None, task: str, rows: Any, dataset_name: str = "uploaded dataset") -> dict[str, Any]:
    if provider not in PROVIDERS:
        raise AIProviderError("Choose DeepSeek, OpenAI, Claude, or Gemini.", 422)
    if task not in ("feature_analysis", "classify"):
        raise AIProviderError("Unsupported DSP AI task.", 422)
    cfg = PROVIDERS[provider]
    chosen_model = model or cfg["default_model"]
    clean = sanitize_rows(rows)
    if task == "feature_analysis":
        summary = summarize_rows(clean)
        system = ("You are a rigorous DSP/statistics assistant. Analyze only the supplied numeric summary and rows. "
                  "Do not invent measurements, p-values, model scores, or causal claims. Distinguish observed values from hypotheses. "
                  "Mention missing features, small sample sizes, possible leakage, scale effects, and validation needed. "
                  "Return concise Markdown with: Findings, Feature usefulness, Caveats, Next validation steps.")
        prompt_data = {"dataset_name": str(dataset_name)[:100], "summary": summary, "sample_rows": clean[:80]}
        target_truth: dict[int, str] = {}
    else:
        # Hold out one labeled row per class (when possible) so the model cannot
        # simply echo its label. Unlabeled rows are also targets. Truth stays local.
        labeled_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        unlabeled: list[dict[str, Any]] = []
        for row in clean:
            if row.get("group") in {"A", "B", "C", "D"}:
                labeled_by_group[row["group"]].append(row)
            else:
                unlabeled.append(row)
        train: list[dict[str, Any]] = []
        targets: list[dict[str, Any]] = []
        target_truth: dict[str, str] = {}
        for row in unlabeled:
            target_id = f"U{len(targets) + 1}"
            targets.append({**{k: v for k, v in row.items() if k != "row"}, "target_id": target_id})
        for group, group_rows in labeled_by_group.items():
            if len(group_rows) >= 2:
                holdout = group_rows[-1]
                target_id = f"T{len(targets) + 1}"
                target = {k: v for k, v in holdout.items() if k not in ("group", "row")}
                target["target_id"] = target_id
                targets.append(target)
                target_truth[target_id] = group
                train.extend(group_rows[:-1])
            else:
                train.extend(group_rows)
        train_groups = {r.get("group") for r in train if r.get("group")}
        if len(train_groups) < 2:
            raise AIProviderError("Classification needs labeled reference examples from at least two groups; use at least two rows per group for a holdout check.", 422)
        if not targets:
            raise AIProviderError("No unlabeled or holdout rows available to classify.", 422)
        summary = summarize_rows(train)
        system = ("You are a DSP feature-based classification assistant, not a trained statistical model. "
                  "Reference examples contain ground-truth group labels; target rows have labels intentionally removed. "
                  "Predict only for target rows using numeric DSP features. Never infer a label from row number or dataset name. "
                  "Assign A/B/C/D only when supported by references; otherwise UNCERTAIN. Do not claim training or general accuracy. "
                  "Return strict JSON object: {\"predictions\":[{\"target_id\":\"T1\",\"predicted_group\":\"A|B|C|D|UNCERTAIN\",\"confidence\":0..1,\"reason\":\"short\"}],\"limitations\":\"string\"}.")
        reference_rows = [{k: v for k, v in row.items() if k != "row"} for row in train[:160]]
        prompt_data = {"dataset_name": str(dataset_name)[:100], "group_feature_summary_from_training_rows_only": summary, "labeled_reference_rows": reference_rows, "target_rows_without_labels": targets[:80], "evaluation_note": "Some targets are hidden-label holdouts; do not expect their true labels. target_id is only a response join key, not a feature."}
    prompt = json.dumps(prompt_data, ensure_ascii=False)
    if len(prompt.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise AIProviderError("Prepared dataset summary is too large for a safe request. Reduce rows/features.", 413)
    text = await asyncio.to_thread(_call_provider, provider, chosen_model, system, prompt)
    parsed = None
    if task == "classify":
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("predictions"), list):
                parsed = None
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None and target_truth:
            predictions = {str(p["target_id"]): str(p.get("predicted_group", "")).upper() for p in parsed["predictions"] if isinstance(p, dict) and p.get("target_id") is not None}
            comparable = [(target_id, truth) for target_id, truth in target_truth.items() if target_id in predictions and predictions[target_id] in {"A", "B", "C", "D", "UNCERTAIN"}]
            parsed["holdout_evaluation"] = {"n_evaluated": len(comparable), "n_holdout": len(target_truth), "accuracy": round(sum(predictions[target_id] == truth for target_id, truth in comparable) / len(comparable), 4) if comparable else None, "note": "Accuracy applies only to this small in-dataset holdout, not to future/generalization performance."}
    return {"status": "ok", "provider": provider, "model": chosen_model, "task": task, "rows_in_dataset": len(clean), "summary": summary, "result": parsed if parsed is not None else text, "format_warning": task == "classify" and parsed is None}
