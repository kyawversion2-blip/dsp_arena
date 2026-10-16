"""Deterministic R1 network-policy tests; no server or external network required."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest


@pytest.fixture
def policy_module():
    import network_policy
    return network_policy


def test_default_is_loopback_and_local_origins(monkeypatch, policy_module):
    for key in ("DSP_ALLOW_LAN", "DSP_AUTH_TOKEN", "DSP_ALLOWED_ORIGINS", "DSP_ALLOWED_HOSTS", "DSP_BIND_HOST"):
        monkeypatch.delenv(key, raising=False)
    policy = policy_module.NetworkPolicy.from_environment(port=8765)
    assert policy.host == "127.0.0.1"
    assert policy.lan_enabled is False
    assert policy.allowed_origins == ("http://127.0.0.1:8765", "http://localhost:8765")
    assert policy.validate_request(host="localhost:8765", origin="http://localhost:8765", token=None) is None
    assert policy.validate_request(host="evil.example:8765", origin="http://localhost:8765", token=None) == "HOST_NOT_ALLOWED"
    assert policy.validate_request(host="localhost:8765", origin="https://evil.example", token=None) == "ORIGIN_NOT_ALLOWED"

    monkeypatch.setenv("DSP_BIND_HOST", "0.0.0.0")
    with pytest.raises(policy_module.NetworkPolicyError):
        policy_module.NetworkPolicy.from_environment(port=8765)
    monkeypatch.delenv("DSP_BIND_HOST")
    monkeypatch.setenv("DSP_ALLOWED_ORIGINS", "*")
    with pytest.raises(policy_module.NetworkPolicyError):
        policy_module.NetworkPolicy.from_environment(port=8765)
    monkeypatch.delenv("DSP_ALLOWED_ORIGINS")


def test_lan_requires_authentication_and_explicit_origins(monkeypatch, policy_module):
    monkeypatch.setenv("DSP_ALLOW_LAN", "1")
    monkeypatch.delenv("DSP_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("DSP_ALLOWED_ORIGINS", "http://192.168.1.2:8765")
    monkeypatch.setenv("DSP_ALLOWED_HOSTS", "192.168.1.2")
    with pytest.raises(policy_module.NetworkPolicyError):
        policy_module.NetworkPolicy.from_environment(port=8765)

    monkeypatch.setenv("DSP_AUTH_TOKEN", "a-secure-token-123")
    policy = policy_module.NetworkPolicy.from_environment(port=8765)
    assert policy.host == "0.0.0.0"
    assert policy.validate_request(host="192.168.1.2:8765", origin="http://192.168.1.2:8765", token=None) == "AUTH_REQUIRED"
    assert policy.validate_request(host="192.168.1.2:8765", origin="http://192.168.1.2:8765", token="wrong") == "AUTH_REQUIRED"
    assert policy.validate_request(host="192.168.1.2:8765", origin="http://192.168.1.2:8765", token="a-secure-token-123") is None
    assert policy.validate_request(host="192.168.1.2:8765", origin="http://evil.example", token="a-secure-token-123") == "ORIGIN_NOT_ALLOWED"
    assert policy.validate_request(host="evil.example:8765", origin="http://192.168.1.2:8765", token="a-secure-token-123") == "HOST_NOT_ALLOWED"


def test_no_wildcard_cors_or_unconditional_wildcard_bind():
    root = Path(__file__).resolve().parents[2]
    live = (root / "live_server.py").read_text()
    backend = (root / "real_dsp_backend.py").read_text()
    assert 'allow_origins=["*"]' not in live
    assert 'allow_origins=["*"]' not in backend
    assert 'host="0.0.0.0"' not in live
    assert 'host="0.0.0.0"' not in backend
    assert "host=NETWORK_POLICY.host" in live
    assert "host=NETWORK_POLICY.host" in backend
    assert "allow_credentials=False" in live
    assert "allow_credentials=False" in backend
    assert "app.add_middleware(_NetworkBoundaryMiddleware)" in live
    assert "app.add_middleware(_NetworkBoundaryMiddleware)" in backend
    assert "NETWORK_POLICY.validate_request" in live
    assert "NETWORK_POLICY.validate_request" in backend
