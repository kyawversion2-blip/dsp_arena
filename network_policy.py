"""Local-only network defaults and explicit authenticated LAN policy for DSP Arena.

This module contains policy/configuration only. It does not create sockets or make
network requests. LAN mode is deliberately fail-closed: it requires a non-empty
authentication token and an explicit origin allowlist.
"""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from urllib.parse import urlsplit


LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class NetworkPolicyError(ValueError):
    """Raised when an unsafe network policy is requested."""


def _csv_env(name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())


def _host_only(value: str) -> str:
    value = value.strip().lower()
    parsed = urlsplit(value if "://" in value else f"//{value}")
    return (parsed.hostname or "").strip("[]").lower()


def _validate_origin(value: str) -> str:
    value = value.strip()
    parsed = urlsplit(value)
    if value == "*" or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise NetworkPolicyError(f"invalid or wildcard origin: {value!r}")
    return value


@dataclass(frozen=True)
class NetworkPolicy:
    host: str
    port: int
    lan_enabled: bool
    auth_token: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls, *, port: int) -> "NetworkPolicy":
        lan_enabled = os.environ.get("DSP_ALLOW_LAN", "").lower() in {"1", "true", "yes"}
        token = os.environ.get("DSP_AUTH_TOKEN", "")
        configured_origins = tuple(_validate_origin(x) for x in _csv_env("DSP_ALLOWED_ORIGINS"))
        configured_hosts = tuple(_host_only(x) for x in _csv_env("DSP_ALLOWED_HOSTS"))
        if any(x == "*" for x in configured_hosts):
            raise NetworkPolicyError("DSP_ALLOWED_HOSTS must not contain a wildcard")
        if lan_enabled and (len(token) < 16 or not configured_origins):
            raise NetworkPolicyError(
                "DSP_ALLOW_LAN requires DSP_AUTH_TOKEN (at least 16 characters) "
                "and a non-empty DSP_ALLOWED_ORIGINS allowlist"
            )

        local_origins = (f"http://127.0.0.1:{port}", f"http://localhost:{port}")
        origins = configured_origins or local_origins
        origin_hosts = tuple(_host_only(urlsplit(x).netloc) for x in origins)
        hosts = tuple(dict.fromkeys((*LOCAL_HOSTS, *configured_hosts, *origin_hosts)))
        requested_host = os.environ.get("DSP_BIND_HOST", "0.0.0.0" if lan_enabled else "127.0.0.1")
        if not lan_enabled and requested_host not in LOCAL_HOSTS:
            raise NetworkPolicyError("non-loopback DSP_BIND_HOST requires DSP_ALLOW_LAN=1")
        return cls(
            host=requested_host,
            port=port,
            lan_enabled=lan_enabled,
            auth_token=token,
            allowed_hosts=hosts,
            allowed_origins=origins,
        )

    def host_allowed(self, host_header: str | None) -> bool:
        if not host_header:
            return False
        return _host_only(host_header) in self.allowed_hosts

    def origin_allowed(self, origin: str | None) -> bool:
        return origin is None or origin in self.allowed_origins

    def authenticated(self, token: str | None) -> bool:
        return bool(self.auth_token and token and hmac.compare_digest(token, self.auth_token))

    def token_from_headers(self, headers) -> str | None:
        authorization = headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return headers.get("x-dsp-auth-token")

    def token_from_websocket(self, websocket) -> str | None:
        token = self.token_from_headers(websocket.headers)
        if token:
            return token
        # Browser WebSocket clients cannot set Authorization headers. Query-token
        # support is opt-in LAN functionality and must use HTTPS outside localhost.
        return websocket.query_params.get("access_token")

    def validate_request(self, *, host: str | None, origin: str | None, token: str | None) -> str | None:
        if not self.host_allowed(host):
            return "HOST_NOT_ALLOWED"
        if not self.origin_allowed(origin):
            return "ORIGIN_NOT_ALLOWED"
        if self.lan_enabled and not self.authenticated(token):
            return "AUTH_REQUIRED"
        return None
