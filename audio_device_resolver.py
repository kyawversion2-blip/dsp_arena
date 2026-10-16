#!/usr/bin/env python3
"""audio_device_resolver.py — Deterministic audio input device resolution.

Supports:
1. Explicit device index (int or numeric string).
2. Device name selector (exact or case-insensitive substring).
3. Automatic default resolution that avoids BlackHole / virtual loopback devices
   in favor of valid physical microphones.
4. Channel count and sample rate capability validation.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import sounddevice as sd

log = logging.getLogger("audio_device_resolver")

VIRTUAL_DEVICE_KEYWORDS = (
    "blackhole",
    "loopback",
    "soundflower",
    "virtual audio",
    "virtual microphone",
    "virtual mic",
    "teams audio",
    "zoom audio",
    "null",
)

PREFERRED_PHYSICAL_KEYWORDS = (
    "macbook air microphone",
    "macbook pro microphone",
    "built-in microphone",
    "internal microphone",
    "macbook microphone",
    "microphone",
)

MAX_SUPPORTED_SR: int = 96_000


class AudioDeviceResolutionError(RuntimeError):
    """Raised when an audio input device cannot be resolved or is incompatible."""
    pass


def is_virtual_device(device_name: str) -> bool:
    """Check if device name indicates a virtual loopback device."""
    name_lower = device_name.strip().lower()
    return any(k in name_lower for k in VIRTUAL_DEVICE_KEYWORDS)


def get_available_input_devices() -> list[dict[str, Any]]:
    """Enumerate all available audio devices that support audio input."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    inputs = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            dev_copy = dict(d)
            dev_copy["index"] = idx
            hapi_idx = d.get("hostapi", 0)
            dev_copy["hostapi_name"] = hostapis[hapi_idx]["name"] if hapi_idx < len(hostapis) else "unknown"
            inputs.append(dev_copy)
    return inputs


def format_device_list(devices: list[dict[str, Any]]) -> str:
    """Format input device list for clear, actionable error reporting."""
    lines = []
    for d in devices:
        idx = d.get("index", "?")
        name = d.get("name", "unknown")
        inch = d.get("max_input_channels", 0)
        sr = d.get("default_samplerate", 0)
        api = d.get("hostapi_name", "unknown")
        virt = " [VIRTUAL/LOOPBACK]" if is_virtual_device(name) else ""
        lines.append(f"  [{idx:2d}] {name:<32} (InCh: {inch}, DefSR: {sr:.0f} Hz, API: {api}){virt}")
    return "\n".join(lines)


def resolve_audio_input_device(
    device_spec: int | str | None = None,
    channels: int = 1,
    requested_sr: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Resolve an audio input device index and metadata deterministically.

    Args:
        device_spec: Device index (int), device name/substring (str), or None for auto.
        channels: Required number of input channels (default 1).
        requested_sr: Optional requested sample rate to validate capability.

    Returns:
        tuple[int, dict[str, Any]]: (resolved_index, device_info_dict)

    Raises:
        AudioDeviceResolutionError: If device cannot be found, is virtual on default fallback,
                                    or does not support requested channels.
    """
    # Check environment variable fallback if device_spec is None
    if device_spec is None:
        env_val = os.environ.get("MIC_DEVICE")
        if env_val is not None and env_val.strip() != "":
            device_spec = env_val.strip()

    available = get_available_input_devices()
    if not available:
        raise AudioDeviceResolutionError("No audio input devices found on system.")

    resolved_idx: int | None = None
    resolved_info: dict[str, Any] | None = None

    # Case 1: Explicit Integer (or numeric string)
    is_numeric = False
    if isinstance(device_spec, int):
        is_numeric = True
        cand_idx = device_spec
    elif isinstance(device_spec, str) and device_spec.strip().isdigit():
        is_numeric = True
        cand_idx = int(device_spec.strip())

    if is_numeric:
        match = next((d for d in available if d["index"] == cand_idx), None)
        if match is None:
            raise AudioDeviceResolutionError(
                f"Configured input device index [{cand_idx}] not found or has 0 input channels.\n"
                f"Available input devices:\n{format_device_list(available)}"
            )
        resolved_idx = cand_idx
        resolved_info = match

    # Case 2: Explicit String (Name or Substring)
    elif isinstance(device_spec, str) and device_spec.strip() != "":
        spec_lower = device_spec.strip().lower()
        # 2a. Exact match
        exact_matches = [d for d in available if d["name"].strip().lower() == spec_lower]
        if len(exact_matches) == 1:
            resolved_info = exact_matches[0]
            resolved_idx = resolved_info["index"]
        elif len(exact_matches) > 1:
            # Prefer Core Audio on macOS
            core_audio = next((d for d in exact_matches if d.get("hostapi_name") == "Core Audio"), None)
            resolved_info = core_audio or exact_matches[0]
            resolved_idx = resolved_info["index"]
        else:
            # 2b. Substring match
            sub_matches = [d for d in available if spec_lower in d["name"].strip().lower()]
            if len(sub_matches) == 1:
                resolved_info = sub_matches[0]
                resolved_idx = resolved_info["index"]
            elif len(sub_matches) > 1:
                # Prefer non-virtual
                non_virt = [d for d in sub_matches if not is_virtual_device(d["name"])]
                pool = non_virt if non_virt else sub_matches
                # If all pool candidates share the same device name across host APIs, choose Core Audio
                unique_names = set(d["name"].strip().lower() for d in pool)
                if len(unique_names) == 1:
                    core_audio = next((d for d in pool if d.get("hostapi_name") == "Core Audio"), None)
                    resolved_info = core_audio or pool[0]
                    resolved_idx = resolved_info["index"]
                else:
                    # Ambiguous match between distinct physical devices
                    names_str = ", ".join(f"[{d['index']}] '{d['name']}'" for d in pool)
                    raise AudioDeviceResolutionError(
                        f"Ambiguous device selector '{device_spec}' matched multiple input devices: {names_str}. "
                        "Please specify an exact device index or unambiguous full name."
                    )
            else:
                raise AudioDeviceResolutionError(
                    f"No input device matching '{device_spec}' found.\n"
                    f"Available input devices:\n{format_device_list(available)}"
                )

    # Case 3: No explicit device configured -> Auto-select physical microphone
    else:
        # Check system default input device
        default_idx, _ = sd.default.device
        default_cand = next((d for d in available if d["index"] == default_idx), None)

        if default_cand is not None and not is_virtual_device(default_cand["name"]):
            resolved_idx = default_idx
            resolved_info = default_cand
        else:
            # Default is virtual (e.g. BlackHole) or missing -> search for physical microphone
            log.warning(
                "System default input [%s] '%s' is virtual or loopback. Searching for physical microphone...",
                default_idx, default_cand["name"] if default_cand else "unknown"
            )
            # Find physical candidates
            physical_cands = [d for d in available if not is_virtual_device(d["name"])]

            best_cand = None
            # Search by preferred keywords
            for kw in PREFERRED_PHYSICAL_KEYWORDS:
                for d in physical_cands:
                    if kw in d["name"].lower():
                        best_cand = d
                        break
                if best_cand:
                    break

            if best_cand is None and physical_cands:
                # Fallback to first non-virtual input device
                best_cand = physical_cands[0]

            if best_cand is not None:
                resolved_idx = best_cand["index"]
                resolved_info = best_cand
                log.info("Auto-selected physical microphone: [%d] %s", resolved_idx, best_cand["name"])
            else:
                raise AudioDeviceResolutionError(
                    "System default input is a virtual loopback device (e.g. BlackHole), "
                    "and no physical microphone could be discovered.\n"
                    f"Available input devices:\n{format_device_list(available)}"
                )

    # Final Validation
    if resolved_info is None or resolved_idx is None:
        raise AudioDeviceResolutionError("Failed to resolve a valid audio input device.")

    max_in = resolved_info.get("max_input_channels", 0)
    if max_in < channels:
        raise AudioDeviceResolutionError(
            f"Device [{resolved_idx}] '{resolved_info['name']}' supports {max_in} input channels, "
            f"but {channels} channels were requested."
        )

    # Sample-rate capability validation and verified supported-rate fallback
    if requested_sr is not None:
        sr_valid = False
        if requested_sr > MAX_SUPPORTED_SR:
            log.warning(
                "Requested sample rate %d Hz on device [%d] '%s' exceeds maximum supported capacity (%d Hz). "
                "Probing supported fallback rates...",
                requested_sr, resolved_idx, resolved_info["name"], MAX_SUPPORTED_SR
            )
        else:
            try:
                sd.check_input_settings(device=resolved_idx, samplerate=requested_sr, channels=channels)
                sr_valid = True
                resolved_info["actual_supported_sr"] = requested_sr
            except Exception as check_exc:
                log.warning(
                    "Device [%d] '%s' rejected requested sample rate %d Hz (%s). Probing supported fallback rates...",
                    resolved_idx, resolved_info["name"], requested_sr, check_exc
                )

        if not sr_valid:
            default_sr = int(round(resolved_info.get("default_samplerate", 0)))
            fallback_candidates = [
                default_sr if default_sr <= MAX_SUPPORTED_SR else 0,
                48000,
                44100,
                96000,
                16000,
            ]
            seen = {requested_sr, 0}
            fallbacks = []
            for r in fallback_candidates:
                if r not in seen and 0 < r <= MAX_SUPPORTED_SR:
                    seen.add(r)
                    fallbacks.append(r)

            for cand_sr in fallbacks:
                try:
                    sd.check_input_settings(device=resolved_idx, samplerate=cand_sr, channels=channels)
                    resolved_info["fallback_sr"] = cand_sr
                    resolved_info["actual_supported_sr"] = cand_sr
                    sr_valid = True
                    log.warning(
                        "Negotiated verified fallback sample rate %d Hz for device [%d] '%s' (requested: %d Hz).",
                        cand_sr, resolved_idx, resolved_info["name"], requested_sr
                    )
                    break
                except Exception:
                    continue

        if not sr_valid:
            if requested_sr > MAX_SUPPORTED_SR:
                raise AudioDeviceResolutionError(
                    f"Requested sample rate {requested_sr} Hz exceeds maximum supported capacity ({MAX_SUPPORTED_SR} Hz), "
                    f"and no supported fallback rate <= {MAX_SUPPORTED_SR} Hz could be negotiated on device [{resolved_idx}] '{resolved_info['name']}'."
                )
            raise AudioDeviceResolutionError(
                f"Device [{resolved_idx}] '{resolved_info['name']}' does not support requested sample rate "
                f"{requested_sr} Hz or any verified fallback rate."
            )

    return resolved_idx, resolved_info
