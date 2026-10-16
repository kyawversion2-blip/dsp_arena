#!/usr/bin/env python3
"""recording_metadata.py — Standardized, backward-compatible recording provenance metadata.

Captures hardware source, sample rates, channels, positioning, distance,
and preprocessing steps without fabricating unknown parameters.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RecordingMetadata:
    """Provenance record for acoustic impact recordings."""
    source_type: str = "unknown"  # "builtin_mic" | "bluetooth_mic" | "imported_wav" | "unknown"
    requested_sr_hz: Optional[int] = None
    actual_sr_hz: Optional[int] = None
    channel_count: Optional[int] = None
    duration_s: Optional[float] = None
    recording_position: str = "unknown"  # "left" | "right" | "center" | "unknown"
    microphone_distance_cm: Optional[float] = None  # None if unknown
    recording_device_name: str = "unknown"
    recording_constraints: dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: list[str] = field(default_factory=list)
    source_file_name: str = "unknown"
    recording_session_id: str = "unknown"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        # Clean null values if appropriate, but preserve explicit unknowns
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordingMetadata:
        """Construct from dictionary safely."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def create_recording_metadata(
    source_type: str = "unknown",
    requested_sr: Optional[int] = None,
    actual_sr: Optional[int] = None,
    channels: Optional[int] = None,
    duration_s: Optional[float] = None,
    position: str = "unknown",
    distance_cm: Optional[float] = None,
    device_name: str = "unknown",
    constraints: Optional[dict[str, Any]] = None,
    preprocessing: Optional[list[str]] = None,
    file_name: str = "unknown",
    session_id: str = "unknown",
    notes: str = "",
) -> dict[str, Any]:
    """Factory function returning a normalized recording metadata dict."""
    valid_sources = {"builtin_mic", "bluetooth_mic", "imported_wav", "unknown"}
    src = source_type if source_type in valid_sources else "unknown"

    valid_positions = {"left", "right", "center", "unknown"}
    pos = position.lower() if position and position.lower() in valid_positions else "unknown"

    meta = RecordingMetadata(
        source_type=src,
        requested_sr_hz=requested_sr,
        actual_sr_hz=actual_sr,
        channel_count=channels,
        duration_s=round(duration_s, 6) if duration_s is not None else None,
        recording_position=pos,
        microphone_distance_cm=float(distance_cm) if distance_cm is not None else None,
        recording_device_name=device_name or "unknown",
        recording_constraints=constraints or {},
        preprocessing_steps=preprocessing or [],
        source_file_name=file_name or "unknown",
        recording_session_id=session_id or "unknown",
        notes=notes or "",
    )
    return meta.to_dict()
