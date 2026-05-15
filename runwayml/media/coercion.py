"""Shared coercion helpers for image/video inputs across Runway nodes.

Mirrors the design of ``griptape_nodes_library.media`` from the standard
library (see commit 2e01a9638b78). The two libraries deliberately share the
same shape so a string returned by ``coerce_media_url_or_data_uri`` is
interchangeable across providers.

Runway nodes accept inputs in many shapes: plain strings (URLs, project macro
paths like ``{inputs}/foo.png``, filesystem paths, ``data:`` URIs), artifact
objects (``ImageUrlArtifact``, ``ImageArtifact`` and their video counterparts),
and serialized artifact dicts. These helpers normalize all of that into a
single string suitable for ``File(...).read_data_uri()`` or for passing to
the RunwayML API.

Pass-through rules:

* HTTP(S) URLs, ``data:<kind>/...`` URIs, ``runway://`` URIs, project macro
  paths, and plain filesystem paths are returned unchanged (whitespace
  stripped).
* Only artifact ``.base64`` payloads, or serialized ``<Kind>Artifact`` dicts,
  are wrapped as ``data:`` URIs. Raw strings are never wrapped.
"""

from __future__ import annotations

from typing import Any, Literal

from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.griptape_nodes import logger

MediaKind = Literal["image", "video"]

_DEFAULT_MIME_SUBTYPE: dict[MediaKind, str] = {
    "image": "png",
    "video": "mp4",
}

_FALLBACK_MIME: dict[MediaKind, str] = {
    "image": "image/png",
    "video": "video/mp4",
}

_ARTIFACT_NAME: dict[MediaKind, str] = {
    "image": "Image",
    "video": "Video",
}

# Schemes the RunwayML API accepts directly, so callers don't have to download
# and re-encode them as data URIs.
DEFAULT_PASS_THROUGH_SCHEMES: tuple[str, ...] = ("https://", "runway://")


def coerce_media_url_or_data_uri(val: Any, *, kind: MediaKind) -> str | None:
    """Extract a usable string from a media input value.

    Returns one of:

    * an HTTP(S) URL,
    * a ``data:<kind>/...`` URI,
    * a ``runway://`` URI,
    * a project macro path like ``{inputs}/foo.png``,
    * a plain filesystem path,

    or ``None`` if the value cannot be resolved. Non-URI strings are NOT
    wrapped as raw base64; only artifact ``.base64`` payloads (or serialized
    ``<Kind>Artifact`` dicts) are wrapped as ``data:`` URIs.
    """
    if val is None:
        return None

    mime_prefix = f"data:{kind}/"
    default_subtype = _DEFAULT_MIME_SUBTYPE[kind]
    raw_artifact_type = f"{_ARTIFACT_NAME[kind]}Artifact"

    if isinstance(val, dict):
        return _coerce_from_dict(
            val,
            mime_prefix=mime_prefix,
            default_subtype=default_subtype,
            raw_artifact_type=raw_artifact_type,
        )

    if isinstance(val, str):
        v = val.strip()
        return v or None

    try:
        to_dict = getattr(val, "to_dict", None)
        if callable(to_dict):
            serialized = to_dict()
            if isinstance(serialized, dict):
                coerced = coerce_media_url_or_data_uri(serialized, kind=kind)
                if coerced:
                    return coerced

        v = getattr(val, "value", None)
        if isinstance(v, str) and v.strip():
            return v.strip()

        b64 = getattr(val, "base64", None)
        if isinstance(b64, str) and b64:
            return b64 if b64.startswith(mime_prefix) else f"{mime_prefix}{default_subtype};base64,{b64}"
    except Exception:  # noqa: BLE001 - unknown artifact shapes raise arbitrary errors; treat as unresolvable
        return None

    return None


def _coerce_from_dict(
    val: dict[str, Any],
    *,
    mime_prefix: str,
    default_subtype: str,
    raw_artifact_type: str,
) -> str | None:
    value = val.get("value")
    if not isinstance(value, str) or not value.strip():
        # Fall back to a top-level ``url`` key, used by some serialized URL artifact shapes.
        url = val.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
        return None
    stripped = value.strip()

    # URLs / data URIs / runway:// URIs always pass through, regardless of declared artifact type.
    if stripped.startswith(("http://", "https://", "runway://", mime_prefix)):
        return stripped

    # Raw <Kind>Artifact: the value is base64 bytes, optionally with a "format"
    # hint that names the MIME subtype.
    if val.get("type") == raw_artifact_type:
        subtype = str(val.get("format") or default_subtype).lower()
        return f"{mime_prefix}{subtype};base64,{stripped}"

    # Anything else (macro paths, filesystem paths, unknown artifact shapes)
    # passes through; File() resolves it downstream.
    return stripped


def prepare_media_data_uri(
    val: Any,
    *,
    kind: MediaKind,
    node_name: str | None = None,
    fallback_mime: str | None = None,
    pass_through_schemes: tuple[str, ...] = DEFAULT_PASS_THROUGH_SCHEMES,
) -> str | None:
    """Coerce a media input and resolve it to a value the RunwayML API accepts.

    Returns one of:

    * a URI matching ``pass_through_schemes`` (default: ``https://``, ``runway://``),
    * a ``data:<kind>/...`` URI already on the input,
    * a ``data:<kind>/...;base64,...`` URI produced by reading the resolved
      path or URL via ``File``,

    or ``None`` if the input is empty/unresolvable, or if the underlying read
    fails. Read failures are logged at debug level; callers that care about
    distinguishing missing inputs from failed reads should check for ``None``
    and surface their own validation error.

    ``fallback_mime`` is used by ``File.read_data_uri`` when it cannot
    otherwise determine a MIME type from the resolved bytes; defaults to the
    canonical MIME for ``kind`` (``image/png`` or ``video/mp4``).
    """
    if not val:
        return None

    media_url = coerce_media_url_or_data_uri(val, kind=kind)
    if not media_url:
        return None

    mime_prefix = f"data:{kind}/"
    if media_url.startswith(mime_prefix):
        return media_url

    if any(media_url.startswith(scheme) for scheme in pass_through_schemes):
        return media_url

    try:
        return File(media_url).read_data_uri(fallback_mime=fallback_mime or _FALLBACK_MIME[kind])
    except FileLoadError as e:
        prefix = f"{node_name} " if node_name else ""
        logger.debug("%sfailed to load %s from %s: %s", prefix, kind, media_url, e)
        return None
