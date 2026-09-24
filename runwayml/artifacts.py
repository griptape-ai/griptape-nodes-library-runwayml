"""Custom artifacts shared by the RunwayML nodes.

`VideoUrlArtifact` is deliberately absent: import it from `griptape.artifacts` rather than
declaring a local subclass, so a generated video is never typed as an image.
"""

from __future__ import annotations

from typing import Any

from griptape.artifacts import BaseArtifact, ImageUrlArtifact

# Runway rejects a reference image whose aspect ratio falls outside this range.
MIN_REFERENCE_ASPECT_RATIO = 0.5
MAX_REFERENCE_ASPECT_RATIO = 2.0

_PREVIEW_LENGTH = 50


class ReferenceImageArtifact(BaseArtifact):
    """An image paired with the `@tag` used to address it from a prompt.

    `image` and `tag` read straight off `value`, so the artifact holds one copy of its data.
    Keeping them as separate attributes as well meant an artifact restored from a serialized
    `value` could disagree with itself.
    """

    def __init__(self, image: ImageUrlArtifact | str | dict, tag: str, name: str | None = None, **kwargs) -> None:
        # BaseArtifact generates a name when none is given, so pass it only when set.
        if name is not None:
            kwargs["name"] = name
        super().__init__(value={"image": image, "tag": tag.strip() if tag else ""}, **kwargs)

    @property
    def image(self) -> Any:
        return self.value["image"]

    @property
    def tag(self) -> str:
        return str(self.value.get("tag") or "")

    def to_text(self) -> str:
        """Return a text representation of the reference image."""
        image_type = type(self.image).__name__
        source = getattr(self.image, "value", self.image)
        preview = str(source)
        if len(preview) > _PREVIEW_LENGTH:
            preview = preview[:_PREVIEW_LENGTH] + "..."

        return f"ReferenceImage(tag='{self.tag}', image={image_type}({preview}))"

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return self.to_text()


def unpack_reference_image(candidate: Any) -> tuple[Any | None, str]:
    """Pull the image and tag out of a reference image, whatever shape it arrives in.

    A `ReferenceImageArtifact` exposes `image`/`tag` as properties over its `value`, but the
    same data arrives from a saved workflow as a plain dict, and from `BaseArtifact`
    deserialization as an artifact whose `value` is that dict. Returns `(None, "")` for
    anything else.
    """
    if hasattr(candidate, "image") and hasattr(candidate, "tag"):
        return candidate.image, str(candidate.tag or "")

    value = getattr(candidate, "value", None)
    if isinstance(value, dict) and "image" in value:
        return value["image"], str(value.get("tag") or "")

    if isinstance(candidate, dict) and "image" in candidate:
        return candidate["image"], str(candidate.get("tag") or "")

    return None, ""
