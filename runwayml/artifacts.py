"""Custom artifacts shared by the RunwayML nodes.

`VideoUrlArtifact` is deliberately absent: import it from `griptape.artifacts`. This
library used to define its own copy in four node modules, each subclassing
`ImageUrlArtifact`, which made every generated video claim to be an image.
"""

from __future__ import annotations

from typing import Any

from griptape.artifacts import BaseArtifact, ImageUrlArtifact

# Runway matches a reference image to the prompt by tag, so the two travel as one value.
# Runway rejects a reference image whose aspect ratio falls outside this range.
MIN_REFERENCE_ASPECT_RATIO = 0.5
MAX_REFERENCE_ASPECT_RATIO = 2.0

_PREVIEW_LENGTH = 50


class ReferenceImageArtifact(BaseArtifact):
    """An image paired with the `@tag` used to address it from a prompt."""

    def __init__(self, image: ImageUrlArtifact | str | dict, tag: str, name: str | None = None, **kwargs) -> None:
        value = {"image": image, "tag": tag.strip() if tag else ""}
        # BaseArtifact generates a name when none is given, so pass it only when set.
        if name is not None:
            kwargs["name"] = name
        super().__init__(value=value, **kwargs)
        self.image = image
        self.tag = tag.strip() if tag else ""

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

    A `ReferenceImageArtifact` exposes `image`/`tag` directly, but the same value comes
    back from a saved workflow as a plain dict, and from `BaseArtifact` deserialization
    as an artifact whose `value` is that dict. Returns `(None, "")` for anything else.
    """
    if hasattr(candidate, "image") and hasattr(candidate, "tag"):
        return candidate.image, str(candidate.tag or "")

    value = getattr(candidate, "value", None)
    if isinstance(value, dict) and "image" in value:
        return value["image"], str(value.get("tag") or "")

    if isinstance(candidate, dict) and "image" in candidate:
        return candidate["image"], str(candidate.get("tag") or "")

    return None, ""
