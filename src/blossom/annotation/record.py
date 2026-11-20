"""Domain model for versioned external annotation records.

An annotation record describes the human/annotator verdict for one sample at
one point in time.  Records are grouped into immutable *releases*; inside a
release the same sample may appear several times with increasing *revision*
numbers.  The newest revision of a sample within a release always wins, which
is what makes later corrections (including revocations) take effect instead of
leaving stale labels in training data.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from blossom.util.type import StrEnum

DEFAULT_RELEASE_FIELD = "release"
DEFAULT_REVISION_FIELD = "revision"
DEFAULT_STATUS_FIELD = "status"
DEFAULT_LABEL_FIELD = "label"
DEFAULT_WEIGHT_FIELD = "weight"
DEFAULT_UPDATED_AT_FIELD = "updated_at"
DEFAULT_ANNOTATION_ID_FIELD = "annotation_id"


class ReviewStatus(StrEnum):
    """Review state of an annotation record.

    Attributes:
        APPROVED: The annotation passed review and may enter training.
        PENDING: The annotation is still under review.
        REJECTED: The annotation was rejected during review.
        REVOKED: A previously approved annotation was withdrawn. Revocation is
            terminal: selecting a newer revoked record must never fall back to
            an older approved revision of the same sample.
    """

    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"
    REVOKED = "revoked"

    @classmethod
    def from_value(cls, value: Any) -> "ReviewStatus":
        """Build a :class:`ReviewStatus` from a string-like value.

        Accepts the enum members, their string values and common uppercase
        spellings.

        Raises:
            ValueError: If the value does not map to a known review status.
        """
        if isinstance(value, ReviewStatus):
            return value
        if not isinstance(value, str):
            raise ValueError(f"Unsupported review status: {value!r}")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported review status: {value!r}")


class DuplicateAnnotationError(ValueError):
    """Raised when an annotation table contains ambiguous duplicate records."""


class AnnotationRecord(BaseModel):
    """One versioned annotation verdict for a sample.

    Attributes:
        sample_id: ID of the annotated sample; joins against the dataset's ID.
        release: Name of the annotation release this record belongs to.
        revision: Monotonically increasing version number inside a release.
        status: Review status of this revision.
        label: Label decided by this revision. Required for retained records;
            rejected/revoked records may leave it unset.
        weight: Training weight decided by this revision. Must be >= 0.
        annotation_id: Optional globally unique ID of the annotation entry.
        updated_at: Optional ordering timestamp (ISO string or comparable
            value); used as a tie-breaker when revisions compare equal.
        extra: Any additional annotation payload, preserved verbatim.
    """

    sample_id: str
    release: str
    revision: int = 0
    status: ReviewStatus = ReviewStatus.APPROVED
    label: Optional[Any] = None
    weight: Optional[float] = None
    annotation_id: Optional[str] = None
    updated_at: Optional[Any] = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}

    @field_validator("sample_id", "release")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("sample_id and release must be non-empty strings")
        return value

    @field_validator("revision")
    @classmethod
    def _non_negative_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("revision must be >= 0")
        return value

    @field_validator("weight")
    @classmethod
    def _non_negative_weight(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("weight must be >= 0")
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _parse_status(cls, value: Any) -> ReviewStatus:
        return ReviewStatus.from_value(value)

    @model_validator(mode="after")
    def _check_retention_fields(self) -> "AnnotationRecord":
        if self.status == ReviewStatus.APPROVED and self.label is None:
            raise ValueError(
                f"approved annotation for sample {self.sample_id!r} must carry a label"
            )
        return self

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        release: Optional[str] = None,
        field_mapping: Optional[dict[str, str]] = None,
    ) -> "AnnotationRecord":
        """Create a record from a raw annotation-table row.

        Args:
            data: Raw row, e.g. one JSON/JSONL object of an annotation export.
            release: Release to assign when the row has no release column.
            field_mapping: Optional override of the canonical column names,
                keyed by the logical field names ``release``, ``revision``,
                ``status``, ``label``, ``weight``, ``updated_at``,
                ``annotation_id`` and ``sample_id``.

        Raises:
            ValueError: If a required field is missing or invalid.
        """
        mapping = dict(field_mapping or {})
        sample_key = mapping.get("sample_id", "sample_id")
        release_key = mapping.get("release", DEFAULT_RELEASE_FIELD)
        revision_key = mapping.get("revision", DEFAULT_REVISION_FIELD)
        status_key = mapping.get("status", DEFAULT_STATUS_FIELD)
        label_key = mapping.get("label", DEFAULT_LABEL_FIELD)
        weight_key = mapping.get("weight", DEFAULT_WEIGHT_FIELD)
        updated_key = mapping.get("updated_at", DEFAULT_UPDATED_AT_FIELD)
        annotation_key = mapping.get(
            "annotation_id", DEFAULT_ANNOTATION_ID_FIELD
        )

        if sample_key not in data:
            raise ValueError(
                f"annotation row is missing sample id column {sample_key!r}"
            )

        resolved_release = data.get(release_key, release)
        if resolved_release is None:
            raise ValueError(
                "annotation row has no release; pass an explicit release"
            )

        known_keys = {
            sample_key,
            release_key,
            revision_key,
            status_key,
            label_key,
            weight_key,
            updated_key,
            annotation_key,
        }
        extra = {
            key: value
            for key, value in data.items()
            if key not in known_keys and value is not None
        }

        return cls(
            sample_id=data[sample_key],
            release=str(resolved_release),
            revision=data.get(revision_key, 0) or 0,
            status=data.get(status_key, ReviewStatus.APPROVED),
            label=data.get(label_key),
            weight=data.get(weight_key),
            annotation_id=data.get(annotation_key),
            updated_at=data.get(updated_key),
            extra=extra,
        )

    def version_key(self) -> tuple[int, Any]:
        """Ordering key used to pick the newest revision inside a release."""
        return self.revision, self.updated_at
