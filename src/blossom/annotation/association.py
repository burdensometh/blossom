"""Associate a dataset with a resolved annotation release.

This module is the boundary between the annotation domain model and the
training data.  :func:`apply_annotations` does not merely attach an audit
table: it rewrites each sample's label and training weight in place and
removes samples whose newest annotation excludes them, so the resulting
dataset is the exact training population defined by the release.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, field_validator

from blossom.annotation.record import AnnotationRecord, ReviewStatus
from blossom.util.type import StrEnum

#: Metadata key holding the effective label used for training.
METADATA_LABEL_KEY = "label"
#: Metadata key holding the effective training weight.
METADATA_WEIGHT_KEY = "weight"
#: Metadata key holding provenance of the applied annotation (release,
#: revision, status, ...). Its presence after a round trip proves which
#: annotation version produced a stored dataset.
METADATA_ANNOTATION_KEY = "annotation"

# Metadata keys managed by this module. Re-applying annotations first clears
# them so values left behind by an older release can never linger.
_MANAGED_KEYS = (METADATA_ANNOTATION_KEY, METADATA_LABEL_KEY, METADATA_WEIGHT_KEY)


class AnnotationOrphanPolicy(StrEnum):
    """What to do when the annotation table references IDs absent from the
    dataset."""

    #: Keep the orphan IDs in the report but do not fail.
    WARN = "warn"
    #: Raise an :class:`OrphanAnnotationError`.
    ERROR = "error"
    #: Record orphans silently in the report.
    IGNORE = "ignore"


class AnnotationUnannotatedPolicy(StrEnum):
    """What to do with dataset samples that have no record in the release."""

    #: Keep the sample, clearing any label/weight written by an older release.
    KEEP = "keep"
    #: Remove the sample from the training population.
    DROP = "drop"
    #: Raise an :class:`UnannotatedError`.
    ERROR = "error"


class UnannotatedError(ValueError):
    """Raised when a dataset sample has no annotation and the policy is
    ``ERROR``."""


class OrphanAnnotationError(ValueError):
    """Raised when annotation records reference unknown sample IDs and the
    orphan policy is ``ERROR``."""


class AnnotationJoinConfig(BaseModel):
    """Configuration controlling how a release is applied to a dataset.

    Attributes:
        release: Name of the annotation release to join. Required when the
            table spans more than one release.
        id_func: Callable extracting the join key from a schema. Defaults to
            the schema's own ``id``.
        include_pending: Keep samples whose newest record is still pending
            review. They are excluded by default.
        pending_weight: Weight assigned to retained pending records when the
            record itself does not specify a weight.
        default_weight: Weight assigned to retained records (approved or
            pending) without an explicit weight.
        orphan_policy: How to handle annotation IDs missing from the dataset.
        unannotated_policy: How to handle dataset samples without annotations.
        clear_stale: Clear label/weight/annotation metadata written by a
            previous release before applying this one.
        write_label_to_data: Also mirror the label into a schema's ``data``
            dict for :class:`~blossom.schema.row_schema.RowSchema` rows, using
            ``label_key`` as the column name.
        label_key: Metadata key (and optional data column) for the label.
        weight_key: Metadata key for the training weight.
    """

    release: Optional[str] = None
    id_func: Optional[Callable[[Any], str]] = Field(default=None, exclude=True)
    include_pending: bool = False
    pending_weight: float = 0.0
    default_weight: float = 1.0
    orphan_policy: AnnotationOrphanPolicy = AnnotationOrphanPolicy.WARN
    unannotated_policy: AnnotationUnannotatedPolicy = (
        AnnotationUnannotatedPolicy.KEEP
    )
    clear_stale: bool = True
    write_label_to_data: bool = False
    label_key: str = METADATA_LABEL_KEY
    weight_key: str = METADATA_WEIGHT_KEY

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("pending_weight", "default_weight")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("weights must be >= 0")
        return value


class AnnotationReport(BaseModel):
    """Audit summary of one association run.

    The report complements — but never replaces — the in-place label/weight
    rewrite. It records counts and the IDs involved in every decision so a
    pipeline can log or assert what the release did.
    """

    release: Optional[str] = None
    total_samples: int = 0
    retained: int = 0
    excluded: int = 0
    unannotated: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)
    excluded_ids: list[str] = Field(default_factory=list)
    unannotated_ids: list[str] = Field(default_factory=list)
    orphan_ids: list[str] = Field(default_factory=list)
    applied_versions: dict[str, int] = Field(default_factory=dict)


class AnnotationAssociationResult:
    """Output of :func:`apply_annotations`: rewritten schemas plus a report."""

    def __init__(self, schemas: list[Any], report: AnnotationReport) -> None:
        self.schemas = schemas
        self.report = report


def _default_id_func(schema: Any) -> str:
    return str(schema.id)


def _clear_managed_metadata(schema: Any, config: AnnotationJoinConfig) -> None:
    for key in _MANAGED_KEYS:
        schema.metadata.pop(key, None)
    # Also clear customized label/weight keys when they differ from defaults.
    schema.metadata.pop(config.label_key, None)
    schema.metadata.pop(config.weight_key, None)


def _provenance(record: AnnotationRecord) -> dict[str, Any]:
    return {
        "release": record.release,
        "revision": record.revision,
        "status": record.status.value,
        "annotation_id": record.annotation_id,
        "updated_at": record.updated_at,
    }


def apply_annotations(
    schemas: list[Any],
    resolved: dict[str, AnnotationRecord],
    config: Optional[AnnotationJoinConfig] = None,
) -> AnnotationAssociationResult:
    """Apply resolved per-sample annotations to a list of schemas.

    Selection rule: ``resolved`` already holds the newest in-release record of
    every sample (see :meth:`VersionedAnnotationTable.resolve_release`), so a
    newer revoked/rejected record removes the sample with no fallback to an
    older approved revision.

    Effects on each retained sample:

    * ``metadata[label_key]`` is replaced by the record's label;
    * ``metadata[weight_key]`` is replaced by the record's weight (or a
      configured default);
    * ``metadata["annotation"]`` records the exact release/revision/status.

    Excluded and dropped samples are physically removed from the returned
    list, i.e. the association changes the dataset's population rather than
    just annotating it.

    Raises:
        UnannotatedError: With ``unannotated_policy=ERROR``.
        OrphanAnnotationError: With ``orphan_policy=ERROR``.
    """
    config = config or AnnotationJoinConfig()
    id_func = config.id_func or _default_id_func

    dataset_ids = {str(id_func(schema)) for schema in schemas}
    orphan_ids = sorted(set(resolved) - dataset_ids)
    if orphan_ids:
        if config.orphan_policy == AnnotationOrphanPolicy.ERROR:
            raise OrphanAnnotationError(
                f"annotation references {len(orphan_ids)} sample id(s) "
                f"absent from the dataset, e.g. {orphan_ids[:5]}"
            )
        if config.orphan_policy == AnnotationOrphanPolicy.WARN:
            warnings.warn(
                f"annotation table contains {len(orphan_ids)} orphan sample "
                f"id(s) absent from the dataset, e.g. {orphan_ids[:5]}",
                stacklevel=2,
            )

    report = AnnotationReport(
        release=config.release,
        total_samples=len(schemas),
        orphan_ids=orphan_ids,
    )

    output: list[Any] = []
    excluded_ids: list[str] = []
    unannotated_ids: list[str] = []
    by_status: dict[str, int] = {}
    applied_versions: dict[str, int] = {}

    for schema in schemas:
        sample_id = str(id_func(schema))
        record = resolved.get(sample_id)

        if config.clear_stale:
            _clear_managed_metadata(schema, config)

        if record is None:
            report.unannotated += 1
            unannotated_ids.append(sample_id)
            if config.unannotated_policy == AnnotationUnannotatedPolicy.ERROR:
                raise UnannotatedError(
                    f"sample {sample_id!r} has no annotation in release "
                    f"{config.release!r}"
                )
            if config.unannotated_policy == AnnotationUnannotatedPolicy.DROP:
                continue
            output.append(schema)
            continue

        status = record.status
        by_status[status.value] = by_status.get(status.value, 0) + 1

        if status in (ReviewStatus.REJECTED, ReviewStatus.REVOKED):
            # Terminal negative verdicts: never fall back to an older
            # approved revision and never enter training.
            excluded_ids.append(sample_id)
            continue
        if status == ReviewStatus.PENDING and not config.include_pending:
            excluded_ids.append(sample_id)
            continue

        if status == ReviewStatus.PENDING:
            weight = (
                record.weight
                if record.weight is not None
                else config.pending_weight
            )
        else:
            weight = (
                record.weight
                if record.weight is not None
                else config.default_weight
            )

        schema.metadata[config.label_key] = record.label
        schema.metadata[config.weight_key] = weight
        schema.metadata[METADATA_ANNOTATION_KEY] = _provenance(record)
        if config.write_label_to_data and hasattr(schema, "data") and isinstance(
            schema.data, dict
        ):
            schema.data[config.label_key] = record.label

        applied_versions[record.release] = (
            applied_versions.get(record.release, 0) + 1
        )
        output.append(schema)

    report.retained = len(output)
    report.excluded = len(excluded_ids)
    report.excluded_ids = excluded_ids
    report.unannotated_ids = unannotated_ids
    report.by_status = by_status
    report.applied_versions = applied_versions
    return AnnotationAssociationResult(output, report)


def collect_annotation_releases(schemas: list[Any]) -> list[str]:
    """Return the sorted annotation releases proven on a list of schemas.

    Reads the provenance block written by :func:`apply_annotations`, so the
    annotation version used to produce a stored dataset is recoverable after
    a JSON/Parquet round trip.
    """
    releases = set()
    for schema in schemas:
        provenance = schema.metadata.get(METADATA_ANNOTATION_KEY)
        if isinstance(provenance, dict) and provenance.get("release"):
            releases.add(str(provenance["release"]))
    return sorted(releases)
