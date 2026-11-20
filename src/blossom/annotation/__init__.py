"""Versioned external annotation support.

External annotation tables are revised continuously per sample ID.  The
objects in this package make those revisions explicit: every record belongs to
a named *release* and carries an increasing *revision* number together with a
review status.  Associating a dataset with a release always selects the newest
in-release revision of every sample, so stale labels can never silently survive
a re-annotation pass.
"""

from blossom.annotation.record import (
    DEFAULT_ANNOTATION_ID_FIELD,
    DEFAULT_LABEL_FIELD,
    DEFAULT_RELEASE_FIELD,
    DEFAULT_REVISION_FIELD,
    DEFAULT_STATUS_FIELD,
    DEFAULT_UPDATED_AT_FIELD,
    DEFAULT_WEIGHT_FIELD,
    AnnotationRecord,
    DuplicateAnnotationError,
    ReviewStatus,
)
from blossom.annotation.association import (
    AnnotationAssociationResult,
    AnnotationJoinConfig,
    AnnotationOrphanPolicy,
    AnnotationReport,
    AnnotationUnannotatedPolicy,
    METADATA_ANNOTATION_KEY,
    METADATA_LABEL_KEY,
    METADATA_WEIGHT_KEY,
    OrphanAnnotationError,
    UnannotatedError,
    apply_annotations,
    collect_annotation_releases,
)
from blossom.annotation.table import VersionedAnnotationTable

__all__ = [
    "DEFAULT_ANNOTATION_ID_FIELD",
    "DEFAULT_LABEL_FIELD",
    "DEFAULT_RELEASE_FIELD",
    "DEFAULT_REVISION_FIELD",
    "DEFAULT_STATUS_FIELD",
    "DEFAULT_UPDATED_AT_FIELD",
    "DEFAULT_WEIGHT_FIELD",
    "AnnotationAssociationResult",
    "AnnotationJoinConfig",
    "AnnotationOrphanPolicy",
    "AnnotationRecord",
    "AnnotationReport",
    "AnnotationUnannotatedPolicy",
    "DuplicateAnnotationError",
    "METADATA_ANNOTATION_KEY",
    "METADATA_LABEL_KEY",
    "METADATA_WEIGHT_KEY",
    "OrphanAnnotationError",
    "ReviewStatus",
    "UnannotatedError",
    "VersionedAnnotationTable",
    "apply_annotations",
    "collect_annotation_releases",
]
