from __future__ import annotations

import pytest

from blossom.annotation import (
    METADATA_ANNOTATION_KEY,
    METADATA_LABEL_KEY,
    METADATA_WEIGHT_KEY,
    AnnotationJoinConfig,
    AnnotationOrphanPolicy,
    AnnotationRecord,
    AnnotationUnannotatedPolicy,
    OrphanAnnotationError,
    UnannotatedError,
    VersionedAnnotationTable,
    apply_annotations,
    collect_annotation_releases,
)
from blossom.schema.row_schema import RowSchema
from blossom.schema.text_schema import TextSchema


def _resolved(records: list[AnnotationRecord]) -> dict[str, AnnotationRecord]:
    return VersionedAnnotationTable(records).resolve_release("r1")


def _schema(sample_id: str, **metadata) -> TextSchema:
    schema = TextSchema(id=sample_id, content=f"content-{sample_id}")
    schema.metadata.update(metadata)
    return schema


def test_approved_records_replace_label_and_weight() -> None:
    resolved = _resolved(
        [
            AnnotationRecord(
                sample_id="s1",
                release="r1",
                revision=2,
                status="approved",
                label="cat",
                weight=0.75,
            )
        ]
    )
    schemas = [_schema("s1", label="stale", weight=9.9)]
    result = apply_annotations(schemas, resolved, AnnotationJoinConfig(release="r1"))

    assert len(result.schemas) == 1
    item = result.schemas[0]
    assert item.metadata[METADATA_LABEL_KEY] == "cat"
    assert item.metadata[METADATA_WEIGHT_KEY] == 0.75
    provenance = item.metadata[METADATA_ANNOTATION_KEY]
    assert provenance == {
        "release": "r1",
        "revision": 2,
        "status": "approved",
        "annotation_id": None,
        "updated_at": None,
    }
    assert result.report.retained == 1
    assert result.report.applied_versions == {"r1": 1}


def test_default_weight_applied_when_record_has_none() -> None:
    resolved = _resolved(
        [AnnotationRecord(sample_id="s1", release="r1", label="cat")]
    )
    result = apply_annotations(
        [_schema("s1")], resolved, AnnotationJoinConfig(release="r1")
    )
    assert result.schemas[0].metadata[METADATA_WEIGHT_KEY] == 1.0


def test_revoked_record_excludes_sample_without_rollback() -> None:
    # Revision 1 was approved, revision 2 revokes it. The newest record wins
    # and the sample must not fall back to the older approved label.
    resolved = _resolved(
        [
            AnnotationRecord(
                sample_id="s1", release="r1", revision=1, label="cat"
            ),
            AnnotationRecord(
                sample_id="s1", release="r1", revision=2, status="revoked"
            ),
        ]
    )
    schemas = [_schema("s1", label="cat", weight=1.0)]
    result = apply_annotations(
        schemas, resolved, AnnotationJoinConfig(release="r1")
    )

    assert result.schemas == []
    assert result.report.excluded == 1
    assert result.report.excluded_ids == ["s1"]
    assert result.report.by_status == {"revoked": 1}
    # Stale metadata is cleared even on the (removed) object, proving the old
    # label cannot survive anywhere in the resulting population.
    assert METADATA_LABEL_KEY not in schemas[0].metadata


def test_rejected_record_excludes_sample() -> None:
    resolved = _resolved(
        [
            AnnotationRecord(
                sample_id="s1", release="r1", status="rejected", label=None
            )
        ]
    )
    result = apply_annotations(
        [_schema("s1")], resolved, AnnotationJoinConfig(release="r1")
    )
    assert result.schemas == []
    assert result.report.excluded_ids == ["s1"]


def test_pending_records_excluded_by_default_and_configurable() -> None:
    resolved = _resolved(
        [
            AnnotationRecord(
                sample_id="s1", release="r1", status="pending", label="maybe"
            )
        ]
    )

    dropped = apply_annotations(
        [_schema("s1")], resolved, AnnotationJoinConfig(release="r1")
    )
    assert dropped.schemas == []
    assert dropped.report.excluded_ids == ["s1"]

    kept = apply_annotations(
        [_schema("s1")],
        resolved,
        AnnotationJoinConfig(
            release="r1", include_pending=True, pending_weight=0.1
        ),
    )
    assert len(kept.schemas) == 1
    item = kept.schemas[0]
    assert item.metadata[METADATA_LABEL_KEY] == "maybe"
    assert item.metadata[METADATA_WEIGHT_KEY] == 0.1
    assert item.metadata[METADATA_ANNOTATION_KEY]["status"] == "pending"


def test_pending_explicit_weight_overrides_pending_default() -> None:
    resolved = _resolved(
        [
            AnnotationRecord(
                sample_id="s1",
                release="r1",
                status="pending",
                label="maybe",
                weight=0.4,
            )
        ]
    )
    result = apply_annotations(
        [_schema("s1")],
        resolved,
        AnnotationJoinConfig(release="r1", include_pending=True),
    )
    assert result.schemas[0].metadata[METADATA_WEIGHT_KEY] == 0.4


def test_unannotated_policies_keep_drop_error() -> None:
    resolved = _resolved(
        [AnnotationRecord(sample_id="s1", release="r1", label="cat")]
    )
    schemas = [_schema("s1"), _schema("s2", label="legacy")]

    kept = apply_annotations(
        schemas,
        resolved,
        AnnotationJoinConfig(
            release="r1",
            unannotated_policy=AnnotationUnannotatedPolicy.KEEP,
        ),
    )
    assert {item.id for item in kept.schemas} == {"s1", "s2"}
    # The unannotated sample keeps its place but loses stale annotation keys.
    s2 = next(item for item in kept.schemas if item.id == "s2")
    assert s2.metadata == {}
    assert kept.report.unannotated == 1
    assert kept.report.unannotated_ids == ["s2"]

    dropped = apply_annotations(
        [_schema("s1"), _schema("s2")],
        resolved,
        AnnotationJoinConfig(
            release="r1", unannotated_policy=AnnotationUnannotatedPolicy.DROP
        ),
    )
    assert [item.id for item in dropped.schemas] == ["s1"]

    with pytest.raises(UnannotatedError, match="has no annotation"):
        apply_annotations(
            [_schema("s1"), _schema("s2")],
            resolved,
            AnnotationJoinConfig(
                release="r1",
                unannotated_policy=AnnotationUnannotatedPolicy.ERROR,
            ),
        )


def test_orphan_ids_are_reported_and_policy_enforced() -> None:
    resolved = _resolved(
        [
            AnnotationRecord(sample_id="s1", release="r1", label="cat"),
            AnnotationRecord(sample_id="ghost", release="r1", label="dog"),
        ]
    )
    with pytest.warns(UserWarning, match="orphan sample id"):
        result = apply_annotations(
            [_schema("s1")],
            resolved,
            AnnotationJoinConfig(
                release="r1", orphan_policy=AnnotationOrphanPolicy.WARN
            ),
        )
    assert result.report.orphan_ids == ["ghost"]

    silent = apply_annotations(
        [_schema("s1")],
        resolved,
        AnnotationJoinConfig(
            release="r1", orphan_policy=AnnotationOrphanPolicy.IGNORE
        ),
    )
    assert silent.report.orphan_ids == ["ghost"]

    with pytest.raises(OrphanAnnotationError, match="absent from the dataset"):
        apply_annotations(
            [_schema("s1")],
            resolved,
            AnnotationJoinConfig(
                release="r1", orphan_policy=AnnotationOrphanPolicy.ERROR
            ),
        )


def test_re_application_clears_stale_metadata_from_older_release() -> None:
    # First release says cat/weight 1.0; second release says dog and revokes
    # another sample. Re-applying must leave no trace of the old values.
    table = VersionedAnnotationTable(
        [
            AnnotationRecord(sample_id="s1", release="r1", label="cat"),
            AnnotationRecord(
                sample_id="s1", release="r2", revision=1, label="dog"
            ),
            AnnotationRecord(sample_id="s2", release="r1", label="bird"),
            AnnotationRecord(
                sample_id="s2", release="r2", revision=1, status="revoked"
            ),
        ]
    )
    schemas = [_schema("s1"), _schema("s2")]

    first = apply_annotations(
        schemas, table.resolve_release("r1"), AnnotationJoinConfig(release="r1")
    )
    assert len(first.schemas) == 2
    s1_after_r1 = next(item for item in first.schemas if item.id == "s1")
    assert s1_after_r1.metadata[METADATA_LABEL_KEY] == "cat"

    second = apply_annotations(
        first.schemas,
        table.resolve_release("r2"),
        AnnotationJoinConfig(release="r2"),
    )
    assert [item.id for item in second.schemas] == ["s1"]
    s1 = second.schemas[0]
    assert s1.metadata[METADATA_LABEL_KEY] == "dog"
    assert s1.metadata[METADATA_ANNOTATION_KEY]["release"] == "r2"
    assert s1.metadata[METADATA_ANNOTATION_KEY]["revision"] == 1
    assert second.report.excluded_ids == ["s2"]


def test_custom_id_func_and_metadata_keys() -> None:
    row = RowSchema(data={"external_id": "ext-7", "text": "hello"})
    resolved = _resolved(
        [
            AnnotationRecord(
                sample_id="ext-7", release="r1", label="positive", weight=2.0
            )
        ]
    )
    config = AnnotationJoinConfig(
        release="r1",
        id_func=lambda schema: schema.data["external_id"],
        label_key="gold_label",
        weight_key="train_weight",
    )
    result = apply_annotations([row], resolved, config)
    item = result.schemas[0]
    assert item.metadata["gold_label"] == "positive"
    assert item.metadata["train_weight"] == 2.0
    assert METADATA_LABEL_KEY not in item.metadata
    assert METADATA_WEIGHT_KEY not in item.metadata


def test_label_mirrored_into_row_data_when_enabled() -> None:
    row = RowSchema(data={"external_id": "ext-7"})
    resolved = _resolved(
        [AnnotationRecord(sample_id="ext-7", release="r1", label="positive")]
    )
    config = AnnotationJoinConfig(
        release="r1",
        id_func=lambda schema: schema.data["external_id"],
        write_label_to_data=True,
    )
    result = apply_annotations([row], resolved, config)
    assert result.schemas[0].data["label"] == "positive"


def test_collect_annotation_releases_reads_provenance() -> None:
    resolved = _resolved(
        [
            AnnotationRecord(sample_id="s1", release="r1", label="cat"),
            AnnotationRecord(sample_id="s2", release="r1", label="dog"),
        ]
    )
    result = apply_annotations(
        [_schema("s1"), _schema("s2")],
        resolved,
        AnnotationJoinConfig(release="r1"),
    )
    assert collect_annotation_releases(result.schemas) == ["r1"]
    assert collect_annotation_releases([_schema("s9")]) == []


def test_negative_config_weights_rejected() -> None:
    with pytest.raises(Exception):
        AnnotationJoinConfig(default_weight=-1.0)
    with pytest.raises(Exception):
        AnnotationJoinConfig(pending_weight=-0.01)
