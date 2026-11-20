from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from blossom.annotation import (
    AnnotationRecord,
    DuplicateAnnotationError,
    ReviewStatus,
    VersionedAnnotationTable,
)


def _record(
    sample_id: str, revision: int, status: str = "approved", **kwargs
) -> AnnotationRecord:
    return AnnotationRecord(
        sample_id=sample_id,
        release="r1",
        revision=revision,
        status=status,
        label=kwargs.pop("label", "cat"),
        **kwargs,
    )


def test_review_status_parsing_accepts_aliases() -> None:
    assert ReviewStatus.from_value("APPROVED") == ReviewStatus.APPROVED
    assert ReviewStatus.from_value(" revoked ") == ReviewStatus.REVOKED
    assert ReviewStatus.from_value(ReviewStatus.PENDING) == ReviewStatus.PENDING
    with pytest.raises(ValueError, match="Unsupported review status"):
        ReviewStatus.from_value("nope")
    with pytest.raises(ValueError, match="Unsupported review status"):
        ReviewStatus.from_value(123)


def test_record_validates_weight_revision_and_approved_label() -> None:
    with pytest.raises(ValidationError):
        AnnotationRecord(sample_id="s", release="r", weight=-0.1)
    with pytest.raises(ValidationError):
        AnnotationRecord(sample_id="s", release="r", revision=-1)
    with pytest.raises(ValidationError, match="must carry a label"):
        AnnotationRecord(sample_id="s", release="r", status="approved")
    # Rejected/revoked records do not require a label.
    revoked = AnnotationRecord(sample_id="s", release="r", status="revoked")
    assert revoked.status == ReviewStatus.REVOKED


def test_from_dict_uses_field_mapping_and_captures_extra() -> None:
    row = {
        "uid": "s1",
        "ver": 3,
        "state": "approved",
        "verdict": "dog",
        "score": 0.25,
        "annotator": "alice",
    }
    record = AnnotationRecord.from_dict(
        row,
        release="rel-2026-11",
        field_mapping={
            "sample_id": "uid",
            "revision": "ver",
            "status": "state",
            "label": "verdict",
            "weight": "score",
        },
    )
    assert record.sample_id == "s1"
    assert record.release == "rel-2026-11"
    assert record.revision == 3
    assert record.label == "dog"
    assert record.weight == 0.25
    assert record.extra == {"annotator": "alice"}


def test_from_dict_requires_sample_id_and_release() -> None:
    with pytest.raises(ValueError, match="sample id column"):
        AnnotationRecord.from_dict({"release": "r"})
    with pytest.raises(ValueError, match="no release"):
        AnnotationRecord.from_dict({"sample_id": "s"})


def test_table_rejects_duplicate_annotation_ids_and_version_keys() -> None:
    table = VersionedAnnotationTable()
    table.add_record(
        AnnotationRecord(
            sample_id="s1",
            release="r1",
            revision=1,
            annotation_id="a1",
            label="cat",
        )
    )
    with pytest.raises(DuplicateAnnotationError, match="duplicate annotation_id"):
        table.add_record(
            AnnotationRecord(
                sample_id="s2",
                release="r1",
                revision=1,
                annotation_id="a1",
                label="dog",
            )
        )
    with pytest.raises(DuplicateAnnotationError, match="duplicate annotation"):
        table.add_record(
            AnnotationRecord(sample_id="s1", release="r1", revision=1, label="cat")
        )
    # The same revision in a *different* release is allowed.
    table.add_record(
        AnnotationRecord(sample_id="s1", release="r2", revision=1, label="cat")
    )


def test_resolve_release_picks_newest_revision() -> None:
    rows = [
        {"sample_id": "s1", "release": "r1", "revision": 1, "label": "cat"},
        {"sample_id": "s1", "release": "r1", "revision": 3, "label": "horse"},
        {"sample_id": "s1", "release": "r1", "revision": 2, "label": "dog"},
        {"sample_id": "s2", "release": "r1", "revision": 1, "label": "bird"},
    ]
    table = VersionedAnnotationTable.from_dicts(rows)
    resolved = table.resolve_release("r1")
    assert resolved["s1"].revision == 3
    assert resolved["s1"].label == "horse"
    assert resolved["s2"].revision == 1


def test_resolve_release_updated_at_breaks_revision_ties() -> None:
    table = VersionedAnnotationTable(
        [
            AnnotationRecord(
                sample_id="s1",
                release="r1",
                revision=1,
                updated_at="2026-10-01T00:00:00Z",
                label="old",
            ),
            AnnotationRecord(
                sample_id="s1",
                release="r1",
                revision=1,
                updated_at="2026-11-01T00:00:00Z",
                label="new",
            ),
        ]
    )
    assert table.resolve_release("r1")["s1"].label == "new"


def test_revocation_wins_over_earlier_approved_revision() -> None:
    table = VersionedAnnotationTable(
        [
            _record("s1", 1, "approved", label="cat"),
            _record("s1", 2, "revoked"),
        ]
    )
    effective = table.resolve_release("r1")["s1"]
    assert effective.status == ReviewStatus.REVOKED
    assert table.history("s1", "r1")[0].status == ReviewStatus.APPROVED


def test_release_selection_rules() -> None:
    table = VersionedAnnotationTable(
        [
            AnnotationRecord(sample_id="s1", release="r1", label="a"),
            AnnotationRecord(sample_id="s1", release="r2", label="b"),
        ]
    )
    assert table.releases() == ["r1", "r2"]
    with pytest.raises(ValueError, match="unknown annotation release"):
        table.resolve_release("r9")
    with pytest.raises(ValueError, match="multiple annotation releases"):
        table.resolve_release()

    single = VersionedAnnotationTable(
        [AnnotationRecord(sample_id="s1", release="only", label="a")]
    )
    assert single.resolve_release()["s1"].release == "only"


def test_status_counts_reflect_effective_records_only() -> None:
    table = VersionedAnnotationTable(
        [
            _record("s1", 1, "approved"),
            _record("s1", 2, "revoked"),
            _record("s2", 1, "approved"),
            _record("s3", 1, "pending"),
        ]
    )
    counts = table.status_counts("r1")
    assert counts == {"approved": 1, "pending": 1, "rejected": 0, "revoked": 1}


def test_empty_table_resolution_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="contains no records"):
        VersionedAnnotationTable().resolve_release()
    assert VersionedAnnotationTable().releases() == []
    assert VersionedAnnotationTable().sample_ids() == set()


def test_record_requires_non_empty_sample_id_and_release() -> None:
    with pytest.raises(ValidationError):
        AnnotationRecord(sample_id="  ", release="r1", label="cat")
    with pytest.raises(ValidationError):
        AnnotationRecord(sample_id="s1", release="", label="cat")


def test_from_json_rejects_non_object_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps([{"sample_id": "s1"}, "not-an-object"]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="list of object rows"):
        VersionedAnnotationTable.from_json(str(bad))

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text(
        json.dumps({"sample_id": "s1"}) + "\n" + json.dumps(["nope"]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="list of object rows"):
        VersionedAnnotationTable.from_json(str(bad_jsonl))


def test_from_json_supports_jsonl_and_json_array(tmp_path: Path) -> None:
    jsonl = tmp_path / "annotations.jsonl"
    jsonl.write_text(
        json.dumps(
            {"sample_id": "s1", "release": "r1", "revision": 1, "label": "cat"}
        )
        + "\n",
        encoding="utf-8",
    )
    table = VersionedAnnotationTable.from_json(str(jsonl))
    assert table.sample_ids() == {"s1"}

    array_file = tmp_path / "annotations.json"
    array_file.write_text(
        json.dumps(
            [
                {
                    "sample_id": "s2",
                    "release": "r1",
                    "revision": 1,
                    "label": "dog",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert VersionedAnnotationTable.from_json(str(array_file)).sample_ids() == {
        "s2"
    }

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert VersionedAnnotationTable.from_json(str(empty)).records == []
