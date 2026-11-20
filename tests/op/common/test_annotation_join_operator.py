from __future__ import annotations

import pytest

from blossom.annotation import (
    METADATA_ANNOTATION_KEY,
    AnnotationJoinConfig,
    AnnotationOrphanPolicy,
    AnnotationUnannotatedPolicy,
    ReviewStatus,
    VersionedAnnotationTable,
)
from blossom.dataframe.local_dataframe import LocalDataFrame
from blossom.op import AnnotationJoinOperator
from blossom.schema.text_schema import TextSchema


def _table() -> VersionedAnnotationTable:
    return VersionedAnnotationTable.from_dicts(
        [
            # s1: approved at rev 1 then revoked at rev 2 -> must be excluded.
            {
                "sample_id": "s1",
                "release": "r1",
                "revision": 1,
                "status": "approved",
                "label": "cat",
            },
            {
                "sample_id": "s1",
                "release": "r1",
                "revision": 2,
                "status": "revoked",
            },
            # s2: approved with an explicit training weight.
            {
                "sample_id": "s2",
                "release": "r1",
                "revision": 1,
                "status": "approved",
                "label": "dog",
                "weight": 0.5,
            },
            # s3: pending -> excluded by default.
            {
                "sample_id": "s3",
                "release": "r1",
                "revision": 1,
                "status": "pending",
                "label": "bird",
            },
            # Orphan: no matching dataset sample.
            {
                "sample_id": "ghost",
                "release": "r1",
                "revision": 1,
                "status": "approved",
                "label": "fish",
            },
        ]
    )


def _dataframe() -> LocalDataFrame:
    return LocalDataFrame(
        [
            TextSchema(id="s1", content="a"),
            TextSchema(id="s2", content="b"),
            TextSchema(id="s3", content="c"),
            TextSchema(id="s4", content="d"),
        ]
    )


def test_operator_changes_population_labels_and_weights() -> None:
    operator = AnnotationJoinOperator(_table())
    result = operator.process(_dataframe()).collect()

    # s1 revoked, s3 pending -> only s2 annotated, s4 unannotated and kept.
    assert sorted(item.id for item in result) == ["s2", "s4"]

    s2 = next(item for item in result if item.id == "s2")
    assert s2.metadata["label"] == "dog"
    assert s2.metadata["weight"] == 0.5
    assert s2.metadata[METADATA_ANNOTATION_KEY] == {
        "release": "r1",
        "revision": 1,
        "status": ReviewStatus.APPROVED.value,
        "annotation_id": None,
        "updated_at": None,
    }

    report = operator.report
    assert report is not None
    assert report.release == "r1"
    assert report.retained == 2
    assert report.excluded == 2
    assert sorted(report.excluded_ids) == ["s1", "s3"]
    assert report.unannotated_ids == ["s4"]
    assert report.orphan_ids == ["ghost"]
    assert report.by_status == {"approved": 1, "revoked": 1, "pending": 1}


def test_operator_include_pending_and_drop_unannotated() -> None:
    config = AnnotationJoinConfig(
        include_pending=True,
        pending_weight=0.0,
        unannotated_policy=AnnotationUnannotatedPolicy.DROP,
    )
    operator = AnnotationJoinOperator(_table(), config)
    result = operator.process(_dataframe()).collect()

    assert sorted(item.id for item in result) == ["s2", "s3"]
    s3 = next(item for item in result if item.id == "s3")
    assert s3.metadata["label"] == "bird"
    assert s3.metadata["weight"] == 0.0
    assert s3.metadata[METADATA_ANNOTATION_KEY]["status"] == "pending"


def test_operator_orphan_error_policy() -> None:
    config = AnnotationJoinConfig(
        orphan_policy=AnnotationOrphanPolicy.ERROR,
    )
    operator = AnnotationJoinOperator(_table(), config)
    with pytest.raises(ValueError, match="absent from the dataset"):
        operator.process(_dataframe())


def test_operator_unknown_release_raises() -> None:
    operator = AnnotationJoinOperator(
        _table(), AnnotationJoinConfig(release="nope")
    )
    with pytest.raises(ValueError, match="unknown annotation release"):
        operator.process(_dataframe())


def test_operator_is_picklable_for_distributed_transforms() -> None:
    import cloudpickle

    operator = AnnotationJoinOperator(_table())
    restored = cloudpickle.loads(cloudpickle.dumps(operator))
    result = restored.process(_dataframe()).collect()
    assert sorted(item.id for item in result) == ["s2", "s4"]


def test_operator_runs_on_multiprocess_dataframe() -> None:
    from blossom.dataframe.multiprocess_dataframe import MultiProcessDataFrame

    operator = AnnotationJoinOperator(
        _table(),
        AnnotationJoinConfig(
            include_pending=True,
            orphan_policy=AnnotationOrphanPolicy.IGNORE,
        ),
    )
    dataframe = MultiProcessDataFrame(
        [
            TextSchema(id="s1", content="a"),
            TextSchema(id="s2", content="b"),
            TextSchema(id="s3", content="c"),
            TextSchema(id="s4", content="d"),
        ]
    )
    result = operator.process(dataframe).collect()
    assert sorted(item.id for item in result) == ["s2", "s3", "s4"]
    s2 = next(item for item in result if item.id == "s2")
    assert s2.metadata["label"] == "dog"
    assert s2.metadata["weight"] == 0.5
    # Distributed reports still name the exact release that was applied.
    assert operator.report is not None
    assert operator.report.release == "r1"


def test_operator_distributed_orphan_error_enforced_per_partition() -> None:
    from blossom.dataframe.multiprocess_dataframe import MultiProcessDataFrame

    operator = AnnotationJoinOperator(
        _table(),
        AnnotationJoinConfig(orphan_policy=AnnotationOrphanPolicy.ERROR),
    )
    # The table contains the orphan sample "ghost"; with the ERROR policy a
    # partition processing it raises instead of silently dropping it.
    with pytest.raises(Exception, match="absent from the dataset"):
        operator.process(
            MultiProcessDataFrame(
                [
                    TextSchema(id="s1", content="a"),
                    TextSchema(id="s2", content="b"),
                    TextSchema(id="s3", content="c"),
                    TextSchema(id="s4", content="d"),
                ]
            )
        ).collect()
