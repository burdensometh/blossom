from __future__ import annotations

import json
from pathlib import Path

import pytest

from blossom.annotation import (
    METADATA_ANNOTATION_KEY,
    AnnotationJoinConfig,
    VersionedAnnotationTable,
)
from blossom.dataset import DatasetEngine, create_dataset, load_dataset
from blossom.schema.text_schema import TextSchema


def _table() -> VersionedAnnotationTable:
    return VersionedAnnotationTable.from_dicts(
        [
            {
                "sample_id": "s1",
                "release": "rel-2026-11",
                "revision": 1,
                "status": "approved",
                "label": "cat",
                "weight": 1.0,
            },
            {
                "sample_id": "s2",
                "release": "rel-2026-11",
                "revision": 1,
                "status": "approved",
                "label": "dog",
                "weight": 0.25,
            },
            {
                "sample_id": "s3",
                "release": "rel-2026-11",
                "revision": 1,
                "status": "revoked",
            },
        ]
    )


def _dataset():
    return create_dataset(
        [
            TextSchema(id="s1", content="alpha"),
            TextSchema(id="s2", content="beta"),
            TextSchema(id="s3", content="gamma"),
        ],
        engine=DatasetEngine.LOCAL,
    )


def test_apply_annotations_rewrites_dataset_and_tracks_release() -> None:
    joined = _dataset().apply_annotations(_table())
    items = joined.collect()

    # Population changed: revoked s3 is physically removed.
    assert sorted(item.id for item in items) == ["s1", "s2"]
    labels = {item.id: item.metadata["label"] for item in items}
    weights = {item.id: item.metadata["weight"] for item in items}
    assert labels == {"s1": "cat", "s2": "dog"}
    assert weights == {"s1": 1.0, "s2": 0.25}
    assert joined.annotation_releases() == ["rel-2026-11"]


def test_annotation_provenance_survives_json_round_trip(tmp_path: Path) -> None:
    joined = _dataset().apply_annotations(_table())
    path = tmp_path / "annotated.jsonl"
    joined.write_json(str(path))

    raw_lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert {line["id"] for line in raw_lines} == {"s1", "s2"}
    for line in raw_lines:
        assert line["metadata"][METADATA_ANNOTATION_KEY]["release"] == (
            "rel-2026-11"
        )

    reloaded = load_dataset(str(path), engine=DatasetEngine.LOCAL)
    assert reloaded.annotation_releases() == ["rel-2026-11"]
    reloaded_items = {item.id: item for item in reloaded.collect()}
    assert reloaded_items["s1"].metadata["label"] == "cat"
    assert reloaded_items["s2"].metadata["weight"] == 0.25
    assert (
        reloaded_items["s2"].metadata[METADATA_ANNOTATION_KEY]["status"]
        == "approved"
    )


def test_annotation_provenance_survives_parquet_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    joined = _dataset().apply_annotations(_table())
    path = tmp_path / "annotated.parquet"
    joined.write_parquet(str(path))

    reloaded = load_dataset(
        str(path), engine=DatasetEngine.LOCAL, data_type="parquet"
    )
    items = {item.id: item for item in reloaded.collect()}
    assert sorted(items) == ["s1", "s2"]
    assert reloaded.annotation_releases() == ["rel-2026-11"]
    assert items["s1"].metadata["label"] == "cat"
    assert items["s2"].metadata["weight"] == 0.25
    assert items["s2"].metadata[METADATA_ANNOTATION_KEY]["release"] == (
        "rel-2026-11"
    )


def test_reapplying_newer_release_replaces_old_labels(tmp_path: Path) -> None:
    table = VersionedAnnotationTable.from_dicts(
        [
            {
                "sample_id": "s1",
                "release": "r-old",
                "revision": 1,
                "status": "approved",
                "label": "cat",
            },
            {
                "sample_id": "s1",
                "release": "r-new",
                "revision": 1,
                "status": "approved",
                "label": "feline",
                "weight": 0.1,
            },
            {
                "sample_id": "s2",
                "release": "r-old",
                "revision": 1,
                "status": "approved",
                "label": "dog",
            },
            {
                "sample_id": "s2",
                "release": "r-new",
                "revision": 1,
                "status": "revoked",
            },
        ]
    )
    dataset = create_dataset(
        [TextSchema(id="s1", content="a"), TextSchema(id="s2", content="b")],
        engine=DatasetEngine.LOCAL,
    )

    old = dataset.apply_annotations(
        table, AnnotationJoinConfig(release="r-old")
    )
    assert old.annotation_releases() == ["r-old"]

    new = old.apply_annotations(
        table, AnnotationJoinConfig(release="r-new")
    )
    items = new.collect()
    assert [item.id for item in items] == ["s1"]
    s1 = items[0]
    # Old label/weight/release must be fully replaced, not merged.
    assert s1.metadata["label"] == "feline"
    assert s1.metadata["weight"] == 0.1
    assert s1.metadata[METADATA_ANNOTATION_KEY]["release"] == "r-new"
    assert new.annotation_releases() == ["r-new"]

    # And the rewritten state persists through a JSON round trip.
    path = tmp_path / "reapplied.jsonl"
    new.write_json(str(path))
    reloaded = load_dataset(str(path), engine=DatasetEngine.LOCAL)
    assert reloaded.annotation_releases() == ["r-new"]
    assert reloaded.collect()[0].metadata["label"] == "feline"
