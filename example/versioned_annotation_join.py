"""Versioned annotation join example.

External annotation tables are revised continuously per sample ID. A plain
``add_metadata`` merge would keep labels that have since been corrected or
revoked. This example shows how to join a dataset against a *released*
annotation table so that:

* only the newest in-release revision of every sample is selected;
* revoked/rejected samples leave the training population (a revocation never
  falls back to an earlier approved revision);
* labels and training weights are physically replaced, not just audited;
* the applied release/revision is recorded on every item and survives
  JSON/Parquet write and re-read.

Run with:
    poetry run python example/versioned_annotation_join.py
"""

from blossom import (
    AnnotationJoinConfig,
    AnnotationOrphanPolicy,
    DatasetEngine,
    ReviewStatus,
    VersionedAnnotationTable,
    create_dataset,
)
from blossom.schema import TextSchema

# Two releases of the same external annotation table. Every revision is kept;
# the join selects the newest one inside the requested release.
annotation_table = VersionedAnnotationTable.from_dicts(
    [
        # Release 2026-10: s1 approved as "cat", s2 approved as "dog".
        {
            "sample_id": "s1",
            "release": "2026-10",
            "revision": 1,
            "status": ReviewStatus.APPROVED,
            "label": "cat",
            "weight": 1.0,
        },
        {
            "sample_id": "s2",
            "release": "2026-10",
            "revision": 1,
            "status": ReviewStatus.APPROVED,
            "label": "dog",
        },
        # Release 2026-11: s1 is corrected to "feline" with a down-weight,
        # s2's earlier approval is revoked, and s3 arrives as pending review.
        {
            "sample_id": "s1",
            "release": "2026-11",
            "revision": 1,
            "status": ReviewStatus.APPROVED,
            "label": "feline",
            "weight": 0.25,
        },
        {
            "sample_id": "s2",
            "release": "2026-11",
            "revision": 1,
            "status": ReviewStatus.REVOKED,
        },
        {
            "sample_id": "s3",
            "release": "2026-11",
            "revision": 1,
            "status": ReviewStatus.PENDING,
            "label": "bird",
        },
    ]
)

dataset = create_dataset(
    [
        TextSchema(id="s1", content="photo one"),
        TextSchema(id="s2", content="photo two"),
        TextSchema(id="s3", content="photo three"),
        TextSchema(id="s4", content="photo four"),  # never annotated
    ],
    engine=DatasetEngine.LOCAL,
)

# Orphan IDs (annotations pointing at unknown samples) raise instead of being
# silently ignored; unannotated samples stay in the dataset by default but
# lose any label/weight left by an older release.
config = AnnotationJoinConfig(
    release="2026-11",
    orphan_policy=AnnotationOrphanPolicy.IGNORE,
    include_pending=False,
)

joined = dataset.apply_annotations(annotation_table, config)

print("applied release:", joined.annotation_releases())
for item in joined.collect():
    print(item.id, item.metadata)

# Persist and re-read: the annotation version used to build the training data
# travels with every row, so downstream consumers can always tell which
# release produced the stored labels and weights.
joined.write_json("example/data/versioned_annotated.jsonl")
reloaded = create_dataset(engine=DatasetEngine.LOCAL).read_json(
    "example/data/versioned_annotated.jsonl"
)
print("reloaded release:", reloaded.annotation_releases())
