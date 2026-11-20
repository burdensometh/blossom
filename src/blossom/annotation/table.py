"""Versioned annotation table: release-aware storage and record resolution."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable, Optional

from blossom.annotation.record import (
    AnnotationRecord,
    DuplicateAnnotationError,
    ReviewStatus,
)


class VersionedAnnotationTable:
    """A collection of :class:`AnnotationRecord` objects with release history.

    The table keeps every revision ever received.  When a dataset is joined
    against a release, :meth:`resolve_release` returns exactly one record per
    sample ID: the newest revision within that release.  Because revoked and
    rejected revisions participate in the same "newest wins" selection, a
    later revocation always removes the sample instead of falling back to an
    earlier approved revision.

    Duplicate records are rejected while the table is built so that ambiguity
    fails loudly instead of silently picking one annotation over another.
    """

    def __init__(self, records: Optional[Iterable[AnnotationRecord]] = None):
        self._records: list[AnnotationRecord] = []
        self._annotation_ids: set[str] = set()
        for record in records or []:
            self.add_record(record)

    @classmethod
    def from_dicts(
        cls,
        rows: Iterable[dict[str, Any]],
        release: Optional[str] = None,
        field_mapping: Optional[dict[str, str]] = None,
    ) -> "VersionedAnnotationTable":
        """Build a table from raw annotation-table rows (see
        :meth:`AnnotationRecord.from_dict`)."""
        table = cls()
        for row in rows:
            table.add_record(
                AnnotationRecord.from_dict(
                    row, release=release, field_mapping=field_mapping
                )
            )
        return table

    @classmethod
    def from_json(
        cls,
        path: str,
        release: Optional[str] = None,
        field_mapping: Optional[dict[str, str]] = None,
    ) -> "VersionedAnnotationTable":
        """Load a table from a JSON or JSONL file.

        A top-level JSON array and newline-delimited JSON objects are both
        supported, matching the dataset's own JSON I/O conventions.
        """
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return cls()
        rows: list[dict[str, Any]] = []
        if text.lstrip().startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(
                f"annotation file {path} must contain a list of object rows"
            )
        return cls.from_dicts(rows, release=release, field_mapping=field_mapping)

    def add_record(self, record: AnnotationRecord) -> None:
        """Add one revision, validating uniqueness constraints.

        Raises:
            DuplicateAnnotationError: If the same annotation ID is reused or
                another record with the identical
                ``(release, sample_id, revision, updated_at)`` version key
                already exists.
        """
        if record.annotation_id is not None:
            if record.annotation_id in self._annotation_ids:
                raise DuplicateAnnotationError(
                    f"duplicate annotation_id {record.annotation_id!r}"
                )
            self._annotation_ids.add(record.annotation_id)

        for existing in self._records:
            if (
                existing.release == record.release
                and existing.sample_id == record.sample_id
                and existing.version_key() == record.version_key()
            ):
                raise DuplicateAnnotationError(
                    "duplicate annotation for sample "
                    f"{record.sample_id!r} in release {record.release!r} "
                    f"at revision {record.revision}"
                )
        self._records.append(record)

    @property
    def records(self) -> list[AnnotationRecord]:
        """All stored records, in insertion order."""
        return list(self._records)

    def releases(self) -> list[str]:
        """Sorted names of every release present in the table."""
        return sorted({record.release for record in self._records})

    def sample_ids(self, release: Optional[str] = None) -> set[str]:
        """All sample IDs referenced by the table, optionally per release."""
        return {
            record.sample_id
            for record in self._records
            if release is None or record.release == release
        }

    def history(
        self, sample_id: str, release: Optional[str] = None
    ) -> list[AnnotationRecord]:
        """Every stored revision of one sample, oldest first."""
        history = [
            record
            for record in self._records
            if record.sample_id == sample_id
            and (release is None or record.release == release)
        ]
        return sorted(history, key=lambda record: record.version_key())

    def _select_release(self, release: Optional[str]) -> str:
        available = self.releases()
        if not available:
            raise ValueError("annotation table contains no records")
        if release is not None:
            if release not in available:
                raise ValueError(
                    f"unknown annotation release {release!r}; "
                    f"available releases: {available}"
                )
            return release
        if len(available) == 1:
            return available[0]
        raise ValueError(
            "multiple annotation releases are available "
            f"{available}; pass release explicitly"
        )

    def resolve_release(
        self, release: Optional[str] = None
    ) -> dict[str, AnnotationRecord]:
        """Resolve one release to its effective per-sample records.

        Returns:
            Mapping ``sample_id -> newest record`` within the release.

        Raises:
            ValueError: If the release does not exist, or no release is given
                and the table contains more than one release.
        """
        selected_release = self._select_release(release)
        grouped: dict[str, list[AnnotationRecord]] = defaultdict(list)
        for record in self._records:
            if record.release == selected_release:
                grouped[record.sample_id].append(record)

        resolved: dict[str, AnnotationRecord] = {}
        for sample_id, versions in grouped.items():
            resolved[sample_id] = max(
                versions, key=lambda record: record.version_key()
            )
        return resolved

    def status_counts(self, release: Optional[str] = None) -> dict[str, int]:
        """Count effective records of each status within a release."""
        resolved = self.resolve_release(release)
        counts = {status.value: 0 for status in ReviewStatus}
        for record in resolved.values():
            counts[record.status.value] += 1
        return counts
