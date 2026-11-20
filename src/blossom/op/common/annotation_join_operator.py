"""Operator joining a dataset with a versioned annotation release.

Unlike :meth:`~blossom.dataframe.dataframe.DataFrame.add_metadata`, which only
merges extra keys and therefore preserves stale labels, this operator resolves
the newest in-release revision of every sample and rewrites the training
representation: labels and weights are replaced and excluded samples are
filtered out of the dataset population.
"""

from __future__ import annotations

from typing import Optional

from blossom.annotation.association import (
    AnnotationAssociationResult,
    AnnotationJoinConfig,
    AnnotationOrphanPolicy,
    AnnotationReport,
    apply_annotations,
)
from blossom.annotation.record import AnnotationRecord
from blossom.annotation.table import VersionedAnnotationTable
from blossom.dataframe.dataframe import DataFrame
from blossom.dataframe.local_dataframe import LocalDataFrame
from blossom.log import logger
from blossom.op.operator import Operator
from blossom.schema.schema import Schema


class AnnotationJoinOperator(Operator):
    """Apply a versioned annotation release to a dataset.

    Args:
        table: A :class:`VersionedAnnotationTable` holding the full revision
            history of the external annotations.
        config: Join configuration (release selection, orphan/unannotated
            policies, weight defaults, ...).

    The :class:`~blossom.annotation.AnnotationReport` of the last run is
    available as :attr:`report` after :meth:`process` returns. On the local
    engine orphan IDs are computed against the whole dataset; on distributed
    engines the configured orphan policy is enforced per partition and a
    warning is logged because a fully global check requires a collect.
    """

    def __init__(
        self,
        table: VersionedAnnotationTable,
        config: Optional[AnnotationJoinConfig] = None,
    ) -> None:
        super().__init__()
        self.table = table
        self.config = config or AnnotationJoinConfig()
        self.report: Optional[AnnotationReport] = None

    def _resolved_records(self) -> dict[str, AnnotationRecord]:
        resolved = self.table.resolve_release(self.config.release)
        # Remember the actually used release so provenance and the report name
        # it even when the caller relied on a single-release table.
        if self.config.release is None:
            self.config.release = (
                next(iter(resolved.values())).release if resolved else None
            )
        return resolved

    def process(self, dataframe: DataFrame) -> DataFrame:
        resolved = self._resolved_records()

        if isinstance(dataframe, LocalDataFrame):
            return self._process_local(dataframe, resolved)
        return self._process_partitioned(dataframe, resolved)

    def _process_local(
        self,
        dataframe: LocalDataFrame,
        resolved: dict[str, AnnotationRecord],
    ) -> DataFrame:
        result: AnnotationAssociationResult = apply_annotations(
            dataframe.collect(), resolved, self.config
        )
        self.report = result.report
        self._log_report(result.report)
        return LocalDataFrame(result.schemas)

    def _process_partitioned(
        self,
        dataframe: DataFrame,
        resolved: dict[str, AnnotationRecord],
    ) -> DataFrame:
        if self.config.orphan_policy == AnnotationOrphanPolicy.ERROR:
            logger.warning(
                "AnnotationJoinOperator orphan_policy=ERROR is enforced per "
                "partition on distributed dataframes; collect the dataset and "
                "use apply_annotations directly for a global orphan check."
            )
        config = self.config

        def join_partition(items: list[Schema]) -> list[Schema]:
            return apply_annotations(items, resolved, config).schemas

        transformed = dataframe.transform(join_partition)
        # Distributed stats are best-effort; the release named here is the
        # exact annotation version carried by every rewritten item's
        # provenance metadata.
        self.report = AnnotationReport(release=config.release)
        return transformed

    def _log_report(self, report: AnnotationReport) -> None:
        logger.info(
            "AnnotationJoinOperator applied release=%s retained=%d "
            "excluded=%d unannotated=%d orphans=%d status_counts=%s",
            report.release,
            report.retained,
            report.excluded,
            report.unannotated,
            len(report.orphan_ids),
            report.by_status,
        )
        if report.orphan_ids:
            logger.warning(
                "annotation table contains %d orphan sample id(s): %s",
                len(report.orphan_ids),
                report.orphan_ids[:10],
            )
