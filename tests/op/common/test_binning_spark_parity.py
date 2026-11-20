from __future__ import annotations

import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession

from blossom.dataframe.dataframe import DataFrame
from blossom.dataframe.local_dataframe import LocalDataFrame
from blossom.dataframe.spark_dataframe import SparkDataFrame
from blossom.op.common.bin_balanced_sampler import BinBalancedSampler
from blossom.op.common.equal_width_binner import EqualWidthBinner
from blossom.schema.row_schema import RowSchema

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("blossom-binning-parity-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def _rows(values: list[float]) -> list[RowSchema]:
    return [RowSchema(data={"value": v}) for v in values]


def _spark_df(spark_session: SparkSession, items: list[RowSchema]) -> SparkDataFrame:
    rdd = spark_session.sparkContext.parallelize([i.to_dict() for i in items], 2)
    return SparkDataFrame(rdd, spark_session)


def test_frozen_state_identical_on_local_and_spark(spark_session: SparkSession) -> None:
    # inline lambdas so Spark workers can unpickle the closures
    local_binner = EqualWidthBinner(
        func=lambda item: item.data["value"], num_bins=3
    ).fit(LocalDataFrame(_rows([0, 3, 6, 9])))
    spark_binner = EqualWidthBinner(
        func=lambda item: item.data["value"], num_bins=3
    ).fit(_spark_df(spark_session, _rows([0, 3, 6, 9])))
    assert local_binner.bin_edges == spark_binner.bin_edges
    assert local_binner.state_dict() == spark_binner.state_dict()

    # the locally fitted state is applied to Spark validation data as-is
    local_bins = {
        item.data["value"]: item.metadata["bin"]
        for item in local_binner.transform(
            LocalDataFrame(_rows([100, -100, 4, 1]))
        ).collect()
    }
    spark_bins = {
        item.data["value"]: item.metadata["bin"]
        for item in local_binner.transform(
            _spark_df(spark_session, _rows([100, -100, 4, 1]))
        ).collect()
    }
    assert local_bins == spark_bins == {100: 2, -100: 0, 4: 1, 1: 0}
    # validation data did not reshape the training boundaries
    assert local_binner.bin_edges == [0.0, 3.0, 6.0, 9.0]


def test_serialized_state_applies_identically_on_spark(
    spark_session: SparkSession, tmp_path
) -> None:
    binner = EqualWidthBinner(func=lambda item: item.data["value"], num_bins=4)
    binner.fit(LocalDataFrame(_rows([0, 1, 2, 3, 4])))
    path = str(tmp_path / "binner.json")
    binner.save_state(path)

    restored = EqualWidthBinner(
        func=lambda item: item.data["value"], num_bins=4
    ).load_state(path)
    local_bins = [
        item.metadata["bin"]
        for item in restored.transform(
            LocalDataFrame(_rows([0.5, 2.5, 100, -100]))
        ).collect()
    ]
    spark_bins = [
        item.metadata["bin"]
        for item in restored.transform(
            _spark_df(spark_session, _rows([0.5, 2.5, 100, -100]))
        ).collect()
    ]
    assert local_bins == spark_bins


def test_sampler_selects_identical_population_on_local_and_spark(
    spark_session: SparkSession,
) -> None:
    values = [0] * 8 + [5] * 3 + [10] * 5
    items = _rows(values)
    binner = EqualWidthBinner(func=lambda item: item.data["value"], num_bins=3).fit(
        LocalDataFrame(items)
    )

    # same underlying item ids on both backends
    binned_local = binner.transform(LocalDataFrame(items))
    binned_spark = binner.transform(_spark_df(spark_session, items))

    out_local = BinBalancedSampler(binner, weight_field="w", seed=123).process(
        binned_local
    ).collect()
    out_spark = BinBalancedSampler(binner, weight_field="w", seed=123).process(
        binned_spark
    ).collect()

    assert sorted(item.id for item in out_local) == sorted(
        item.id for item in out_spark
    )
    local_weights = {item.id: item.metadata["w"] for item in out_local}
    spark_weights = {item.id: item.metadata["w"] for item in out_spark}
    assert local_weights == spark_weights
    # 3 per bin (smallest bin), weights sum back to the original population
    assert len(out_local) == 9
    for bin_index, original_count in ((0, 8), (1, 3), (2, 5)):
        total = sum(
            item.metadata["w"]
            for item in out_local
            if item.metadata["bin"] == bin_index
        )
        assert total == pytest.approx(original_count)


def test_upsample_parity_on_local_and_spark(spark_session: SparkSession) -> None:
    values = [0] * 2 + [10] * 5
    items = _rows(values)
    binner = EqualWidthBinner(func=lambda item: item.data["value"], num_bins=2).fit(
        LocalDataFrame(items)
    )

    def run(df: DataFrame) -> dict[int, int]:
        out = BinBalancedSampler(binner, mode="upsample", seed=5).process(df).collect()
        counts: dict[int, int] = {}
        for item in out:
            counts[item.metadata["bin"]] = counts.get(item.metadata["bin"], 0) + 1
        return counts

    local_counts = run(binner.transform(LocalDataFrame(items)))
    spark_counts = run(binner.transform(_spark_df(spark_session, items)))
    assert local_counts == spark_counts == {0: 5, 1: 5}
