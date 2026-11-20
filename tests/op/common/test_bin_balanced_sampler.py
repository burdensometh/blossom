from __future__ import annotations

import pytest

from blossom.dataframe.local_dataframe import LocalDataFrame
from blossom.op.common.bin_balanced_sampler import BinBalancedSampler
from blossom.op.common.equal_width_binner import EqualWidthBinner
from blossom.schema.row_schema import RowSchema


def _value(item: RowSchema) -> float:
    return item.data["value"]


def _df(values: list[float]) -> LocalDataFrame:
    return LocalDataFrame([RowSchema(data={"value": v}) for v in values])


def _fitted_binner(
    values: list[float], num_bins: int = 2, **kwargs
) -> EqualWidthBinner:
    return EqualWidthBinner(func=_value, num_bins=num_bins, **kwargs).fit(_df(values))


def _bin_counts(items: list[RowSchema]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in items:
        bin_index = item.metadata["bin"]
        counts[bin_index] = counts.get(bin_index, 0) + 1
    return counts


def test_downsample_changes_population_and_balances_bins() -> None:
    values = [0] * 6 + [10] * 2
    binner = _fitted_binner(values)
    binned = binner.transform(_df(values))

    sampler = BinBalancedSampler(binner, seed=42)
    result = sampler.process(binned).collect()

    # population actually shrank to 2 per bin, not just metadata recorded
    assert len(result) == 4
    assert _bin_counts(result) == {0: 2, 1: 2}
    assert sampler.bin_counts_ == {0: 6, 1: 2}
    # output is a subset of the input population
    input_ids = {item.id for item in binned.collect()}
    assert {item.id for item in result} <= input_ids


def test_downsample_writes_optional_sample_weights() -> None:
    values = [0] * 6 + [10] * 2
    binner = _fitted_binner(values)
    binned = binner.transform(_df(values))

    sampler = BinBalancedSampler(binner, weight_field="sample_weight", seed=1)
    result = sampler.process(binned).collect()

    for item in result:
        if item.metadata["bin"] == 0:
            assert item.metadata["sample_weight"] == pytest.approx(3.0)
        else:
            assert item.metadata["sample_weight"] == pytest.approx(1.0)
    # per-bin weights sum back to the original bin population
    for bin_index, original_count in ((0, 6), (1, 2)):
        total = sum(
            item.metadata["sample_weight"]
            for item in result
            if item.metadata["bin"] == bin_index
        )
        assert total == pytest.approx(original_count)


def test_no_weight_field_means_no_weight_metadata() -> None:
    values = [0, 0, 10]
    binner = _fitted_binner(values)
    binned = binner.transform(_df(values))
    result = BinBalancedSampler(binner, seed=0).process(binned).collect()
    assert all("sample_weight" not in item.metadata for item in result)
    assert all("weight" not in item.metadata for item in result)


def test_samples_per_bin_overrides_min_count() -> None:
    values = [0] * 6 + [10] * 4
    binner = _fitted_binner(values)
    binned = binner.transform(_df(values))

    result = BinBalancedSampler(binner, samples_per_bin=3, seed=0).process(
        binned
    ).collect()
    assert _bin_counts(result) == {0: 3, 1: 3}


def test_upsample_repeats_minority_bins_with_fresh_ids() -> None:
    values = [0] * 2 + [10] * 5
    binner = _fitted_binner(values)
    binned = binner.transform(_df(values))

    sampler = BinBalancedSampler(binner, mode="upsample", weight_field="w", seed=3)
    result = sampler.process(binned).collect()

    assert _bin_counts(result) == {0: 5, 1: 5}
    # duplicates are distinct items with unique ids
    ids = [item.id for item in result]
    assert len(set(ids)) == len(ids)
    for item in result:
        if item.metadata["bin"] == 0:
            assert item.metadata["w"] == pytest.approx(2 / 5)
        else:
            assert item.metadata["w"] == pytest.approx(1.0)


def test_upsample_with_explicit_samples_per_bin() -> None:
    values = [0] * 2 + [10] * 5
    binner = _fitted_binner(values)
    binned = binner.transform(_df(values))

    result = BinBalancedSampler(
        binner, mode="upsample", samples_per_bin=3, seed=0
    ).process(binned).collect()
    assert _bin_counts(result) == {0: 3, 1: 3}


def test_sampling_is_deterministic_for_a_seed() -> None:
    values = [0] * 6 + [10] * 4
    binner = _fitted_binner(values)
    binned = binner.transform(_df(values))

    first = BinBalancedSampler(binner, seed=7).process(binned).collect()
    second = BinBalancedSampler(binner, seed=7).process(binned).collect()
    assert [item.id for item in first] == [item.id for item in second]


def test_requires_fitted_binner() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2)
    sampler = BinBalancedSampler(binner)
    with pytest.raises(ValueError, match="fitted"):
        sampler.process(_df([0, 1])).collect()


def test_missing_bin_metadata_raises() -> None:
    binner = _fitted_binner([0, 10])
    sampler = BinBalancedSampler(binner)
    with pytest.raises(ValueError, match="bin"):
        sampler.process(_df([5])).collect()


def test_null_bin_items_are_excluded() -> None:
    values = [0, 10, float("nan")]
    binner = EqualWidthBinner(func=_value, num_bins=2, non_finite="null")
    binned = binner.fit_transform(_df(values))

    result = BinBalancedSampler(binner, seed=0).process(binned).collect()
    assert len(result) == 2
    assert all(item.metadata["bin"] != -1 for item in result)


def test_empty_bins_are_ignored() -> None:
    values = [0, 1, 2]  # bin 1 of a 4-bin fit stays empty
    binner = _fitted_binner(values, num_bins=4)
    binned = binner.transform(_df(values))

    sampler = BinBalancedSampler(binner, seed=0)
    result = sampler.process(binned).collect()
    assert sampler.bin_counts_ == {0: 1, 1: 0, 2: 1, 3: 1}
    assert _bin_counts(result) == {0: 1, 2: 1, 3: 1}


def test_all_items_unbinned_yields_empty_output() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2, non_finite="null")
    binner.fit(_df([0, 10]))
    binned = binner.transform(_df([float("nan"), float("inf")]))
    result = BinBalancedSampler(binner, seed=0).process(binned).collect()
    assert result == []


def test_constructor_validation() -> None:
    binner = _fitted_binner([0, 1])
    with pytest.raises(TypeError, match="EqualWidthBinner"):
        BinBalancedSampler(binner="not-a-binner")
    with pytest.raises(ValueError, match="mode"):
        BinBalancedSampler(binner, mode="bogus")
    with pytest.raises(ValueError, match="samples_per_bin"):
        BinBalancedSampler(binner, samples_per_bin=0)
