from __future__ import annotations

import json

import pytest

from blossom.dataframe.local_dataframe import LocalDataFrame
from blossom.dataset import create_dataset
from blossom.op.common.equal_width_binner import EqualWidthBinner
from blossom.schema.row_schema import RowSchema


def _value(item: RowSchema) -> float:
    return item.data["value"]


def _df(values: list[float]) -> LocalDataFrame:
    return LocalDataFrame([RowSchema(data={"value": v}) for v in values])


def _bins_by_value(items: list[RowSchema]) -> dict[float, int]:
    return {item.data["value"]: item.metadata["bin"] for item in items}


def test_equal_width_binner_metadata() -> None:
    data = [RowSchema(data={"value": v}) for v in [0, 5, 9]]
    df = LocalDataFrame(data)
    op = EqualWidthBinner(func=lambda s: s.data["value"], num_bins=3)
    result = op.process(df).collect()
    for item in result:
        assert "bin" in item.metadata
        assert "bin_label" in item.metadata
        assert item.metadata["bin"] in {0, 1, 2}
        assert item.metadata["bin_label"] in {0.0, 3.0, 6.0}


def test_fit_freezes_boundaries_for_validation_data() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=3)
    binner.fit(_df([0, 3, 6, 9]))
    assert binner.is_fitted
    assert binner.bin_edges == [0.0, 3.0, 6.0, 9.0]
    assert binner.bin_labels == [0.0, 3.0, 6.0]

    result = binner.transform(_df([100, -100, 4])).collect()
    bins = _bins_by_value(result)
    assert bins[100] == 2  # clamped to the top bin
    assert bins[-100] == 0  # clamped to the bottom bin
    assert bins[4] == 1
    # validation data must not reshape the frozen training boundaries
    assert binner.bin_edges == [0.0, 3.0, 6.0, 9.0]


def test_process_fits_once_and_never_refits() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2)
    binner.process(_df([0, 10])).collect()
    edges_after_train = binner.bin_edges

    validation = binner.process(_df([1000, -1000])).collect()
    assert binner.bin_edges == edges_after_train
    bins = _bins_by_value(validation)
    assert bins[1000] == 1
    assert bins[-1000] == 0


def test_fit_transform_matches_separate_fit_and_transform() -> None:
    values = [0, 1, 2, 3, 4]
    combined = EqualWidthBinner(func=_value, num_bins=2)
    separate = EqualWidthBinner(func=_value, num_bins=2)
    combined_out = combined.fit_transform(_df(values)).collect()
    separate_out = separate.fit(_df(values)).transform(_df(values)).collect()
    assert [i.metadata["bin"] for i in combined_out] == [
        i.metadata["bin"] for i in separate_out
    ]


def test_transform_requires_fit() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2)
    with pytest.raises(ValueError, match="not fitted"):
        binner.transform(_df([1])).collect()
    with pytest.raises(ValueError, match="not fitted"):
        binner.state_dict()


def test_state_dict_round_trip_is_json_serializable() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=4, labels=["a", "b", "c", "d"])
    binner.fit(_df([0, 1, 2, 3, 4]))

    state = binner.state_dict()
    json.dumps(state)  # must be JSON-serializable

    restored = EqualWidthBinner(func=_value, num_bins=4, labels=["a", "b", "c", "d"])
    restored.load_state_dict(json.loads(json.dumps(state)))
    assert restored.is_fitted
    assert restored.bin_edges == binner.bin_edges
    assert restored.bin_labels == binner.bin_labels

    probe = [0.5, 2.5, 3.5, 100, -100]
    expected = [i.metadata["bin"] for i in binner.transform(_df(probe)).collect()]
    actual = [i.metadata["bin"] for i in restored.transform(_df(probe)).collect()]
    assert actual == expected
    labels = [i.metadata["bin_label"] for i in restored.transform(_df(probe)).collect()]
    assert labels == ["a", "c", "d", "d", "a"]


def test_save_and_load_state(tmp_path) -> None:
    path = str(tmp_path / "binner_state.json")
    binner = EqualWidthBinner(func=_value, num_bins=3)
    binner.fit(_df([1, 2, 3]))
    binner.save_state(path)

    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["bin_edges"] == [1.0, 1.0 + 2.0 / 3.0, 1.0 + 4.0 / 3.0, 3.0]

    restored = EqualWidthBinner(func=_value, num_bins=3).load_state(path)
    assert restored.is_fitted
    assert restored.bin_edges == binner.bin_edges
    assert restored.bin_labels == binner.bin_labels


def test_load_state_dict_validation() -> None:
    fitted = EqualWidthBinner(func=_value, num_bins=2, labels=["x", "y"])
    fitted.fit(_df([0, 1]))
    state = fitted.state_dict()

    with pytest.raises(ValueError, match="inconsistent with the labels"):
        EqualWidthBinner(func=_value, num_bins=2, labels=["p", "q"]).load_state_dict(
            state
        )
    with pytest.raises(ValueError, match="bin_edges"):
        EqualWidthBinner(func=_value).load_state_dict({"bin_edges": [0.0]})
    with pytest.raises(ValueError, match="bin_edges"):
        EqualWidthBinner(func=_value).load_state_dict(
            {"bin_edges": [2.0, 1.0, 3.0], "bin_labels": ["a", "b"]}
        )
    with pytest.raises(ValueError, match="bin_labels"):
        EqualWidthBinner(func=_value).load_state_dict(
            {"bin_edges": [0.0, 1.0], "bin_labels": ["a", "b"]}
        )
    with pytest.raises(ValueError, match="version"):
        EqualWidthBinner(func=_value).load_state_dict(
            {"version": 999, "bin_edges": [0.0, 1.0], "bin_labels": ["a"]}
        )


def test_out_of_range_error_policy() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2, out_of_range="error")
    binner.fit(_df([0, 10]))
    with pytest.raises(ValueError, match="outside the fitted range"):
        binner.transform(_df([10.5])).collect()
    with pytest.raises(ValueError, match="outside the fitted range"):
        binner.transform(_df([-0.5])).collect()
    # in-range values still work
    assert binner.transform(_df([5])).collect()[0].metadata["bin"] == 1


def test_constant_column_collapses_to_single_bin() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=5)
    binner.fit(_df([7, 7, 7]))
    assert binner.bin_edges == [7.0, 7.0]
    assert binner.bin_labels == [7.0]

    result = binner.transform(_df([7, 7])).collect()
    assert all(item.metadata["bin"] == 0 for item in result)
    assert all(item.metadata["bin_label"] == 7.0 for item in result)

    # constant state survives serialization
    restored = EqualWidthBinner(func=_value, num_bins=5).load_state_dict(
        binner.state_dict()
    )
    assert restored.transform(_df([7])).collect()[0].metadata["bin"] == 0


def test_constant_column_uses_first_custom_label() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=3, labels=["lo", "mid", "hi"])
    binner.fit(_df([2, 2]))
    assert binner.bin_labels == ["lo"]

    # the collapsed state round-trips into an instance with the same labels
    restored = EqualWidthBinner(func=_value, num_bins=3, labels=["lo", "mid", "hi"])
    restored.load_state_dict(binner.state_dict())
    assert restored.bin_labels == ["lo"]
    assert restored.transform(_df([2])).collect()[0].metadata["bin_label"] == "lo"


def test_constant_column_error_policy() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=4, on_constant="error")
    with pytest.raises(ValueError, match="constant"):
        binner.fit(_df([3, 3, 3]))


def test_non_finite_ignored_at_fit_and_rejected_at_transform() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2)
    binner.fit(_df([0, 10, float("nan"), float("inf"), float("-inf")]))
    assert binner.bin_edges == [0.0, 5.0, 10.0]

    with pytest.raises(ValueError, match="non-finite"):
        binner.transform(_df([float("nan")])).collect()
    with pytest.raises(ValueError, match="non-finite"):
        binner.transform(_df([float("inf")])).collect()


def test_non_finite_null_policy() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2, non_finite="null")
    binner.fit(_df([0, 10]))
    result = binner.transform(_df([float("nan"), 5])).collect()
    assert result[0].metadata["bin"] == -1
    assert result[0].metadata["bin_label"] is None
    assert result[1].metadata["bin"] == 1
    assert result[1].metadata["bin_label"] == 5.0


def test_fit_without_finite_values_raises() -> None:
    with pytest.raises(ValueError, match="no finite values"):
        EqualWidthBinner(func=_value).fit(_df([float("nan"), float("inf")]))
    with pytest.raises(ValueError, match="no finite values"):
        EqualWidthBinner(func=_value).fit(_df([]))


def test_explicit_range_is_not_expanded_by_data() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2, range_min=0.0, range_max=1.0)
    binner.fit(_df([0, 100]))
    assert binner.bin_edges == [0.0, 0.5, 1.0]
    bins = _bins_by_value(binner.transform(_df([0.4, 0.6, 100])).collect())
    assert bins[0.4] == 0
    assert bins[0.6] == 1
    assert bins[100] == 1  # clamped, range stays as configured


def test_partial_explicit_range() -> None:
    binner = EqualWidthBinner(func=_value, num_bins=2, range_min=0.0)
    binner.fit(_df([5, 15]))
    assert binner.bin_edges == [0.0, 7.5, 15.0]


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="num_bins"):
        EqualWidthBinner(func=_value, num_bins=0)
    with pytest.raises(ValueError, match="Number of labels"):
        EqualWidthBinner(func=_value, num_bins=2, labels=["only"])
    with pytest.raises(ValueError, match="on_constant"):
        EqualWidthBinner(func=_value, on_constant="bogus")
    with pytest.raises(ValueError, match="out_of_range"):
        EqualWidthBinner(func=_value, out_of_range="bogus")
    with pytest.raises(ValueError, match="non_finite"):
        EqualWidthBinner(func=_value, non_finite="bogus")
    with pytest.raises(ValueError, match="range_min"):
        EqualWidthBinner(func=_value, range_min=float("nan"))
    with pytest.raises(ValueError, match="must not exceed"):
        EqualWidthBinner(func=_value, range_min=2.0, range_max=1.0)


def test_dataset_execute_fits_on_train_and_freezes_for_validation(
    dummy_context,
) -> None:
    train_ds = create_dataset(
        [RowSchema(data={"value": v}) for v in [0, 5, 10]], context=dummy_context
    )
    val_ds = create_dataset(
        [RowSchema(data={"value": v}) for v in [7, 100]], context=dummy_context
    )

    binner = EqualWidthBinner(func=_value, num_bins=2)
    train_out = train_ds.execute([binner]).collect()
    edges_after_train = binner.bin_edges
    assert [i.metadata["bin"] for i in train_out] == [0, 1, 1]

    val_out = val_ds.execute([binner]).collect()
    # validation execution reuses the frozen training state
    assert binner.bin_edges == edges_after_train
    bins = _bins_by_value(val_out)
    assert bins[7] == 1
    assert bins[100] == 1
