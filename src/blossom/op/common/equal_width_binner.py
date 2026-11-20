import json
import math
import numbers
from typing import Any, Callable, Optional, Union

import numpy as np

from blossom.dataframe.dataframe import DataFrame
from blossom.op.operator import Operator
from blossom.schema.schema import Schema


DEFAULT_BIN_FIELD = "bin"
DEFAULT_BIN_LABEL_FIELD = "bin_label"

NULL_BIN_INDEX = -1

ON_CONSTANT_SINGLE_BIN = "single_bin"
ON_CONSTANT_ERROR = "error"

OUT_OF_RANGE_CLAMP = "clamp"
OUT_OF_RANGE_ERROR = "error"

NON_FINITE_ERROR = "error"
NON_FINITE_NULL = "null"

_STATE_VERSION = 1


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and math.isfinite(float(value))


class EqualWidthBinner(Operator):
    """
    Assign numeric values to equal-width bins under an explicit fit/transform
    contract.

    `fit()` computes the bin edges and labels exactly once, from the training
    dataframe only. `transform()` applies that frozen state to any dataframe
    (validation, test, or new production data) without refitting, so a bin
    index keeps the same meaning across splits. `process()` fits on first use
    and only transforms afterwards, which keeps single-dataset pipelines
    working while guaranteeing that data processed later (e.g. a validation
    set) never reshapes the fitted training boundaries.

    The fitted state is serializable: use `state_dict()`/`load_state_dict()`
    or `save_state()`/`load_state()` to persist the frozen bin edges and
    labels and restore them into another instance or process.

    Edge cases are handled explicitly:
    - Out-of-range values (`out_of_range`): "clamp" (default) assigns them to
      the nearest edge bin, "error" raises.
    - Constant columns (`on_constant`): "single_bin" (default) collapses the
      fitted state to a single degenerate bin, "error" raises at fit time.
    - Non-finite values (`non_finite`): NaN/inf are ignored when deriving the
      range at fit time; at transform time "error" (default) raises, "null"
      assigns bin index -1 with a None label.
    """

    def __init__(
        self,
        func: Callable[[Schema], Union[int, float]],
        num_bins: int = 10,
        labels: Optional[list[Any]] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        bin_field: str = DEFAULT_BIN_FIELD,
        bin_label_field: str = DEFAULT_BIN_LABEL_FIELD,
        on_constant: str = ON_CONSTANT_SINGLE_BIN,
        out_of_range: str = OUT_OF_RANGE_CLAMP,
        non_finite: str = NON_FINITE_ERROR,
    ):
        super().__init__()
        if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins < 1:
            raise ValueError(f"num_bins must be a positive integer, got {num_bins!r}")
        if labels is not None and len(labels) != num_bins:
            raise ValueError(f"Number of labels must match number of bins: {num_bins}")
        if on_constant not in (ON_CONSTANT_SINGLE_BIN, ON_CONSTANT_ERROR):
            raise ValueError(
                f"on_constant must be '{ON_CONSTANT_SINGLE_BIN}' or "
                f"'{ON_CONSTANT_ERROR}', got {on_constant!r}"
            )
        if out_of_range not in (OUT_OF_RANGE_CLAMP, OUT_OF_RANGE_ERROR):
            raise ValueError(
                f"out_of_range must be '{OUT_OF_RANGE_CLAMP}' or "
                f"'{OUT_OF_RANGE_ERROR}', got {out_of_range!r}"
            )
        if non_finite not in (NON_FINITE_ERROR, NON_FINITE_NULL):
            raise ValueError(
                f"non_finite must be '{NON_FINITE_ERROR}' or "
                f"'{NON_FINITE_NULL}', got {non_finite!r}"
            )
        for bound, name in ((range_min, "range_min"), (range_max, "range_max")):
            if bound is not None and not _is_finite_number(bound):
                raise ValueError(f"{name} must be a finite number, got {bound!r}")
        if range_min is not None and range_max is not None and range_min > range_max:
            raise ValueError(
                f"range_min ({range_min}) must not exceed range_max ({range_max})"
            )

        self.func = func
        self.num_bins = num_bins
        self.labels = labels
        self.range_min = range_min
        self.range_max = range_max
        self.bin_field = bin_field
        self.bin_label_field = bin_label_field
        self.on_constant = on_constant
        self.out_of_range = out_of_range
        self.non_finite = non_finite

        self._bin_edges: Optional[list[float]] = None
        self._bin_labels: Optional[list[Any]] = None
        self._constant: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether this binner holds a frozen fitted state."""
        return self._bin_edges is not None

    @property
    def bin_edges(self) -> list[float]:
        """The frozen bin edges (num_bins + 1 entries, or 2 for a constant column)."""
        self._require_fitted()
        assert self._bin_edges is not None
        return list(self._bin_edges)

    @property
    def bin_labels(self) -> list[Any]:
        """The frozen bin labels (one per bin)."""
        self._require_fitted()
        assert self._bin_labels is not None
        return list(self._bin_labels)

    def fit(self, dataframe: DataFrame) -> "EqualWidthBinner":
        """
        Compute the bin edges and labels from a (training) dataframe and
        freeze them. Always refits: call this only on training data. Any
        object with the DataFrame protocol works, including Dataset.

        Non-finite values are ignored when deriving the range from data.

        Args:
            dataframe: Training dataframe to derive the bin range from

        Returns:
            This binner, now fitted

        Raises:
            ValueError: If no finite values are available, the derived range
                is empty, or the column is constant and
                on_constant="error"
        """
        range_min, range_max = self._resolve_range(dataframe)
        if range_min > range_max:
            raise ValueError(
                f"Cannot fit EqualWidthBinner: empty range "
                f"[{range_min}, {range_max}]"
            )
        if range_min == range_max:
            if self.on_constant == ON_CONSTANT_ERROR:
                raise ValueError(
                    f"Cannot fit EqualWidthBinner: the target column is "
                    f"constant ({range_min}); pass on_constant="
                    f"'{ON_CONSTANT_SINGLE_BIN}' to collapse it to one bin"
                )
            self._bin_edges = [range_min, range_max]
            self._bin_labels = (
                [self.labels[0]] if self.labels is not None else [range_min]
            )
            self._constant = True
        else:
            bin_edges = [
                float(edge)
                for edge in np.linspace(
                    range_min, range_max, self.num_bins + 1
                ).tolist()
            ]
            self._bin_edges = bin_edges
            self._bin_labels = self._gen_bin_labels(bin_edges)
            self._constant = False
        return self

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Apply the frozen bin edges and labels to a dataframe, writing the bin
        index and label into each item's metadata. Never refits: validation
        or new data cannot change the training boundaries.

        Args:
            dataframe: Dataframe to bin

        Returns:
            Dataframe with bin metadata added

        Raises:
            ValueError: If the binner is not fitted, or a value violates the
                configured out_of_range/non_finite policy
        """
        self._require_fitted()
        bin_edges = self._bin_edges
        bin_labels = self._bin_labels
        assert bin_edges is not None and bin_labels is not None

        def _map_item(item: Schema) -> Schema:
            value = self.func(item)
            bin_index = self._assign_bin_index(value, bin_edges)
            item.metadata[self.bin_field] = bin_index
            item.metadata[self.bin_label_field] = (
                bin_labels[bin_index] if bin_index != NULL_BIN_INDEX else None
            )
            return item

        return dataframe.map(_map_item)

    def fit_transform(self, dataframe: DataFrame) -> DataFrame:
        """Fit on the given (training) dataframe, then transform it."""
        return self.fit(dataframe).transform(dataframe)

    def process(self, dataframe: DataFrame) -> DataFrame:
        """
        Pipeline entry point. Fits on first use (so single-dataset pipelines
        keep working), then only applies the frozen state: executing this
        operator on a validation dataset after a training dataset reuses the
        training boundaries instead of refitting.
        """
        if not self.is_fitted:
            self.fit(dataframe)
        return self.transform(dataframe)

    def state_dict(self) -> dict[str, Any]:
        """
        Export the frozen bin edges and labels as a JSON-serializable dict.

        Returns:
            The fitted state

        Raises:
            ValueError: If the binner is not fitted
        """
        self._require_fitted()
        assert self._bin_edges is not None and self._bin_labels is not None
        return {
            "version": _STATE_VERSION,
            "bin_edges": list(self._bin_edges),
            "bin_labels": list(self._bin_labels),
            "constant": self._constant,
            "bin_field": self.bin_field,
            "bin_label_field": self.bin_label_field,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "EqualWidthBinner":
        """
        Restore a frozen state produced by `state_dict()`, replacing any
        currently fitted state.

        Args:
            state: Serialized state dict

        Returns:
            This binner, now fitted with the restored state

        Raises:
            ValueError: If the state is malformed or its labels conflict with
                the labels passed to the constructor
        """
        version = state.get("version", _STATE_VERSION)
        if version != _STATE_VERSION:
            raise ValueError(f"Unsupported binner state version: {version!r}")
        bin_edges = state.get("bin_edges")
        bin_labels = state.get("bin_labels")
        if (
            not isinstance(bin_edges, list)
            or len(bin_edges) < 2
            or not all(_is_finite_number(edge) for edge in bin_edges)
        ):
            raise ValueError(
                "Invalid binner state: 'bin_edges' must be a list of at "
                "least two finite numbers"
            )
        edges = [float(edge) for edge in bin_edges]
        if any(edges[i] > edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError(
                "Invalid binner state: 'bin_edges' must be sorted in "
                "non-decreasing order"
            )
        if not isinstance(bin_labels, list) or len(bin_labels) != len(edges) - 1:
            raise ValueError(
                "Invalid binner state: 'bin_labels' must contain exactly "
                "len(bin_edges) - 1 entries"
            )
        if self.labels is not None and list(self.labels)[: len(bin_labels)] != list(
            bin_labels
        ):
            raise ValueError(
                "Serialized bin labels are inconsistent with the labels "
                "passed to the constructor"
            )
        self._bin_edges = edges
        self._bin_labels = list(bin_labels)
        self._constant = bool(state.get("constant", False))
        return self

    def save_state(self, path: str) -> None:
        """
        Write the frozen bin edges and labels to a JSON file.

        Args:
            path: Destination file path
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    def load_state(self, path: str) -> "EqualWidthBinner":
        """
        Load a frozen state written by `save_state()`.

        Args:
            path: State file path

        Returns:
            This binner, now fitted with the restored state
        """
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return self.load_state_dict(state)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(
                "EqualWidthBinner is not fitted yet; call fit() on the "
                "training dataframe (or load a saved state) first"
            )

    def _resolve_range(self, dataframe: DataFrame) -> tuple[float, float]:
        range_min = self.range_min
        range_max = self.range_max
        if range_min is None or range_max is None:
            finite_df = (
                dataframe.filter(lambda item: _is_finite_number(self.func(item)))
                .cache()
            )
            try:
                if range_min is None:
                    range_min = finite_df.min(self.func)
                if range_max is None:
                    range_max = finite_df.max(self.func)
            except ValueError as e:
                raise ValueError(
                    "Cannot fit EqualWidthBinner: no finite values available "
                    "to derive the bin range"
                ) from e
        assert range_min is not None and range_max is not None
        return float(range_min), float(range_max)

    def _gen_bin_labels(self, bin_edges: list[float]) -> list[Any]:
        if self.labels is not None:
            return list(self.labels)
        return [bin_edges[i] for i in range(len(bin_edges) - 1)]

    def _assign_bin_index(
        self, value: Union[int, float], bin_edges: list[float]
    ) -> int:
        if not _is_finite_number(value):
            if self.non_finite == NON_FINITE_NULL:
                return NULL_BIN_INDEX
            raise ValueError(
                f"Cannot bin non-finite value {value!r}; set non_finite="
                f"'{NON_FINITE_NULL}' to map such values to bin "
                f"{NULL_BIN_INDEX} with a null label"
            )
        v = float(value)
        range_min, range_max = bin_edges[0], bin_edges[-1]
        if v < range_min or v > range_max:
            if self.out_of_range == OUT_OF_RANGE_ERROR:
                raise ValueError(
                    f"Value {v} falls outside the fitted range "
                    f"[{range_min}, {range_max}]; set out_of_range="
                    f"'{OUT_OF_RANGE_CLAMP}' to clamp such values to the "
                    f"edge bins"
                )
            v = min(max(v, range_min), range_max)
        index = int(np.digitize(v, bin_edges) - 1)
        return max(0, min(index, len(bin_edges) - 2))
