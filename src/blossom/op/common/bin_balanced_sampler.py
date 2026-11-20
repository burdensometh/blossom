import copy
import random
import uuid
from typing import Optional

from blossom.dataframe.dataframe import DataFrame
from blossom.op.common.equal_width_binner import EqualWidthBinner
from blossom.op.operator import Operator
from blossom.schema.schema import Schema

MODE_DOWNSAMPLE = "downsample"
MODE_UPSAMPLE = "upsample"

_MISSING = object()


class BinBalancedSampler(Operator):
    """
    Rebalance a dataframe across the bins of a fitted EqualWidthBinner.

    The sampler consumes the binner's frozen state (its configured bin field
    and bin count) together with the per-item bin metadata written by
    `EqualWidthBinner.transform()`/`process()`, and changes the actual output
    population: every bin contributes the same number of items (capped by
    what a bin contains when downsampling). Items outside the fitted bin
    space (e.g. the null bin produced by non_finite="null") are excluded
    from the balanced output.

    Optionally, each emitted item receives a sample weight in its metadata
    (`weight_field`) so downstream consumers can recover the original
    distribution: weight = (items in bin) / (items emitted for bin), i.e.
    the per-bin weights always sum back to the original bin count.

    Selection is deterministic for a given `seed` and identical input items,
    regardless of the dataframe backend (Local or Spark): items are ordered
    by id within each bin before sampling. Upsampled duplicates are deep
    copies with fresh ids.
    """

    def __init__(
        self,
        binner: EqualWidthBinner,
        mode: str = MODE_DOWNSAMPLE,
        samples_per_bin: Optional[int] = None,
        weight_field: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if not isinstance(binner, EqualWidthBinner):
            raise TypeError(
                f"binner must be an EqualWidthBinner, got {type(binner).__name__}"
            )
        if mode not in (MODE_DOWNSAMPLE, MODE_UPSAMPLE):
            raise ValueError(
                f"mode must be '{MODE_DOWNSAMPLE}' or '{MODE_UPSAMPLE}', "
                f"got {mode!r}"
            )
        if samples_per_bin is not None and samples_per_bin < 1:
            raise ValueError(
                f"samples_per_bin must be a positive integer, got "
                f"{samples_per_bin!r}"
            )
        self.binner = binner
        self.mode = mode
        self.samples_per_bin = samples_per_bin
        self.weight_field = weight_field
        self.seed = seed
        self.bin_counts_: dict[int, int] = {}

    def process(self, dataframe: DataFrame) -> DataFrame:
        """
        Sample a bin-balanced subset (or superset, when upsampling) of the
        dataframe.

        Args:
            dataframe: Dataframe carrying this binner's bin metadata

        Returns:
            A new dataframe containing only the sampled items, with optional
            sample weights in metadata

        Raises:
            ValueError: If the binner is not fitted, or an item is missing
                the binner's bin metadata
        """
        if not self.binner.is_fitted:
            raise ValueError(
                "BinBalancedSampler requires a fitted EqualWidthBinner; fit "
                "the binner on the training dataset before sampling"
            )
        bin_field = self.binner.bin_field
        num_bins = len(self.binner.bin_labels)

        items_by_bin: dict[int, list[Schema]] = {b: [] for b in range(num_bins)}
        for item in dataframe.collect():
            raw_index = item.metadata.get(bin_field, _MISSING)
            if raw_index is _MISSING:
                raise ValueError(
                    f"Item {item.id} has no '{bin_field}' metadata; apply "
                    f"the binner's transform() before sampling"
                )
            try:
                bin_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= bin_index < num_bins:
                items_by_bin[bin_index].append(item)

        self.bin_counts_ = {b: len(members) for b, members in items_by_bin.items()}

        available = [count for count in self.bin_counts_.values() if count > 0]
        if not available:
            return dataframe.from_list([])

        if self.samples_per_bin is not None:
            target = self.samples_per_bin
        elif self.mode == MODE_DOWNSAMPLE:
            target = min(available)
        else:
            target = max(available)

        rng = random.Random(self.seed)
        selected: list[Schema] = []
        for b in range(num_bins):
            members = sorted(items_by_bin[b], key=lambda item: item.id)
            count = len(members)
            if count == 0:
                continue
            if self.mode == MODE_DOWNSAMPLE:
                k = min(target, count)
            else:
                k = target
            weight = count / k
            if k <= count:
                for item in rng.sample(members, k):
                    selected.append(self._with_weight(item, weight))
            else:
                for item in members:
                    selected.append(self._with_weight(item, weight))
                for _ in range(k - count):
                    duplicate = copy.deepcopy(rng.choice(members))
                    duplicate.id = str(uuid.uuid4())
                    selected.append(self._with_weight(duplicate, weight))

        return dataframe.from_list(selected)

    def _with_weight(self, item: Schema, weight: float) -> Schema:
        if self.weight_field is not None:
            item.metadata[self.weight_field] = weight
        return item
