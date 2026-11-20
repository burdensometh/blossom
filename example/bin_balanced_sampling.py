from blossom.dataset import create_dataset
from blossom.op import BinBalancedSampler, EqualWidthBinner
from blossom.schema import RowSchema

train_data = [
    RowSchema(data={"score": s})
    for s in [1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 8, 13, 21, 34, 55, 89]
]
val_data = [RowSchema(data={"score": s}) for s in [4, 200, -5]]

binner = EqualWidthBinner(func=lambda s: s["score"], num_bins=4)

# Fit happens exactly once, on the training dataset: the first execute()
# fits and transforms, later executions only apply the frozen state.
train_ds = create_dataset(train_data).execute([binner])
print("fitted bin edges:", binner.bin_edges)

# The frozen edges and labels are serializable and can be restored into
# another instance (e.g. in a separate validation/serving process).
restored = EqualWidthBinner(func=lambda s: s["score"], num_bins=4)
restored.load_state_dict(binner.state_dict())

# Validation data is binned with the frozen training boundaries: out-of-range
# scores (200, -5) are clamped to the edge bins instead of refitting.
val_ds = create_dataset(val_data).execute([restored])
print(
    "validation bins:",
    [(item["score"], item.metadata["bin"]) for item in val_ds.collect()],
)
print("boundaries after validation:", restored.bin_edges)

# The bin-balanced sampler consumes the binning result and actually changes
# the output population (downsampling majority bins), optionally attaching
# sample weights that sum back to the original population per bin.
sampler = BinBalancedSampler(restored, weight_field="sample_weight", seed=42)
balanced_ds = train_ds.execute([sampler])
print("bin counts before sampling:", sampler.bin_counts_)
print(
    "balanced population:",
    [
        (item["score"], item.metadata["bin"], item.metadata["sample_weight"])
        for item in balanced_ds.collect()
    ],
)
