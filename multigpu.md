# Sharding and multi-device execution

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

Current multi-device tensors are represented through UOps and device tuples, not the old `MultiLazyBuffer` class. Sharding describes placement; lowering must turn that placement into local work, transfers, and any required collective operations.

## Replication versus partitioning

[Tensor.shard][tensor] takes a tuple of devices and an optional axis. With an axis, tensor data is partitioned along that dimension. With no axis, the placement is replicated. The single-device case reduces to moving the Tensor to that device.

A reduction over an unsharded dimension can be performed independently within shards. Reducing a sharded dimension requires combining contributions to obtain the global result. Replicated values must not be accidentally counted once per device.

## Follow the current representation

[UOp.shard and sharding metadata][ops] describe device-axis relationships. [schedule/multi.py][multi] rewrites elementwise operations, movement, reductions, and copies around this representation. The operation vocabulary includes `MSTACK`, `MSELECT`, `UNSHARD`, and `ALLREDUCE`.

`reduce_multi` distinguishes reductions that remove sharding axes from those that preserve them. In the former case, local reduction results feed collective combination. The actual communication path depends on the selected backend and available connectivity.

## A two-device experiment

The following is a hardware exercise, not part of the portable CPU example checks. Use two supported devices and substitute their actual names:

```text
x = Tensor([1., 2., 3., 4.]).shard(("DEVICE:0", "DEVICE:1"), axis=0)
y = (x * 2).realize()
check gathered values equal [2., 4., 6., 8.]
check x.sum() equals 10.
```

Verify both the partition-preserving operation and the cross-shard reduction. Inspect transfers and collectives as well as values. Two logical device labels on one CPU are not a validation of GPU peer transfers or distributed synchronization.

## Training introduces more state

Parameters, gradients, optimizer state, and input batches need compatible placement. A correct forward result does not prove backward aggregation or optimizer updates are correct. Compare several full training steps with a single-device reference using controlled initial values.

No multi-GPU hardware validation is claimed in this tutorial revision. The representation and source paths above were checked against the pinned implementation.

[tensor]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/tensor.py
[ops]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/ops.py
[multi]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/multi.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/multigpu.md).
