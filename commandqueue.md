# Execution order, command queues, and completion

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

A dependency-correct schedule is the input to device execution. It must still become commands with valid argument addresses, synchronization, lifetimes, and completion behavior.

The current [run_linear][realize] path compiles and links a `LINEAR` graph before dispatching calls. The historical Python queue built from ScheduleItems is no longer the general execution model.

## Follow an ordinary kernel call

The kernel handler resolves parameter buffers and symbolic values, ensures storage is allocated, obtains a runtime program, computes launch dimensions, and invokes the program.

Other handlers execute copies, backend graphs, encoding/decoding work, or HCQ batches. A `CALL` node therefore does not universally mean one source-level compute kernel.

## HCQ compilation and linking

[runtime/support/hcq2.py][hcq] groups eligible calls, constructs command representations, encodes queue data, and links values needed for execution. `HWQueue` has rewrite rules for execution, copies, waits, signals, timestamps, barriers, and queue loops.

The `HCQ_DEVS` set and eligibility checks matter: not every backend or copy uses this path. Backend-specific graph mechanisms coexist with it. Avoid treating the NVIDIA or AMD command format as a universal tinygrad queue.

A wait establishes an ordering relation only according to the device's signaling semantics. Buffer ownership, peer access, and lifetimes must also be correct. Submitting two commands in Python order is not sufficient evidence of cross-device ordering.

## Observe completion from application code

```python
from tinygrad import Tensor, Device

x = Tensor([2.0, 4.0, 6.0]).realize()
y = (x + 3).realize()
Device[y.device].synchronize()
assert y.tolist() == [5.0, 7.0, 9.0]
print(y.tolist())
```

The explicit synchronization is useful when delimiting a measurement or an external consumer's access. `tolist()` also crosses a host-read boundary, but it adds transfer/conversion work that should not be silently included in a kernel-only benchmark.

To debug missing or stale results, check schedule dependencies, encoded waits/signals, argument patching, and buffer lifetimes before interpreting the final arithmetic.

[realize]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/engine/realize.py
[hcq]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/runtime/support/hcq2.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/commandqueue.md).
