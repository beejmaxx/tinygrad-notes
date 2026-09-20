# Adding an accelerator to current tinygrad

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

An accelerator integration must preserve the shared compiler's semantics while satisfying a concrete device interface. Start from a current backend with similar execution and memory behavior; the old Backend/Linearizer scaffold no longer matches today's interfaces.

## Establish the boundary

Read [Compiled, Allocator, Compiler, Program, and TinyELF][device], then compare [CPU][cpu] and the target's closest existing runtime. A renderer declares supported operations and target properties; a compiler creates binary code; a runtime loads it, binds arguments, and launches it.

A minimal behavioral path needs host-to-device input, one kernel, completion synchronization, and device-to-host output. Without that complete round trip, an apparently successful launch cannot establish correctness.

## Bring up a discriminating workload

Use initialized, nonuniform inputs. Test an elementwise operation with a size that exposes boundary handling, then compare every output against a simple host reference. A buffer full of zeros can hide missing loads, wrong offsets, and kernels that never ran.

Next add noncontiguous movement, masked accesses, reductions, and supported dtypes. For a reduction that spans cooperating lanes, test the actual communication/barrier behavior rather than assuming a one-lane run proves parallel execution.

## Where each change belongs

Extend an existing renderer/compiler when the device already consumes a supported representation. Add target-specific lowering only when the shared representation requires it. Keep host allocation/launch details in the runtime; do not repair a shared indexing error with a backend special case.

The current `Program(dev, obj)` interface receives a `TinyELF` with program metadata and binary payload. Inspect a contemporary backend's signature handling; reproducing an old tuple of pointers can silently misbind scalar arguments.

## Evidence before performance claims

Verify fail-before/pass-after behavior on the intended compiler and hardware. Record unsupported operations explicitly, including precision, vector widths, local memory, and synchronization limits. Compare compilation and runtime costs separately.

Consult upstream contribution rules before preparing a PR. This chapter describes an implementation investigation, not a claim that a new accelerator is a small or automatically acceptable change. No new hardware backend is implemented or validated by these tutorials.

[device]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/device.py
[cpu]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/runtime/ops_cpu.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/addingaccelerator.md).
