# Tutorials on Tinygrad

[View on Github](https://github.com/beejmaxx/tinygrad-notes) |
[View on Website](https://beejmaxx.github.io/tinygrad-notes/)

Tutorials on tinygrad internals, adapted from [Di Zhu's original notes](https://github.com/mesozoic-egg/tinygrad-notes).
Updated September 21, 2026 against [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642).

The chapters now explain the current UOp, rangeification, scheduling, compilation, and runtime implementation. Existing chapter addresses are preserved; each chapter links its original version. These are independent tutorials, revised with AI assistance.

## Start here

- [Setup and how to use these notes](current-tinygrad.md)
- [Introduction: from Tensor values to execution](20241231_intro.md)
- [Worked compiler trace: transpose → pad → add → reduce](movement-trace.md) — captured UOps, CPU/Metal source, and an independent regression oracle
- [Shapes, movement operations, and indexing](20241217_st.md) — replaces the ShapeTracker chapter
- [Scheduling with LINEAR, CALL, and buffer dependencies](scheduleitem.md)
- [The current UOp IR](uops-doc.md)
- [Pattern matching and graph rewriting](20241112_pm.md)
- [UOp interning and shared graphs](20250119_uop_singleton.md)

## Computation and optimization

- [Dot products](dotproduct.md)
- [Matrix multiplication](20241203_matmul.md)
- [Convolution windows and arange](20241208_conv.md)
- [Operator fusion](20250117_fusion.md)
- [Following a reduction through UOps](uops.md)
- [Reshape composition and dimension simplification](mergedim.md)
- [Symbolic sizes and means](symbolic-mean.md)
- [BEAM search and kernel optimization](20241203_beam.md)
- [Upcasting](upcast.md)
- [Index expressions after range splitting](upcast2.md)
- [Code generation](codegen.md)
- [Instruction ordering and control flow](uops-2.md)
- [Kernel names and axis colors](colors.md)

## Execution and inspection

- [TinyJit capture and replay](20240102_jit.md)
- [Graph rewrites with VIZ](20241129_viz.md)
- [Profiling compilation, execution, and replay](profiling.md)
- [Host memoryviews and Tensor storage](20250114_memoryview.md)
- [Backend responsibilities](backends.md)
- [Command queues and completion](commandqueue.md)
- [Sharding and multi-device execution](multigpu.md)
- [Adding an accelerator](addingaccelerator.md)

## Hardware details

- [Following the Metal runtime](20240921_metal.md)
- [Tensor-core tiles and fragment layouts](cuda-tensor-core-pt1.md)
- [LOP3 truth tables](20250325_lop3_table.md)

Runnable Python blocks are checked by [the example runner](scripts/check_examples.py). See [validation coverage](current-tinygrad.md#validation-coverage) for the tested revision, devices, and hardware limits.
