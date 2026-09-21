# Scheduling: from Tensor graphs to LINEAR and CALL

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

The current scheduler produces UOps. The old `ScheduleItem` list, `LazyBuffer`, and `run_schedule` examples no longer describe this implementation.

A computation graph answers what each value means. A schedule additionally answers when computations can run, which intermediate values need storage, and what must finish before another operation can read or overwrite a buffer.

## Inspect and execute a schedule

```python
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import run_linear

x = Tensor([1.0, 2.0, 3.0]).realize()
y = (x + 1) * 2
linear = y.schedule_linear()
assert linear.op is Ops.LINEAR
assert all(call.op is Ops.CALL for call in linear.src)
assert len(linear.src) > 0
print([(call.op.name, call.body.op.name) for call in linear.src])
run_linear(linear)
assert y.tolist() == [4.0, 6.0, 8.0]
```

Realizing the input first keeps its initialization outside the inspected schedule. `schedule_linear` updates Tensor graph references as part of preparation; execute the returned graph before reading this output. For ordinary application code, call `realize` instead of managing schedules manually.

The example checks the graph's kind and result, not a universal kernel count. Fusion decisions depend on the operation, layout, configuration, and compiler revision.

## The transformations before ordering

[tensor.py][tensor] bufferizes requested outputs and constructs a call. In [schedule/__init__.py][schedule], `lower_sink_to_linear` processes a precompiled call whose body is a non-kernel sink. On a cache miss it runs:

```text
prepare_rangeify(function)
    → get_kernel_graph(...)
    → create_schedule(...)
```

[Rangeification](20241217_st.md) introduces explicit coordinates and realization boundaries. Kernel splitting builds calls and buffer-state dependencies. Schedule-cache reuse is separate from compiled-program reuse: a reusable execution structure does not by itself mean that its kernels are compiled or its buffers allocated.

## Why AFTER matters

The [worked movement trace](movement-trace.md#4-schedule-one-call-two-buffers) includes the complete scheduled `LINEAR`: one call, a six-float input, and a five-float output. Inspecting its parameters shows that the logically 20-element padded intermediate has not been allocated as a separate buffer. This connects fusion to concrete storage, rather than inferring it from Tensor syntax alone.

`AFTER` passes its first source through while imposing dependencies on its other sources. The scheduler tracks buffer states, not just a bag of arithmetic expressions.

Suppose one kernel reads a buffer's old value and another overwrites that buffer. The reader must precede the overwrite even if the overwrite does not consume the reader's output. This is a write-after-read dependency.

`create_schedule` collects producer/consumer relationships, includes read-after-write dependencies, adds write-after-read ordering for superseded states, and topologically sorts the calls. A cycle is reported as an error.

## Where memory planning and execution fit

A topological order is not a complete memory allocation strategy. [schedule/memory.py][memory] handles memory-planning rewrites. JIT capture also plans the combined captured graph, which can expose different reuse opportunities from isolated eager realizations.

At execution, `run_linear` compiles/links the linear graph and dispatches its calls. See [code generation](codegen.md) and [command queues](commandqueue.md) for the next boundaries.

[schedule]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/__init__.py
[tensor]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/tensor.py
[memory]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/memory.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/scheduleitem.md).
