# Current tinygrad: from tensors to execution

[All tutorials](README.md)

Updated September 21, 2026 against [upstream commit `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642).
The latest tagged release is [0.14.0](https://github.com/tinygrad/tinygrad/releases/tag/v0.14.0); this chapter follows the newer master snapshot above. Python 3.11 or newer is required.

Di Zhu's original tutorials teach useful compiler ideas, but several internal APIs have moved or disappeared. Start with a small program that runs on today's tree, then follow the objects it actually uses.

## A computation with known inputs

```python
from tinygrad import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[10.0, 20.0], [30.0, 40.0]])
c = a + b
print(c.tolist())
# [[11.0, 22.0], [33.0, 44.0]]
```

`a + b` constructs a computation; `tolist()` asks for its values and transfers them to a Python list. To request execution while retaining a Tensor, use `c.realize()`.

Use initialized inputs for experiments. `Tensor.empty` allocates uninitialized storage: zeros observed on one device are not a guarantee. The old introduction's zero output should not be used as a correctness expectation.

Save the program as `example.py`. Run `DEBUG=2 python3 example.py` to inspect execution statistics, then `DEBUG=4 python3 example.py` to see generated source. Exact kernels, launch dimensions, and timings depend on the selected backend and optimization choices.

## Follow the current pipeline

The [upstream developer overview](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/docs/developer/developer.md) divides the framework into frontend, scheduler, lowering, and execution.

| Stage | Current source | What to look for |
| --- | --- | --- |
| Tensor frontend | [`tinygrad/tensor.py`](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/tensor.py) | Tensor operations construct UOps; `realize` requests execution. |
| Shared graph representation | [`tinygrad/uop/ops.py`](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/ops.py) | `UOp`, `UPat`, `PatternMatcher`, and `graph_rewrite`. |
| Scheduling | [`tinygrad/schedule/__init__.py`](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/__init__.py) | Preparation and rangeification create a kernel graph; `create_schedule` orders it into a `LINEAR` UOp. |
| Kernel lowering | [`tinygrad/codegen`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen) | Rewrites and optimization turn kernel computations into target-ready operations. |
| Rendering and device support | [`tinygrad/renderer`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/renderer), [`tinygrad/runtime`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/runtime) | Target representation, compilation, allocation, and program launch. |
| Execution | [`tinygrad/engine/realize.py`](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/engine/realize.py) | `run_linear` dispatches the calls in the execution graph. |

The schedule is now represented with UOps too: a `LINEAR` contains ordered `CALL`s. Older tutorials showing lists of `ScheduleItem`s or importing graph operations from `tinygrad.ops` need translation to the current tree. For an experiment, inspect `Tensor.schedule_linear`; do not assume an old `Tensor.schedule()` snippet still applies.

## Where did ShapeTracker go?

The current tree has no `tinygrad.shape.shapetracker` or `tinygrad.shape.view` module. The [original ShapeTracker chapter](20241217_st.md) remains useful for understanding strides, views, masks, and index expressions, but its imports belong to the historical implementation.

Read today's movement operations in `tinygrad/uop/ops.py` alongside [`tinygrad/schedule/rangeify.py`](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/rangeify.py) and [`tinygrad/schedule/indexing.py`](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/indexing.py). These are starting points for following how movement and indexing are represented and lowered; there is no drop-in replacement import for the old tutorial.

This public Tensor example still demonstrates the underlying idea:

```python
from tinygrad import Tensor

x = Tensor([[1, 2], [3, 4]])
print(x.permute(1, 0).tolist())
# [[1, 3], [2, 4]]
```

The logical transpose does not, by itself, establish how a particular downstream kernel will access memory. Inspect the generated program to answer that question.

## What TinyJit captures now

[`tinygrad/engine/jit.py`](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/engine/jit.py) makes the default sequence explicit: first call runs the Python function, second call runs it while capturing execution, subsequent calls replay the captured work. Capture combines `LINEAR` graphs, plans memory, and lowers the captured execution. Device graph batching is backend-dependent.

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def double(x):
  return (x * 2).realize()

for value in (1.0, 2.0, 3.0):
  x = Tensor([value]).realize()
  print(double(x).tolist())
# [2.0]
# [4.0]
# [6.0]
```

This is execution capture and replay, not merely a second stage of GPU compilation. The replay path checks input names and expected input information; arbitrary changes to input structure are not automatically retraced. Python side effects inside the function do not run on every replay.

## Inspect rewrites with VIZ

The current [VIZ README](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/viz/README.md) covers both rewrite inspection and profiling. `VIZ=1 python3 example.py` records the data and, in an interactive shell, launches the viewer. A command-line viewer is also available through `python3 -m tinygrad.viz.cli`.

The old screenshots remain illustrations of earlier versions. Follow the current README for viewer commands and flags.

## Continuing the update

This first pass checks the three runnable examples above and maps the major changes in the execution path. The next chapters to revise in depth are the introduction, JIT, pattern matching, and the historical ShapeTracker material. The original BEAM, convolution, matrix-multiplication, and hardware articles still require individual checks against this snapshot.

Before contributing upstream, read the current [contribution rules](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/README.md#contributing), including the requirements for regression tests, benchmarks, and disclosure of AI use. This fork is an independent tutorial update, with AI assistance; it is not official tinygrad documentation.
