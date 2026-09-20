# Following a reduction through UOps

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

A dot product is a useful compiler example because it combines loads, multiplication, a reduction, and a scalar output. Current tinygrad uses UOps through the frontend, scheduling, and kernel stages; the historical LazyBuffer-to-Linearizer route is gone.

## Start with an observable result

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0]).realize()
b = Tensor([4.0, 5.0, 6.0]).realize()
out = (a * b).sum()
names = {u.op.name for u in out.uop.toposort()}
assert "MUL" in names and "REDUCE" in names
assert out.item() == 32.0
print(sorted(names))
```

Inspect the graph before `item()`: realization can replace Tensor references with storage-backed forms. The presence of MUL and REDUCE describes the requested computation, not the final instructions.

## At the indexing stage

[convert_reduce_to_reduce_with_ranges][indexing] rewrites an axis-based reduction into a reduction with explicit range sources. Movement operations have already supplied the mappings needed to index the inputs.

At kernel optimization, ranges can be split, grouped, unrolled, or mapped to device execution. At reduction lowering, the compiler constructs the operations and storage needed to accumulate results.

The final implementation could use a serial loop, parallel partial sums, vector operations, or specialized target instructions. Source-level `sum` does not select one of these by itself.

## Read dependencies as well as math

A kernel may need local storage and synchronization when several lanes cooperate. An output store must happen after its contributing computation. Correct arithmetic with missing ordering is still a wrong kernel.

Use [VIZ](20241129_viz.md) to inspect the reduction before and after each pass, then [compile a schedule](codegen.md) to inspect its actual program.

[indexing]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/indexing.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/uops.md).
