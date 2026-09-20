# Upcasting: multiple values per execution lane

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

In tinygrad's kernel optimizer, upcasting means expanding work over an axis so one execution lane handles several values. It does not mean converting a value to a wider dtype.

For sixteen elementwise outputs, one possible organization has sixteen lanes producing one value each. Another has four lanes producing four values each. They cover the same logical output, but differ in address arithmetic, instruction count, vectorization opportunities, and register demand.

## The current API

```python
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import AxisType

upcast = Opt(OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))
unroll = Opt(OptOps.SPLIT, axis=0, arg=(4, AxisType.UNROLL))
assert upcast != unroll
print(upcast)
print(unroll)
```

These are requested transformations, not executable kernels. [postrange.Scheduler][scheduler] validates a split against its source range and allowed target axis role. An output axis and a reduction axis are not interchangeable.

The old `Linearizer.upcast()` and `OptOps.UPCAST` examples have been replaced by range roles and `SPLIT`. [codegen's expander][codegen] builds a range map for UPCAST and UNROLL axes and replaces selected ranges with shaped constants before later lowering.

## Why it can help or hurt

Processing several adjacent values can expose vector operations and reuse computations. Expanding too much work can increase live values, register pressure, instruction size, or compilation cost. On a GPU, it can also reduce occupancy.

The optimizer's choices depend on target capabilities and layout. Four values per lane does not promise a single four-wide load: alignment, validity masks, memory layout, and renderer support still constrain the generated instructions.

Use [BEAM](20241203_beam.md) or the existing heuristics to compare legal candidates. For correctness, cover tails and masked accesses as well as evenly divisible sizes; for performance, report the target and generated program.

[scheduler]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/opt/postrange.py
[codegen]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/__init__.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/upcast.md).
