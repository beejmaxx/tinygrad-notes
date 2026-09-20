# Reading kernel names and axis colors

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

Kernel names are debugging output. Their dimensions and colors summarize an optimized iteration structure, not necessarily the original Tensor shape or a literal GPU grid.

Current naming lives in [Scheduler.get_optimized_ast][scheduler]. It chooses a reduction or elementwise prefix and incorporates special dimensions and optimized ranges. The old explanation based on a `Linearizer.linearize` method is obsolete.

## Print the current axis legend

```python
from tinygrad.uop.ops import axis_colors

for axis, color in axis_colors.items():
  print(axis.name, color)
assert axis_colors
```

The exact mapping comes from [uop/ops.py][ops]. `Scheduler.colors` additionally handles weak ranges according to whether they participate in outputs and can become global work, so the dictionary alone does not explain every colored field.

A GLOBAL axis and a LOCAL axis describe different execution roles. UPCAST/UNROLL describe expanded work. REDUCE/GROUP_REDUCE describe reduction organization. Multiplying every visible extent and calling it a thread count conflates these roles.

## Connect a name to an actual launch

Run an initialized Tensor computation with `DEBUG=2` to find the kernel's name and execution statistics. Use `DEBUG=4` or VIZ to inspect source and launch metadata. Check the target's mapping of global and local dimensions before interpreting a printed factor as a block or thread count.

Names can change after a compiler optimization without changing the output. Tests should normally assert behavior, not incidental kernel spellings. When investigating performance, preserve the full target/configuration alongside names and timings.

[scheduler]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/opt/postrange.py
[ops]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/ops.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/colors.md).
