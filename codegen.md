# Code generation: kernel sink to executable program

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

Current codegen operates on a kernel `SINK` and returns a `PROGRAM` UOp. The old `Linearizer` class and `UOps.ALU` examples are obsolete; arithmetic is represented by operations such as `Ops.ADD` and `Ops.MUL`.

## Inspect a compiled program

```python
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import compile_linear, run_linear

x = Tensor([1.0, 2.0, 3.0]).realize()
y = x * 3 + 1
linear = y.schedule_linear()
compiled = compile_linear(linear)
programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM]
assert programs
for program in programs:
  sources = [u.arg for u in program.src if u.op is Ops.SOURCE]
  assert sources
  print(sources[0])
run_linear(compiled)
assert y.tolist() == [4.0, 7.0, 10.0]
```

The program source is intentionally not pasted as a universal expected output. It depends on the target renderer, compiler configuration, and optimization decisions. The numerical assertion checks the actual execution.

## Read the lowering sequence

[full_rewrite_to_sink][codegen] resolves multi-device/in-kernel sharding structure, preprocesses movement, simplifies expressions/ranges, and applies kernel optimization. It then expands selected ranges, lowers reductions, introduces local buffers where required, and adds GPU dimensions before later target work.

The later stages prepare operations the renderer can handle, arrange control flow, and, for applicable ISA renderers, perform instruction selection and register allocation. Source-language renderers and direct instruction renderers have different downstream responsibilities.

`do_to_program` derives `ProgramInfo` from the lowered sink and renderer target. Its program rewrite produces a `PROGRAM` with `SINK`, `LINEAR`, `SOURCE`, and `BINARY` components.

## Two meanings of LINEAR

A schedule's `LINEAR` contains executable calls. A program's `LINEAR` contains the ordered low-level operations for one kernel. The opcode alone is not enough to identify the abstraction level; inspect its surrounding graph and sources.

Likewise, a `RANGE` in a high-level kernel can be transformed into GPU launch dimensions, expanded values, or ordinary loops. Do not assume every range becomes a source-language `for` loop.

## Caching is part of compilation

`to_program_key` includes the AST key, renderer type/target, and relevant compiler settings. A cache hit reuses the compiled representation; it does not establish that input values or output buffers are reusable.

Compilation correctness requires more than valid source syntax. The generated program must preserve indexing, validity, dtype semantics, reduction behavior, and synchronization. Test on the intended target after inspecting the graph.

[codegen]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/__init__.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/codegen.md).
