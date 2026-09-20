# Reading the current UOp IR

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

`UOp` represents graph nodes at several compiler levels. There is no single flat list of opcodes that means the same thing at every point in compilation. The [Ops enumeration][enum] gives names; [the specification matchers][spec] define valid structures for particular stages.

## Construct scalar expressions

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad import dtypes

x = UOp.variable("x", 0, 10)
expression = x * 3 + 2
assert expression.op is Ops.ADD
assert expression.sym_infer({"x": 4}) == 14
constant = UOp.const(7, dtypes.int32)
assert constant.dtype == dtypes.int32
print(expression.render())
```

The constructor is `UOp(op, src=..., arg=..., tag=...)`. Dtype is inferred from structure and arguments; helpers such as `const`, casts, and typed parameters encode it. Do not reuse old examples with a dtype as the constructor's second positional argument.

| Family | Examples | What to inspect |
| --- | --- | --- |
| Arithmetic | ADD, MUL, MAX, WHERE | Sources, types, precision, validity. |
| Tensor movement | RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP | Logical shape and coordinate mapping. |
| Memory | BUFFER, PARAM, INDEX, LOAD, STORE | Storage/address space, element index, bounds. |
| Iteration/control | RANGE, END, IF, ENDIF, BARRIER | Extent, axis role, control dependencies. |
| Execution structure | CALL, AFTER, LINEAR | Invocation arguments and required ordering. |
| Compiled representation | PROGRAM, SOURCE, BINARY | Target, instruction list, generated text, compiled bytes. |

## Retired names are not aliases

The old `UOps.ALU`, `DEFINE_GLOBAL`, `LOOP`, and `ENDLOOP` descriptions should not be translated by blindly renaming symbols. Current arithmetic has distinct opcodes; parameters carry structured information; ranges have roles; execution and buffer dependencies are represented explicitly.

`INDEX` computes an addressed reference; `LOAD` reads a value and `STORE` writes one. Confusing these changes what a graph means. Similarly, `AFTER` adds ordering while passing through a value/state.

Read [the codegen chapter](codegen.md) to compare a kernel sink, a program's linear instruction list, and a schedule's linear list of calls.

[enum]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/__init__.py
[spec]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/spec.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/uops-doc.md).
