# Symbolic sizes and reduction means

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

A mean is a sum divided by the number of reduced elements. With symbolic shapes, that count must remain an expression until the relevant dimension values are bound.

Current symbolic values are UOps. The old `tinygrad.shape.symbolic.Variable` import and LazyBuffer constructor examples no longer apply.

## Inspect a symbolic element count

```python
from tinygrad.uop.ops import UOp

n = UOp.variable("n", 1, 8)
count = n * 3
assert count.sym_infer({"n": 2}) == 6
assert count.sym_infer({"n": 8}) == 24
bound = n.bind(4)
variable, value = bound.unbind()
assert variable is n and value == 4
print(count.render())
```

`sym_infer` takes values keyed by variable names. Binding constructs a state expression that carries a concrete value subject to the variable's bounds. The current representation is documented directly by [UOp.variable/bind][ops]; it is not the old symbolic-node hierarchy.

## Follow mean's implementation

```python
from tinygrad import Tensor

x = Tensor([[1, 2, 3], [4, 5, 6]])
assert x.mean().item() == 3.5
assert x.mean(axis=1).tolist() == [2.0, 5.0]
assert x.mean(axis=0, keepdim=True).tolist() == [[2.5, 3.5, 4.5]]
print(x.mean().item())
```

[OpMixin.mean][op] chooses a result dtype, casts for accumulation, sums along the requested axes, computes a denominator from the reduced shape dimensions, divides, and casts the result. It preserves symbolic arithmetic where sizes are symbolic rather than evaluating every shape as a Python integer.

This page checks scalar symbolic evaluation and concrete mean behavior. It does not claim that arbitrary shape-polymorphic Tensor programs work across every backend or JIT input pattern. For a symbolic-shape change, test the actual Tensor construction, bindings, scheduling, and replay path over several legal values.

Zero-length reductions and bounds are separate edge cases. Do not infer their behavior from the positive-size example.

[ops]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/ops.py
[op]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/mixin/op.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/symbolic-mean.md).
