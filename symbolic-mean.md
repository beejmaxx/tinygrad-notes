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

## Execute a genuinely symbolic reduction dimension

```python
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

base = Tensor.arange(24).reshape(8, 3).realize()
n = UOp.variable("n", 1, 8)
for size in (1, 3, 8):
  x = base.shrink(((0, n.bind(size)), (0, 3)))
  assert isinstance(x.shape[0], UOp)
  actual = x.mean(axis=0).tolist()
  expected = [sum(3*r+c for r in range(size))/size for c in range(3)]
  assert actual == expected, (size, actual, expected)
print("symbolic reduction bound at 1, 3, and 8")
```

The backing allocation has eight rows, but the logical first dimension is a bound symbolic UOp. Reducing that dimension must divide by the bound row count, not the backing allocation's row count or the variable's maximum. The independent Python oracle checks both endpoints and one interior binding. This exercises Tensor construction, symbolic shrinking, scheduling, and execution; it does not claim arbitrary shape-polymorphic JIT replay or support across every backend.

Zero-length reductions and bounds are separate edge cases. Do not infer their behavior from the positive-size example.

[ops]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/ops.py
[op]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/mixin/op.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/symbolic-mean.md).
