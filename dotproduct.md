# Dot products: values, reduction, and fusion

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

For two vectors, a dot product multiplies corresponding elements and sums the products. tinygrad expresses this through its ordinary Tensor operations, making it a compact example of fusion and reduction.

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0]).realize()
b = Tensor([4.0, 5.0, 6.0]).realize()
assert a.dot(b).item() == 32.0
assert (a * b).sum().item() == 32.0
print("1*4 + 2*5 + 3*6 = 32")
```

[OpMixin.dot][op] handles more than vectors: higher ranks require shape normalization and a transpose of the appropriate right-hand axis. The [matrix-multiplication chapter](20241203_matmul.md) works through that case.

For the vector case, the product need not become a separately allocated array. Its values can flow directly into accumulation. Calling `realize()` on the product first requests an intermediate and changes the scheduling opportunities.

The old article described LazyBuffers and ScheduleItems. To trace today's implementation, inspect `out.uop` before realization, follow [rangeification](20241217_st.md), inspect [the scheduled calls](scheduleitem.md), then examine [the generated program](codegen.md).

Floating-point summation order matters. A parallel reduction can group additions differently from a scalar Python loop. Use a justified numerical tolerance for non-exact examples, and state the input and accumulation dtypes. The small integer-valued float example above has an exactly representable result.

[op]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/mixin/op.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/dotproduct.md).
