# Index expressions after range splitting

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

Upcasting and loop unrolling change how a logical coordinate is represented. They must preserve the mapping from every output/reduction coordinate to its input element.

## Split one coordinate into two

Suppose `r` ranges over twelve elements. A split by four represents it as `4*outer + inner`, with `outer` in `[0,3)` and `inner` in `[0,4)`.

```python
from tinygrad.uop.ops import UOp

outer = UOp.variable("outer", 0, 2)
inner = UOp.variable("inner", 0, 3)
r = outer * 4 + inner
seen = [r.sym_infer({"outer": a, "inner": b}) for a in range(3) for b in range(4)]
assert seen == list(range(12))
print(seen)
```

This is the coordinate identity a split must preserve. It is not a request to the kernel optimizer; it isolates the indexing argument so it can be checked independently.

## Translate a transposed address

If an input's address expression contains `3*r + j`, substitution produces `12*outer + 3*inner + j`. If the inner axis is expanded, the compiler can specialize that expression for inner values 0, 1, 2, and 3.

Current [Scheduler.shift_to][scheduler] creates ranges and rewrites indexing according to a requested split. Later [expansion][codegen] turns selected UPCAST/UNROLL ranges into shaped constants. The renderer receives the lowered result, not a ShapeTracker.

## Validity must survive substitution

When a split/padding strategy introduces coordinates beyond the original extent, the accesses must remain guarded or otherwise made valid. A rewrite that preserves the arithmetic for valid points but loses the bounds condition is incorrect.

See [movement and indexing](20241217_st.md) for padding validity and reshape composition. When debugging, trace the coordinate expression and its validity together; a pretty simplified address is only half of the access semantics.

[scheduler]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/opt/postrange.py
[codegen]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/__init__.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/upcast2.md).
