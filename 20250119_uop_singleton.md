# UOp interning and shared graphs

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

A UOp is not a singleton in the sense of one object for an entire class or operation. tinygrad interns structurally identical nodes: constructing the same node while it is still alive can return the same Python object.

## Observe identity

```python
from tinygrad.uop.ops import UOp

a = UOp.const(7)
b = UOp.const(7)
assert a is b

x = UOp.variable("x", 0, 10)
left = x + 7
right = x + 7
assert left is right
assert UOp.const(True) is not UOp.const(1)
print("identical live nodes are shared")
```

[UOpMetaClass][ops] currently keys its weak-reference cache by `(op, src, arg, tag, type(arg))`. Including the argument's type distinguishes values such as `True` and `1`, which compare equal as Python dictionary keys but imply different constants.

The current constructor is `UOp(op, src=..., arg=..., tag=...)`. Old snippets passing a dtype as the second positional argument are obsolete. Use helpers such as `UOp.const(value, dtype)`, and inspect the current dtype inference and cast operations.

## Identity is not algebraic equivalence

Two nodes can compute equal values without having the same structure. `x + x` and `x * 2` are an example. A symbolic rewrite may relate them, but interning alone does not prove the equivalence or canonicalize every expression.

Shared structure makes replacement discipline important. In-place mutation would affect every consumer and undermine cached properties. Compiler rewrites return replacement nodes and rebuild the affected graph.

## Weak references and lifetime

The cache contains weak references. Do not assume an object address remains a permanent identifier once no strong references remain. Compiler cache keys and serialized graph representations solve different problems from Python object identity.

Metadata is maintained separately from the structural key. Attaching provenance does not mean a distinct mathematical operation has been created.

See [pattern matching](20241112_pm.md) for a rewrite that preserves this shared-graph model.

[ops]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/uop/ops.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/20250119_uop_singleton.md).
