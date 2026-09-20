# Reshape composition and dimension simplification

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

The old `View._merge_dims` explanation no longer describes current tinygrad. Today, movement operations become index expressions, and symbolic/range simplification removes unnecessary structure.

## Begin with a flat address

A contiguous `(2, 3, 4)` array maps `(i,j,k)` to `12*i + 4*j + k`. The last two axes can be treated as a single twelve-element coordinate without changing that address.

```python
from tinygrad import Tensor

x = Tensor.arange(24).reshape(2, 3, 4).realize()
flattened = x.reshape(2, 12)
restored = flattened.reshape(2, 3, 4)
assert restored.tolist() == x.tolist()
transposed = x.permute(1, 0, 2).reshape(3, 8)
assert transposed.tolist() == [
  [0, 1, 2, 3, 12, 13, 14, 15],
  [4, 5, 6, 7, 16, 17, 18, 19],
  [8, 9, 10, 11, 20, 21, 22, 23],
]
print(transposed.tolist())
```

The second reshape follows a permutation. Its logical order differs from flattening the original storage. A correct compiler must compose the mappings; it cannot discard the permutation merely because the element counts agree.

## How current tinygrad composes views

[_apply_reshape][indexing] flattens output coordinates and reconstructs input coordinates with division and remainder. It then invokes symbolic simplification. `apply_movement_op` also handles permutation, shrink, flip, expand, and pad.

Later [codegen/simplify.py][simplify] performs range simplification, flattening, and splitting. This is a different representation from combining a list of stride-bearing View objects.

Broadcasting and padding complicate merging. A broadcast dimension may not advance an input address; a padded region carries validity conditions. Removing an axis is safe only if the address and valid-value behavior remain equivalent for every relevant coordinate.

## What to inspect

Use a small initialized array with distinct values, as above. Equal shapes alone cannot detect ordering errors. Check every coordinate before looking at optimized source, then compare index expressions before and after the relevant rewrite in VIZ.

A smaller index expression can reduce integer work, but the performance effect depends on the full kernel. Simplification should first preserve the mapping; benchmark separately.

[indexing]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/schedule/indexing.py
[simplify]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/simplify.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/mergedim.md).
