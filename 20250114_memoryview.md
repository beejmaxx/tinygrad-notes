# Host memoryviews and Tensor storage

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

Python's `memoryview` exposes bytes through the buffer protocol without necessarily copying them. tinygrad's `Tensor.data()` returns a typed, shaped memoryview of host-accessible data. These statements do not mean an arbitrary GPU tensor is directly readable by Python without synchronization or transfer.

## Inspect typed data

```python
from tinygrad import Tensor, dtypes

x = Tensor([[1, 2], [3, 4]], dtype=dtypes.int32)
view = x.data()
assert isinstance(view, memoryview)
assert view.shape == (2, 2)
assert view.itemsize == 4
assert view.tolist() == [[1, 2], [3, 4]]
assert view.nbytes == 16
print(view.format, view.shape, view.tolist())
```

The shape and item size explain the sixteen bytes: four elements, four bytes each. A memoryview's format describes how to interpret those bytes. Casting bytes to another format does not numerically convert each element.

## Trace the host boundary

[Tensor.data][tensor] first commits weak dtypes if necessary, handles empty tensors, rejects unresolved symbolic shapes, obtains a buffer through `_buffer()`, and casts the resulting memoryview to the Tensor's format and shape.

`Tensor._buffer()` makes the value contiguous and realizes it. [Buffer.as_memoryview][device] defaults to `allow_zero_copy=False`: even when storage has a host view, it synchronizes and copies into a new bytearray. `Tensor.data()` uses this default. Its returned view is therefore not a promise of a writable alias to the original Tensor storage.

Keep the owner alive while using a view. Do not infer that mutating an exported view is a supported way to update every device Tensor; aliasing, copied storage, and synchronization differ across paths. Use Tensor operations for portable updates.

## NumPy and byte interpretation

`numpy.frombuffer` can interpret a memoryview with a dtype, while `Tensor.numpy()` offers a Tensor-level conversion. Matching dtype, shape, byte order, and lifetime is essential. An equal byte count does not establish an equal value representation.

For debugging, `tolist()` creates ordinary Python values and avoids lifetime/aliasing ambiguity. For performance-sensitive interop, inspect the exact backend and ownership contract before calling a path zero-copy.

[tensor]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/tensor.py
[device]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/device.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/20250114_memoryview.md).
