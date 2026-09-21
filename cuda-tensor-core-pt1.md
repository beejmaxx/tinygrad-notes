# Tensor-core tiles and fragment layouts

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

A matrix multiplication expression does not prove that a device used tensor-core instructions. The compiler must recognize a supported pattern, choose a tile/precision combination, distribute fragments across lanes, and emit the corresponding target operations.

## Inspect a current fragment description

```python
from tinygrad.renderer.tc import cuda_81616
from collections import Counter

tc = cuda_81616[0]
assert tc.dims == (8, 16, 16)  # tinygrad's order is N, M, K
assert tc.threads == 32
a, b, c = tc.frag_coords()
assert len(a) == len(b) == len(c) == 32
assert {p for lane in c for p in lane} == {
  (m, n) for m in range(16) for n in range(8)
}
for fragment, shape, slots in ((a, (16, 16), 8), (b, (16, 8), 4), (c, (16, 8), 4)):
  assert all(len(lane) == slots for lane in fragment)
  coverage = Counter(p for lane in fragment for p in lane)
  assert coverage == Counter({(i, j): 1 for i in range(shape[0]) for j in range(shape[1])})
print("N,M,K:", tc.dims, "lanes:", tc.threads)
print("lane 0 output coordinates:", c[0])
```

This verifies the Python layout description, without requiring an NVIDIA GPU. It does not execute an MMA instruction.

[TensorCore in renderer/tc.py][tc] expresses operand fragments as lane bits and element bits in tile coordinates. `frag_coords()` expands that description into the tile coordinate associated with each lane/element slot. The output coverage assertion checks that the description spans the expected matrix tile.

## From reduction to instruction

### Read one lane without confusing it with a whole tile

For this description, lane 0 owns the following coordinates (obtained from `frag_coords()`):

| Fragment | Coordinates in lane 0 | Slots per lane |
| --- | --- | --- |
| A, `(m,k)` | `(0,0), (0,1), (8,0), (8,1), (0,8), (0,9), (8,8), (8,9)` | 8 |
| B, `(k,n)` | `(0,0), (1,0), (8,0), (9,0)` | 4 |
| C, `(m,n)` | `(0,0), (0,1), (8,0), (8,1)` | 4 |

Across 32 lanes, these are 256 A slots, 128 B slots, and 128 C slots. The strengthened assertions verify exact coverage with multiplicity one, not merely set coverage: a duplicated coordinate could otherwise pass a set-based test. This is a property of this selected fragment description, not a universal rule for every tensor-core layout.

Lane 0's output includes `C[0,0]`, whose dot product uses all 16 K values. Its eight A slots and four B slots plainly do not contain that whole dot product. The MMA operation is collective across the warp; do not emulate it as 32 independent scalar matmuls using only each lane's local fragment values. Fragment coordinates describe distributed ownership, while the instruction defines the collective arithmetic.

[postrange.Scheduler][scheduler] applies tensor-core optimization when the target and expression qualify. Codegen transforms the chosen fragment structure; the renderer lowers `Ops.WMMA`.

In the [PTX renderer][ptx], `render_wmma` packs fragment values into registers, emits an `mma.sync.aligned` instruction with shape and dtype fields, and unpacks the accumulator. The exact supported combinations depend on target architecture and the current capability table.

A tile's logical shape, per-lane fragment layout, and global Tensor layout are separate mappings. Getting the tile size right does not establish that each lane loaded the right elements.

## Verify actual hardware use

On a supported NVIDIA device, inspect generated PTX and final machine instructions, verify nonuniform input results, and measure warmed execution. State input and accumulation precision, including TF32-related settings where relevant.

The current chapter validates the fragment mapping and traces compiler source. It makes no NVIDIA performance claim or hardware-execution claim. Older assembly examples remain in the historical chapter linked below.

[tc]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/renderer/tc.py
[scheduler]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/opt/postrange.py
[ptx]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/renderer/ptx.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/cuda-tensor-core-pt1.md).
