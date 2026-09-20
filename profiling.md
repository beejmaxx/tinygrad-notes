# Profiling compilation, execution, and replay

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

A useful timing states what starts and stops the clock. tinygrad workloads can spend time in Python graph construction, scheduling, code generation, native compilation, allocation, transfer, launch, or device execution.

## Measure a warmed replay with synchronization

```python
import time
from tinygrad import Tensor, TinyJit, Device

x = Tensor([float(i) for i in range(256)]).realize()

@TinyJit
def step(a):
  return (a * 2 + 1).realize()

step(x)
step(x)
Device[x.device].synchronize()
start = time.perf_counter()
for _ in range(10):
  result = step(x)
Device[x.device].synchronize()
elapsed = time.perf_counter() - start
assert result.tolist() == [float(i * 2 + 1) for i in range(256)]
print("seconds per warmed call:", elapsed / 10)
```

The first two calls cover ordinary execution and capture. Synchronizing before the timer excludes previous queued work; synchronizing afterward includes completion. Result conversion occurs outside the measured interval.

This measures warmed end-to-end replay including Python/launch overhead. It is not a pure kernel benchmark, and the tiny workload is chosen for clarity rather than meaningful peak throughput.

## Use debug output and recording

`DEBUG=2` exposes operation timings and statistics; `DEBUG=4` adds generated source. [VIZ](20241129_viz.md) records execution and graph transformations for deeper inspection.

A profiler can change execution behavior. Debug timing may force waiting where the normal program overlaps work. Compare performance with an appropriate low-overhead measurement after using detailed traces to identify the cause.

## Interpret throughput carefully

Reported operations and memory traffic are compiler estimates. They need not equal physical device instructions or actual off-chip bytes. Cache reuse, fusion, vectorization, tensor cores, and transfers change how estimates relate to hardware counters.

Record backend, renderer, device, dtype, input sizes, environment settings, commit, warmup, and cache conditions. Separate cold compilation/search cost from warm execution. If evaluating an optimization, compare equivalent workloads and verify results before trusting the timing.

See the pinned [speed overview][speed] and [execution statistics implementation][realize] for the current definitions.

[speed]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/docs/developer/speed.md
[realize]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/engine/realize.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/profiling.md).
