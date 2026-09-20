# Backends: renderer, compiler, allocation, and launch

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

A backend is more than a source-code printer. It must connect the shared compiler representation to a device's memory, executable programs, launch interface, and synchronization rules.

The current common interfaces live in [device.py][device]. Concrete runtimes live under `tinygrad/runtime/ops_*.py`; renderers live under `tinygrad/renderer`.

## Identify the selected target

```python
from tinygrad import Tensor, Device

x = Tensor([1.0, 2.0, 3.0]).realize()
device = Device[x.device]
print("Tensor device:", x.device)
print("renderer:", type(device.renderer).__name__)
assert (x + 1).tolist() == [2.0, 3.0, 4.0]
```

Use `DEV=CPU` to request CPU execution. `DEV=CPU:LLVM` selects a renderer explicitly when its dependencies are available. Other backends have their own target/interface requirements; consult the pinned [runtime documentation][runtime].

## The responsibilities

| Component | Responsibility |
| --- | --- |
| Renderer | Lowered operation representation and target capabilities; produce target source/representation. |
| Compiler | Turn that representation into executable bytes, with compilation caching where applicable. |
| Allocator | Allocate/free storage and support copies, transfers, or host access as the device allows. |
| Program | Load the compiled object and execute it with buffers, scalar arguments, and launch dimensions. |
| Compiled device | Compose these facilities and provide synchronization and optional graph support. |

Current `Program` initialization receives a device and `TinyELF` object. Older snippets that instantiate a program using only a name and byte string should not be used as a backend template.

## Trace CPU as a concrete implementation

[CPUDevice][cpu] selects among available CPU renderers and uses `HostAllocator` and `CPUProgram`. The program loads executable code, handles platform-specific executable memory, and invokes the function through a native call interface.

A CPU renderer can express loops differently from a GPU renderer that maps work to global/local dimensions. Backend capability declarations therefore affect optimization, not just the last textual output.

## Debug a boundary at a time

If results are wrong, first inspect the shared kernel and then generated target code. If those are correct, check argument layout, allocation size, load/launch behavior, and synchronization. For performance, separate compilation, copies, launch overhead, and kernel execution.

A passing CPU example does not verify another device's runtime. Keep backend-specific tests and observed hardware results explicit.

[device]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/device.py
[cpu]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/runtime/ops_cpu.py
[runtime]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/docs/runtime.md

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/backends.md).
