# Setup and how to use these notes

[All tutorials](README.md) · Updated September 21, 2026

These tutorials follow [tinygrad commit `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642), a master snapshot after the 0.14.0 release. Internal APIs change frequently; a source link's pinned commit is part of the example's context.

## Install the documented revision

Use Python 3.11 or newer. Check `python3 --version`; some Macs still resolve that name to an older system Python.

```sh
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
git switch --detach 8ad8f738755c3ab157d356aedd5d5c108aa1a642
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

Use a fresh clone for this command sequence. Detaching at a known commit makes examples reproducible; use a separate branch/current checkout when developing a contribution.

For the portable examples, use `DEV=CPU`. The CPU compiler path needs its native compiler dependencies; consult the [pinned runtime documentation](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/docs/runtime.md) if your environment cannot compile a kernel.

## Read the implementation in order

Start with [the introduction](20241231_intro.md), then [movement and indexing](20241217_st.md), [scheduling](scheduleitem.md), and [code generation](codegen.md).

The earlier tutorials used ShapeTracker, LazyBuffer, ScheduleItem, and a Linearizer class. Today's chapters explain the current representations instead:

| Earlier explanation | Current starting point |
| --- | --- |
| ShapeTracker/View transformations | Movement UOps and `schedule/indexing.py` coordinate propagation |
| LazyBuffer graph | Tensor `.uop` graph |
| ScheduleItem lists | LINEAR/CALL UOps and buffer-state dependencies |
| Linearizer optimization methods | Kernel range optimization in `codegen/opt/postrange.py` |
| ALU opcode with an arithmetic argument | Individual ADD, MUL, and other operation nodes |
| JIT as a second compilation step | Captured execution graph and replay |

These are conceptual transitions, not a table of interchangeable imports. Read the linked chapters before porting an old snippet.

## Validation coverage

The [worked compiler trace](movement-trace.md) adds captured Tensor, prepared, rangeified, scheduled, and compiled graphs, with executed CPU arm64 and Metal Apple7 output. The example suite also checks 15 transpose/flip/asymmetric-pad/broadcast/reduction combinations against a Python oracle, JIT replay with changed data and incompatible-shape rejection, and symbolic means at three bindings. CI regenerates a CPU trace to detect broken instrumentation; it does not compare target-dependent emitted source byte for byte.

The 36 Python code blocks in the revised chapters are self-contained and executable. The example runner launches each in a separate process with ordinary JIT enabled. Assertions check values, graph properties, coordinate mappings, capture behavior, or fragment layouts as appropriate. GitHub Actions runs these examples on Linux with Python 3.12 for each change. The local validation targets are CPU and Metal with Python 3.14.6 on macOS against the pinned checkout.

From the notes repository, using the Python environment containing tinygrad's dependencies:

```sh
python3 scripts/check_examples.py /path/to/tinygrad
```

Use `--device METAL` on a supported Mac to repeat the suite there. Use `--allow-other-revision` deliberately when checking a newer checkout; passing on one revision does not guarantee future compatibility.

Shell commands for BEAM tuning, VIZ inspection, and hardware exercises are not silently counted as executed Python examples. CUDA tensor-core layout checks run on the host; they do not execute NVIDIA instructions. Multi-GPU transfers and collectives require real multi-device testing. No portable hardware speedup numbers are claimed.

## Provenance and contributions

Di Zhu's original articles remain available through the historical link at the end of each revised chapter. The original screenshots and image files remain in the repository, but current explanations do not present historical screenshots as captures of this revision.

This fork's updates were made with AI assistance. For upstream tinygrad contributions, read the current [README contribution rules](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/README.md#contributing) and [AGENTS.md](https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/AGENTS.md), including AI disclosure requirements. These tutorials are independent of the project.
