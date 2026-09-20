# From a UOp graph to an instruction order

[All tutorials](README.md) · Updated September 21, 2026 · [tinygrad `8ad8f73`](https://github.com/tinygrad/tinygrad/tree/8ad8f738755c3ab157d356aedd5d5c108aa1a642)

Graph edges express dependencies; a renderer generally needs a legal ordered representation. Current tinygrad performs late control-flow construction and linearization in [codegen/late/linearizer.py][linearizer].

A topological traversal is useful for understanding data dependencies, but it is not by itself the complete kernel linearization algorithm. Loops, conditionals, range endings, memory effects, and synchronization constrain placement.

## Distinguish the two orderings

The high-level scheduler orders `CALL` nodes that execute kernels or other work. The late kernel linearizer orders operations within one program. Both use the shared IR vocabulary, but they solve different problems.

For a reduction, input values must be available before the arithmetic that consumes them, accumulation must happen within the appropriate iteration domain, and the final store must occur after the reduction completes. Moving a store outside or inside the wrong range changes the result.

Read [the reduction walkthrough](uops.md) first. Then use [the program inspection example](codegen.md) to compare the kernel's graph with its `PROGRAM`/instruction representation. Finally compare the rendered source.

When a renderer rejects an unexpected operation, investigate which earlier pass was supposed to eliminate or lower it. Adding a textual rendering for an operation is insufficient unless its control, type, and memory semantics are defined at that stage.

[linearizer]: https://github.com/tinygrad/tinygrad/blob/8ad8f738755c3ab157d356aedd5d5c108aa1a642/tinygrad/codegen/late/linearizer.py

Original chapter by Di Zhu: [historical version](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/uops-2.md).
