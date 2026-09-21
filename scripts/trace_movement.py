"""Capture a real transpose/pad/reduction compilation. Run with pinned tinygrad on PYTHONPATH."""
import argparse
from collections import Counter
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

from tinygrad import Tensor
import tinygrad
from tinygrad.device import Device
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops

PIN = "8ad8f738755c3ab157d356aedd5d5c108aa1a642"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  checkout = Path(tinygrad.__file__).resolve().parents[1]
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
  assert revision == PIN, (revision, PIN)
  assert not subprocess.check_output(["git", "diff", "HEAD", "--", "tinygrad"], cwd=checkout), "modified tinygrad sources"
  args.output.mkdir(parents=True, exist_ok=True)
  counts = {}

  def record(name, graph):
    (args.output / f"{name}.txt").write_text(str(graph) + "\n")
    counts[name] = dict(sorted(Counter(u.op.name for u in graph.toposort()).items()))
    return graph

  def observe(module, function, name):
    original = getattr(module, function)
    def wrapped(*a, **kw):
      record("02-prepared", a[0])
      return record(name, original(*a, **kw))
    return patch.object(module, function, wrapped)

  x = Tensor([[1., 2., 3.], [4., 5., 6.]]).realize()
  y = (x.T.pad(((1, 1), (1, 1))) + 10).sum(axis=1)
  record("01-tensor", y.uop)
  rangeify = importlib.import_module("tinygrad.schedule.rangeify")
  with Context(SCACHE=0, BEAM=0), ExitStack() as stack:
    stack.enter_context(observe(rangeify, "run_rangeify", "03-rangeified"))
    linear = record("05-schedule", y.schedule_linear())
    compiled = compile_linear(linear)
    programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM]
    assert len(programs) == 1, "this pinned example is expected to fuse into one program"
    program = programs[0]
    record("06-program", program)
    sources = [u.arg for u in program.src if u.op is Ops.SOURCE]
    assert len(sources) == 1
    (args.output / "07-source.txt").write_text(sources[0] + "\n")
    run_linear(compiled)
  result = y.tolist()
  assert result == [40., 45., 47., 49., 40.], result
  assert all(name in counts for name in ("02-prepared", "03-rangeified"))
  metadata = dict(revision=revision, python=platform.python_version(), device=Device.DEFAULT,
                  renderer=type(Device[Device.DEFAULT].renderer).__name__, target=str(program.arg),
                  settings=dict(SCACHE=0, BEAM=0), result=result, nodes=counts)
  (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
  print(json.dumps({k: v for k, v in metadata.items() if k != "nodes"}, indent=2))


if __name__ == "__main__":
  main()
