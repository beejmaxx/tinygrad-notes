"""Execute each current tutorial's Python blocks against a specified tinygrad checkout."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

PIN = "8ad8f738755c3ab157d356aedd5d5c108aa1a642"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkout", type=Path)
  parser.add_argument("--device", default="CPU")
  parser.add_argument("--allow-other-revision", action="store_true")
  args = parser.parse_args()
  if sys.version_info < (3, 11):
    parser.error("tinygrad requires Python 3.11 or newer; use that interpreter to run this script")
  checkout = args.checkout.resolve()
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
  if revision != PIN and not args.allow_other_revision:
    parser.error(f"expected {PIN}, found {revision}; use --allow-other-revision for a forward-compatibility check")
  root = Path(__file__).resolve().parents[1]
  failures = []
  count = 0
  print(f"Python {sys.version.split()[0]} | {args.device} | tinygrad {revision}", flush=True)
  for path in sorted(root.glob("*.md")):
    source = path.read_text()
    if "Updated September 21, 2026" not in source:
      continue
    for block, code in enumerate(re.findall(r"```python\n(.*?)```", source, re.S), 1):
      count += 1
      name = f"{path.name}:{block}"
      try:
        result = subprocess.run(
          [sys.executable, "-c", code], cwd=checkout,
          env={**os.environ, "PYTHONPATH": str(checkout), "DEV": args.device, "BEAM": "0", "JIT": "1"},
          text=True, capture_output=True, timeout=90,
        )
        if result.returncode:
          failures.append(name)
          print(f"FAIL {name}\n{result.stdout}{result.stderr}", flush=True)
        else:
          print(f"PASS {name}", flush=True)
      except subprocess.TimeoutExpired:
        failures.append(name)
        print(f"FAIL {name}: exceeded 90 seconds", flush=True)
  print(f"{count - len(failures)}/{count} examples passed", flush=True)
  return bool(failures) or count == 0


if __name__ == "__main__":
  raise SystemExit(main())
