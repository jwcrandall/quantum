"""Compile the isolated quantum workload to Adaptive_RI QIR; does not submit jobs."""

import argparse
from pathlib import Path

from qdk import qsharp


def compile_quantum_entry():
    src = Path(__file__).resolve().parent / "src"
    # Do not initialize the full hybrid project: Std.Random is a host/simulator
    # facility, not part of the adaptive hardware profile. Load the quantum
    # module and its entry point without importing training or noise samplers.
    qsharp.init(target_profile=qsharp.TargetProfile.Adaptive_RI)
    for name in ("Core.qs", "QuantumEntry.qs"):
        qsharp.eval((src / name).read_text(encoding="utf-8"))
    return str(qsharp.compile("MuTA.QuantumEntry.SampleConnectedLayer()"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qir = compile_quantum_entry()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(qir, encoding="utf-8")
    print(f"Wrote Adaptive_RI QIR to {args.output}")


if __name__ == "__main__":
    main()
