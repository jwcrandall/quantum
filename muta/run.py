"""Run Q# examples. Algorithm and optimizer implementations live in src/*.qs."""

import argparse
from pathlib import Path

from qdk import qsharp


EXAMPLES = {
    "gates": "GateLearning",
    "greedy": "GreedyLearning",
    "dqn": "DqnLearning",
    "classifier": "ClassifierLearning",
    "teleportation": "TeleportationLearning",
    "kernel": "KernelClassification",
    "noise": "NoiseComparison",
    "noise-training": "NoisyGateLearning",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=[*EXAMPLES, "all"], default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    qsharp.init(project_root=str(Path(__file__).resolve().parent))
    qsharp.set_quantum_seed(args.seed)
    qsharp.set_classical_seed(args.seed)
    selected = EXAMPLES if args.example == "all" else {args.example: EXAMPLES[args.example]}
    for name, operation in selected.items():
        print(f"{name}: {qsharp.eval(f'MuTA.Examples.{operation}()')}", flush=True)


if __name__ == "__main__":
    main()
