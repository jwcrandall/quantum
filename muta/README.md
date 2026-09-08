# MuTA in hardware-agnostic Q#

Reusable implementations of the multiple-triangle Ansatz and the computational
methods in Mantilla Calderón et al.,
[*Measurement-based quantum machine learning*](https://doi.org/10.1103/2snk-m8c6),
Physical Review A **113**, 042421 (2026).

The project includes graph-state MBQC, a circuit translation, continuous and
discrete training, quantum-state classification, a teleportation instrument,
quantum kernels and SVM classification, and the paper's noise-model families.
Its small examples and checks demonstrate the algorithms; they do not reproduce
the paper's numerical figures or establish a learning advantage.

This is an independent Q# project. Load `muta/`, not the whole repository: the
neighboring learning exercises intentionally reuse namespaces and callable names.

## Run locally

Use the modern Python QDK package, pinned to `qdk==1.32.0`. From the repository
root, a separate environment can be prepared with:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r muta/requirements-dev.txt
.venv/bin/python -m pytest muta/tests -q
.venv/bin/python muta/run.py --example gates --seed 42
```

Use the corresponding environment interpreter on other operating systems.
The runner accepts `gates`, `greedy`, `dqn`, `classifier`, `teleportation`,
`kernel`, `noise`, `noise-training`, or `all` as its example. For example:

```sh
.venv/bin/python muta/run.py --example all --seed 42
```

The examples combine the library's objectives and trainers in small experiments.
The checks exercise independent circuit references, phase-sensitive MBQC
behavior, optimizer components, and application identities. A fixed seed makes
a particular simulator run repeatable; it does not establish convergence across
seeds or reproduce a published dataset.

| Example | Small experiment |
| --- | --- |
| `gates` | Train a four-angle wire on Haar-sampled inputs; report held-out infidelity before and after training. |
| `greedy`, `dqn` | Learn a phase gate using one free measurement angle and the chosen discrete optimizer. |
| `classifier` | Jointly train tied wire angles and the polynomial classifier head on a small diagnostic dataset. |
| `teleportation` | Train four outcome-controlled correction angles while holding Bell preparation and analysis fixed. |
| `kernel` | Estimate MuTA kernels with shots and fit a Q# SVM. |
| `noise` | Compare a clean objective, fixed bit-flip and Brownian noisy labels, and per-shot resource depolarization. |
| `noise-training` | Fit three models with bit-flip labels, Brownian labels, and resource noise; evaluate them on clean held-out probes. |

The reusable objectives support larger parameter sets, entangling-gate targets,
and full twenty-angle instrument training; the runner keeps demonstrations small.

For direct use from Python:

```python
from qdk import qsharp

qsharp.init(project_root="muta")
print(qsharp.eval("MuTA.Core.Layer(2, 0, [1])"))
```

## Library organization

| Namespace | Purpose |
| --- | --- |
| [`MuTA.Core`](src/Core.qs) | Open graphs with supplied causal flow; MuTA layers and concatenation; MBQC execution; the Appendix B circuit translation. |
| [`MuTA.Training`](src/Training.qs) | Adam, parameter-shift and finite-difference derivatives, Algorithm 1's slicewise search, and Algorithm 2's neural DQN. |
| [`MuTA.Applications`](src/Applications.qs) | Gate-learning losses, classifier primitives, the teleportation instrument, kernels and SVMs, and sampled noise programs. |
| [`MuTA.Expressivity`](src/Expressivity.qs) | Bounded Pauli-word Lie closure and the conditional variance-bound expression from Appendix C. |
| [`MuTA.Examples`](src/Examples.qs) | Small runnable experiments combining quantum objectives and classical training. |
| [`MuTA.QuantumEntry`](src/QuantumEntry.qs) | A standalone connected-layer quantum workload for QIR export. |

The [paper coverage document](docs/paper-coverage.md) maps each method to the
paper, gives operator conventions, and records implementation choices and limits.
Public callable signatures and preconditions are documented in the Q# sources.

## Build and execute a measurement pattern

`MuTA.Core.Layer(width, tip, targets)` builds five-vertex wires numbered
`5 * row + column`. `tip = -1` with an empty target list creates disconnected
wires. Otherwise each target row receives a triangle connection to the tip.
`Concatenate` joins selected output **vertex IDs** of one pattern to selected
input vertex IDs of the next; unjoined interfaces remain external.

Use `AnglesFromVertices` when translating a paper figure. `Execute` expects
angles in `pattern.Order`, the measured-vertex order, and produces its result
array in the same order. Output vertices do not consume angle parameters.
`ParameterSlices` returns groups of angle indices for the discrete trainers.

This Q# example creates a connected two-wire layer and implements
`exp(-i π X⊗X / 4)` on `|00⟩`:

```qsharp
namespace MyMuTAExample {
    import MuTA.Core.*;
    import Std.Math.*;
    import Std.Measurement.*;

    operation SampleConnectedLayer() : Result[] {
        let pattern = Layer(2, 0, [1]);
        let angles = AnglesFromVertices(pattern, [
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -PI() / 2.0, 0.0, 0.0, 0.0
        ]);
        use input = Qubit[2];
        use output = Qubit[2];
        let branch = Execute(pattern, angles, input, output);
        return MResetEachZ(output);
    }
}
```

Save an operation such as this under `muta/src/` before loading the project to
invoke it from Python. The sample yields correlated `00` or `11` outcomes.
`branch` contains the intermediate measurement outcomes, whose corrections have
already been applied. The input register is consumed and reset; the output
register must start in `|0…0⟩`, be disjoint from the input, and be cleaned up by
the caller. The returned output order is `pattern.Outputs`.

`ExecuteWithNoise` additionally accepts an operation acting once on the complete
entangled resource before measurement. `Compile` and `ApplyCompiled` implement
the corresponding unitary circuit for patterns with equal input and output
counts. The latter is reversible and supports adjoint and controlled use.

## Choose a training method

The trainers accept a Q# objective operation with the contract
`Double[] => Double`, so a caller can supply a shot-based quantum loss, a
classically computed objective, or an application-specific hybrid loss.

- `TrainAdam` optimizes continuous parameters. Use parameter shift only for
  independent Pauli-generated rotations with a linear expectation/fidelity
  objective. Use finite differences or an appropriate classical chain rule for
  tied parameters and nonlinear loss functions.
- `GreedyOptimize` enumerates assignments to consecutive temporal slices,
  with epsilon exploration, restart control, and an evaluation budget. The
  cost grows exponentially with the number of parameters in a search window.
- `TrainDqn` learns a one-hidden-layer ReLU Q network from partial-pattern
  episodes. It uses replay, terminal rewards, Bellman targets, and a distinct
  periodically updated target network.

The discrete trainers accept a configurable alphabet. Choosing
`[0.0, PI()/4.0, PI()/2.0]` matches the paper's nominal GKP angle restriction at
the algorithm level. The quantum reference executes ideal qubit measurements;
it does not prepare GKP states, inject magic states, or compile a physical
Clifford-plus-T instruction stream.

Finite-shot objectives fluctuate. Report evaluation counts, shots per loss,
training and test splits, seeds, and optimizer settings when comparing methods.
Keep noisy-label samples fixed when the intended input is a fixed noisy dataset.

## Application entry points

| Task | Main callables |
| --- | --- |
| Supervised gate learning | `SampleHaarDataset`, `PrepareDatasetState`, `AverageGateInfidelity`; `SampleHaarSingleQubit` and `ApplyIsingXX` construct the paper's target families. |
| Learning with noise | `SampleBitFlipProgram`, `SampleBrownianProgram`, `AverageNoisyLabelInfidelity`, `AverageNoisyResourceInfidelity`. |
| Metrological classification | `PrepareStateFamily`, `TiedClassifierAngles`, `ClassifierProbabilities`, `PolynomialHead`, `QuantumClassifierLoss`. |
| Teleportation instrument | `TeleportWithMuTA`, `TeleportationInfidelity`; `ExactTeleportationParameters` supplies a known correct reference. |
| Classical-data classification | `PrepareMuTAKernel`, `MuTAKernel`, `KernelMatrix`, `TrainSvm`, `PredictSvm`; `PrepareKernel` and `ExactKernel` supply independent references. |

Gate-learning objectives project onto a known, reversible target preparation;
the caller supplies that preparation. The included amplitude-vector preparation
is intended for small simulated datasets and has exponential classical cost.
It is not an efficient loader for arbitrary classical data or unknown quantum
states. Kernel SVM labels use `-1/+1`, while the QFI classifier uses `0/1`.
Finite-shot SWAP estimates can be negative, and an estimated kernel matrix is
not guaranteed positive semidefinite; account for this when interpreting an SVM
fit. The SVM's simplified SMO solver is an implementation choice, with an
explicit iteration limit.

## Hardware portability

There are no provider settings, credentials, device layouts, or tetron-specific
primitives. Algorithms use standard qubits, gates, measurements, reset, and
state-preparation callbacks. Hardware-agnostic source is distinct from universal
execution portability: a target still needs suitable compilation, resources,
mid-circuit measurement, reset, and classical feedforward.

The full hybrid optimizer can run in QDK simulation. A hardware integration can
run the classical optimizer on a host and submit supported quantum loss
evaluations separately. Target selection, error correction, restricted-angle
synthesis, and physical resource preparation belong in that integration.

The included export script demonstrates this boundary by compiling only
`Core.qs` and `QuantumEntry.qs` to QIR using QDK's `Adaptive_RI` profile:

```sh
.venv/bin/python muta/export_qir.py --output /tmp/muta.ll
```

Choose a suitable writable output path on your system. The exporter writes QIR
and does not submit a job. The full hybrid project includes `Std.Random` for
host/simulator-side sampling and cannot be compiled wholesale under that profile.
Successful export verifies a quantum-module compilation path; actual execution
still requires a compatible target and integration.
