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

Use Python 3.12, as in CI, and the modern Python QDK package, pinned to
`qdk==1.32.0`. From the repository root, a separate environment can be prepared
with:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r muta/requirements-dev.txt
.venv/bin/python -m pytest muta/tests -q
.venv/bin/python muta/run.py --example gates --seed 42
```

The commands use macOS/Linux paths. On Windows, create the environment with
`py -3.12 -m venv .venv` and use `.venv\Scripts\python.exe` instead.
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

## Train once, then predict

After the environment setup above, the quickest training demonstrations are:

```sh
.venv/bin/python muta/run.py --example classifier --seed 42
.venv/bin/python muta/run.py --example kernel --seed 42
```

`classifier` prints `(initial training loss, reevaluated fitted training loss,
objective evaluations)`, not predicted labels. `kernel` prints `(predicted
labels, kernel matrix)`, but those predictions are on the **training samples**.
Neither command saves a model; every invocation trains from scratch. There are
no separate `train` or `predict` subcommands.

For separate training and prediction runs, the following walkthrough saves the
quantum-state classifier's fitted parameters to JSON and loads them in a new
Python process. Run both blocks from the **repository root** after setup.
The shell blocks work in Bash/Zsh; on other systems, save each Python body as a
script and run it with the environment's interpreter.

### 1. Train and save the classifier

This uses the same six-state diagnostic dataset and Adam configuration as
`ClassifierLearning` in [`src/Examples.qs`](src/Examples.qs). It jointly fits four
shared measurement angles and six polynomial-head coefficients. Finite
differences (`false` in `AdamDefaults`) are required here rather than applying
parameter shift directly to this tied, nonlinear objective.

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path

from qdk import qsharp

qsharp.init(project_root="muta")
qsharp.set_quantum_seed(42)
qsharp.set_classical_seed(42)

parameters, initial_loss, best_loss, evaluations = qsharp.eval("""
    let initial = [0.1, -0.1, 0.1, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let fit = MuTA.Training.TrainAdam(
        MuTA.Examples.JointClassifierObjective,
        initial,
        MuTA.Examples.AdamDefaults(8, 0.08, false)
    );
    (fit.Parameters, fit.History[0], fit.Loss, fit.Evaluations)
""")

model_path = Path("muta-classifier.json")
model_path.write_text(
    json.dumps({"parameters": parameters}, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(f"Initial training loss: {initial_loss}")
print(f"Best observed training loss: {best_loss}; evaluations: {evaluations}")
print(f"Saved {model_path}")
PY
```

The file contains the **best evaluated** parameters, not necessarily the final
optimizer step: positions `0..3` are shared wire angles and `4..9` are head
coefficients. Keep this ordering and the same two-wire pattern for prediction.
Rerunning this block starts a new fit and overwrites `muta-classifier.json`; it
does not resume optimizer state. The printed losses are sampled training
objectives, not held-out accuracy or a guarantee of convergence.

### 2. Load the model and predict a new state

This block does **not** invoke a trainer. It prepares a new two-qubit state from
family `0` with `theta = 0.3` and `phi = 0.4` radians, estimates its output
probabilities with the saved angles, and applies the saved polynomial head.
These state parameters are not one of the six training examples.

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path

from qdk import qsharp

model = json.loads(Path("muta-classifier.json").read_text(encoding="utf-8"))
parameters_qs = json.dumps(model["parameters"], allow_nan=False)

qsharp.init(project_root="muta")
qsharp.set_quantum_seed(43)
qsharp.set_classical_seed(43)

p00, p11, score, label = qsharp.eval(f"""
    let parameters = {parameters_qs};
    let pattern = MuTA.Core.Layer(2, -1, []);
    let angles = MuTA.Applications.TiedClassifierAngles(pattern, parameters[0..3]);
    let probabilities = MuTA.Applications.ClassifierProbabilities(
        pattern,
        angles,
        MuTA.Applications.PrepareStateFamily(0, 0.3, 0.4, _),
        1024
    );
    let score = MuTA.Applications.PolynomialHead(parameters[4..9], probabilities);
    let (p00, p11) = probabilities;
    (p00, p11, score, score > 2.0 ? 1 | 0)
""")

print(f"p00={p00:.4f}, p11={p11:.4f}")
print(f"Classifier score={score:.4f}; predicted label={label}")
PY
```

The classifier's boundary is the standard quantum limit, **2.0**, not zero or
0.5: this example assigns label `1` above it and `0` otherwise. The polynomial
score is not a probability or a guaranteed exact QFI value. Each of the 1,024
shots prepares a fresh copy of the input state; predictions can fluctuate with
sampling, especially near the boundary.

To predict again, rerun only step 2, changing the arguments to
`PrepareStateFamily` or replacing it with your own state-preparation operation.
For your own training set, supply a preparation callback and matching `0/1`
labels to `QuantumClassifierLoss` instead of `JointClassifierObjective`, then
retrain. Use independent labeled states to measure held-out accuracy; this
small walkthrough demonstrates the workflow, not a useful accuracy guarantee.

### Other training and prediction tasks

| Task | What to retain after training | How to use it without retraining |
| --- | --- | --- |
| Continuous or discrete gate learning (`TrainAdam`, `GreedyOptimize`, `TrainDqn`) | `TrainingResult.Parameters` and the matching pattern/parameter mapping. | Reconstruct the full angle vector, prepare a new input, and use `Execute` or `ApplyCompiled`. Measure or otherwise consume the output: these models produce quantum states, not class labels. |
| Classical-data kernel classification (`TrainSvm`) | The `SvmModel` fields **and the ordered training feature vectors**. | For a new two-feature point `x`, compute `MuTAKernel(x, training[i], shots)` for every training point in the original order, then pass that vector to `PredictSvm`. Labels are `-1/+1`. |

The kernel feature map is fixed; `TrainSvm` fits the classical SVM rather than
optimizing its MuTA measurement angles. A kernel matrix of only the new inputs
is not sufficient for prediction: similarities must be against the saved
training points. Python handles model storage and QDK execution in the
walkthrough above; training and prediction calculations remain in Q#.

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
