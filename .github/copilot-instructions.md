# Copilot Instructions

## Project shape

- This is a modern Q# project for the Azure Quantum Development Kit (QDK).
- `muta/` is an independent Q# project for measurement-based quantum machine learning. Load it with `from qdk import qsharp` and `qsharp.init(project_root="muta")` from the repository root; do not include the neighboring katas in its compilation context.
- `main.qs` is a standalone scratch program that demonstrates allocation, entanglement, state inspection, measurement, and reset.
- `qdk-learning/` is a kata corpus grouped by quantum-computing topic:
  - `exercises/` contains intentionally incomplete operations and functions. Preserve the requested callable name, signature, characteristics, and supplied helpers because the learning environment expects that contract.
  - `examples/` contains self-contained demos and end-to-end checks, usually with an `@EntryPoint()`.
- Treat each kata file as an independent program or exercise, not as part of one shared compilation unit. Many files deliberately reuse the `Kata` namespace and operation names, so do not combine the whole tree into one project.
- `qdk-learning.json` stores the QDK learning course position and completion state.

## Run and validate

### MuTA

Use the Python QDK dependency pinned in `muta/requirements-dev.txt`. CI uses Python 3.12. From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r muta/requirements-dev.txt

# Full MuTA suite.
.venv/bin/python -m pytest muta/tests -q

# One parametrized Q# check (quote the brackets for shell compatibility).
.venv/bin/python -m pytest 'muta/tests/test_muta.py::test_training_algorithms[TestAdam]' -q

# One Python-side contract check.
.venv/bin/python -m pytest muta/tests/test_muta.py::test_angle_and_slice_conventions -q

# One experiment, or all experiments as run in CI.
.venv/bin/python muta/run.py --example gates --seed 42
.venv/bin/python muta/run.py --example all --seed 42

# Compile the isolated quantum workload to Adaptive_RI QIR.
.venv/bin/python muta/export_qir.py --output /tmp/muta.ll
```

Use the equivalent environment interpreter on other operating systems. `.github/workflows/muta.yml` runs the suite and all examples for changes to MuTA or its workflow.

The mathematical checks live in `muta/src/{Core,Training,Application}Tests.qs`; `muta/tests/test_muta.py` initializes a fresh QDK context for each test, sets both quantum and classical seeds, invokes named Q# checks, and asserts expected failures. Register new Q# checks in the relevant pytest parametrization so CI executes them. Core channel comparisons use entangled reference states and independent circuit implementations to detect relative-phase errors, not just measurement-frequency differences.

### Scratch program and katas

Use the QDK VS Code extension or the `qsharp` CLI to validate each standalone file.

```bash
# Run the root program.
qsharp run main.qs

# Run one standalone example or self-checking scenario.
qsharp run qdk-learning/examples/oracles/oracles__test_meeting_oracle.qs

# Run another topic's end-to-end demo.
qsharp run qdk-learning/examples/grovers_search/grovers_search__e2edemo.qs
```

Files under `exercises/` generally have no entry point and are intended to be invoked by the QDK learning environment. For an exercise change, use editor diagnostics on that file and run the corresponding file under `examples/<topic>/` when one exists. Those examples are separate programs, not harnesses that import the exercise implementation. Self-checking kata examples report outcomes with `Message`; inspect failure messages rather than relying only on exit status or a final success message.

## MuTA architecture and contracts

- Quantum primitives, classical optimizers, and application algorithms all live in Q#. Python's `run.py`, `export_qir.py`, and pytest harness manage execution and compiler contexts rather than implement the algorithms.
- `MuTA.Core` constructs open graph patterns with supplied causal flow, composes layers, executes measurement-based computations, and translates equal-input/output patterns into reversible circuits. `MuTA.Applications` builds losses, classifiers, teleportation, kernels/SVMs, and noise models on Core. `MuTA.Training` accepts objective callbacks of type `Double[] => Double`; `MuTA.Examples` connects application objectives to trainers. `MuTA.Expressivity` provides bounded Pauli-word Lie closure and a conditional variance bound.
- `export_qir.py` deliberately loads only `Core.qs` and `QuantumEntry.qs` under `Adaptive_RI`. Preserve this boundary: the full hybrid project uses `Std.Random` for host/simulator sampling and cannot be compiled wholesale under that profile. The exporter generates QIR, not hardware jobs.
- Layer vertices are numbered `5 * row + column`, but angle arrays and returned measurement outcomes use `pattern.Order` slots, not vertex IDs. Outputs have no angle. Use `AnglesFromVertices` for vertex-labelled data and `ParameterSlices` for optimizer slices; slices partition canonical angle indices without reordering the parameter vector. `Concatenate` joins output/input **vertex IDs**, not interface-array positions.
- `Execute` consumes and resets the input register. The output register must be disjoint, start in the zero state, and be cleaned up by the caller; corrected outputs follow `pattern.Outputs`. `ExecuteWithNoise` calls its noise operation once on the complete entangled resource before measurements.
- The equatorial measurement convention gives a corrected wire step `H Rz(-alpha)`. `Execute` applies physical outcome-dependent Pauli corrections while leaving nominal angles unchanged. Preserve the minus sign and output ordering when changing the circuit translation; `ApplyCompiled` remains `Adj + Ctl`.
- Trainers return the best evaluated parameters, not the last explored parameters. `History` records best-so-far loss after every objective evaluation; derivative probes count toward `Evaluations` and the evaluation budget and may improve the incumbent. Parameter shift is valid for independent Pauli-generated rotations with linear expectation/fidelity objectives; tied parameters and nonlinear classifier losses need finite differences or a suitable chain rule.
- Set both QDK quantum and classical seeds for repeatable runs. Keep sampled noisy labels fixed for a fixed corrupted dataset; resource noise is sampled per execution. Resource depolarization uses total nonidentity Pauli probability `p`, with `p/3` for each of X, Y, and Z. SVM labels are `-1/+1`; QFI classifier labels are `0/1`.
- Consult `muta/README.md` for public usage and `muta/docs/paper-coverage.md` for operator conventions, paper mappings, and implementation limits. Keep claims scoped to the implemented algorithms and small demonstrations, rather than reproduction of the paper's numerical figures or a demonstrated learning advantage.

## Q# conventions

- Use modern imports such as `import Std.Diagnostics.*;` inside `namespace Kata { ... }`. The root `main.qs` is intentionally top-level and imports before its operation.
- Allocate qubits with `use`. Return every allocated qubit to `|0⟩` before it leaves scope:
  - Prefer `MResetZ`, `MResetEachZ`, or `ResetAll` when measurement and cleanup can be combined.
  - Otherwise call `Reset` explicitly after measurement or before returning.
- Use `DumpMachine()` from `Std.Diagnostics` to inspect simulator state during development; avoid making diagnostic dumps part of an exercise's required behavior unless the example is specifically about diagnostics.
- Preserve quantum/classical type distinctions used throughout the katas: measurements produce `Result`; APIs often expose classical values by comparing with `Zero`/`One` or using helpers such as `ResultArrayAsBoolArray`.
- Operations used as oracles or state preparations commonly require `is Adj + Ctl`. Keep implementations reversible, preserve input registers, and toggle only the documented target.
- Use `within { ... } apply { ... }` when temporary basis changes or ancilla computations must be uncomputed automatically. This is the standard pattern in the oracle, Grover search, SAT, and error-correction examples.
- Follow the existing callable shapes when passing operations as values, including characteristics, for example `(Qubit[], Qubit) => Unit is Adj + Ctl`.
- Keep a kata solution local to its file and reuse the helper operations supplied beneath the exercise stub instead of introducing cross-file dependencies.
- Examples commonly validate probabilistic behavior by repeating an experiment, and validate reversible operations by comparing with a classical implementation and checking that input registers return unchanged.
