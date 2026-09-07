# Copilot Instructions

## Project shape

- This is a modern Q# project for the Azure Quantum Development Kit (QDK).
- `main.qs` is a standalone scratch program that demonstrates allocation, entanglement, state inspection, measurement, and reset.
- `qdk-learning/` is a kata corpus grouped by quantum-computing topic:
  - `exercises/` contains intentionally incomplete operations and functions. Preserve the requested callable name, signature, characteristics, and supplied helpers because the learning environment expects that contract.
  - `examples/` contains self-contained demos and end-to-end checks, usually with an `@EntryPoint()`.
- Treat each kata file as an independent program or exercise, not as part of one shared compilation unit. Many files deliberately reuse the `Kata` namespace and operation names, so do not combine the whole tree into one project.
- `qdk-learning.json` stores the QDK learning course position and completion state.

## Run and validate

Use the QDK VS Code extension or the `qsharp` CLI. There is no repository-wide build, test, or lint configuration; validate the individual file being changed.

```bash
# Run the root program.
qsharp run main.qs

# Run one standalone example or self-checking scenario.
qsharp run qdk-learning/examples/oracles/oracles__test_meeting_oracle.qs

# Run another topic's end-to-end demo.
qsharp run qdk-learning/examples/grovers_search/grovers_search__e2edemo.qs
```

Files under `exercises/` generally have no entry point and are intended to be invoked by the QDK learning environment. For an exercise change, use editor diagnostics on that file and run the corresponding file under `examples/<topic>/` when one exists. Self-checking examples report outcomes with `Message`; there is no separate unit-test runner or single-test selector.

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
