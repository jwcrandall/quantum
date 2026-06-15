# Copilot Instructions

## Language & Runtime

This is a Q# (Q Sharp) project for quantum computing. Q# files use the `.qs` extension and run on the Azure Quantum Development Kit (QDK).

## Running

Run programs using the Azure QDK VS Code extension or the `qsharp` CLI:

```bash
# Run the main program
qsharp run main.qs
```

## Conventions

- Always `Reset()` qubits before they go out of scope to avoid leaving them in an unknown state.
- Use `DumpMachine()` from `Std.Diagnostics` for debugging quantum state during development.
- Import namespaces with `import Std.*` syntax (modern Q# style, not `open` statements).
