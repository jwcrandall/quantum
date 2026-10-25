# Quantum experiments

- [MuTA in Q#](muta/README.md): hardware-agnostic measurement-based quantum
  machine learning, including graph execution, circuit translation, continuous
  and discrete trainers, application examples, and automated checks.
  Start with the [training and prediction walkthrough](muta/README.md#train-once-then-predict)
  to fit a classifier, save its parameters, and load them for a separate prediction run.
  The [baseline notebook](muta/notebooks/hardware_agnostic_baseline.ipynb) runs a
  five-seed training, held-out inference, and robustness battery and exports
  Figure 1 with supporting diagnostics and reproducibility artifacts.
- [QDK learning exercises](qdk-learning/): independent kata files grouped by
  topic. These intentionally reuse names and should be opened individually.
- [Bell-state scratch program](main.qs): a standalone introductory example.

Each directory or standalone example has its own execution context; the kata
corpus is not a single Q# compilation unit.
