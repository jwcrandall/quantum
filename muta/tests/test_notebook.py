"""Execute the real notebook with tiny budgets; the saved reference uses five seeds."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

for dependency in ("numpy", "matplotlib", "nbformat", "nbclient", "nbconvert", "ipykernel"):
    pytest.importorskip(dependency, reason="Install muta/requirements-notebook.txt for notebook tests.")

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks" / "hardware_agnostic_baseline.ipynb"


def load_benchmarks():
    spec = importlib.util.spec_from_file_location("muta_benchmarks_test", PROJECT / "benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_notebook_end_to_end(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    configuration = next(cell for cell in notebook.cells if "parameters" in cell.metadata.get("tags", []))
    configuration.source = (
        "config = benchmarks.Config("
        "seeds=(3, 7), train_states=2, test_states=4, train_shots=2, test_shots=4, "
        "gate_steps=1, entangler_steps=1, discrete_budget=10, "
        "classifier_train_per_family=2, classifier_test_per_family=2, "
        "classifier_steps=1, classifier_shots=2, teleport_steps=1, "
        "teleport_full_steps=1, teleport_shots=2, teleport_test_shots=16, "
        "noise_steps=1, kernel_train_per_class=1, kernel_test_per_class=1, "
        "kernel_shots=4, shot_sweep=(4, 8), resource_noise=(0.0, 0.05))\n"
        f"OUTPUT = Path({str(tmp_path)!r})\n"
        "started = perf_counter()\n"
    )
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT.parent)}},
    )
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    executed = client.execute()
    code = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    assert sum(
        "image/png" in output.get("data", {}) for cell in code for output in cell.outputs
    ) == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["artifacts_sha256"]) == {
        "datasets.json", "models.json", "results.json", "summary.csv", "figure-1-caption.txt",
        "figure-1.pdf", "figure-1.svg", "figure-1.png", "diagnostics.pdf", "diagnostics.svg",
        "diagnostics.png",
    }
    for name in manifest["artifacts_sha256"]:
        assert (tmp_path / name).stat().st_size > 0


def test_saved_model_rejects_changed_data_and_sources(tmp_path):
    benchmarks = load_benchmarks()
    config = benchmarks.Config(seeds=(3,))
    benchmarks.initialize()
    datasets = benchmarks.draw_data(config)
    benchmarks.save_training(tmp_path, datasets, {}, config)
    loaded, _ = benchmarks.load_training(tmp_path, config)
    assert loaded == json.loads(json.dumps(datasets))
    path = tmp_path / "datasets.json"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="datasets changed"):
        benchmarks.load_training(tmp_path, config)
    path.write_bytes(original)
    bundle_path = tmp_path / "models.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["source_sha256"]["src/Core.qs"] = "changed"
    benchmarks.write_json(bundle_path, bundle)
    with pytest.raises(ValueError, match="sources changed"):
        benchmarks.load_training(tmp_path, config)
