"""Notebook orchestration and analysis; quantum algorithms and fits remain in Q#."""

from dataclasses import asdict, dataclass
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from statistics import mean, stdev
import subprocess
from time import perf_counter

from qdk import qsharp


PROJECT = Path(__file__).resolve().parent
NAMESPACE = "MuTA.Benchmarks"
CLASSIFIER_INITIAL = [0.1, -0.1, 0.1, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
NOISE_CASES = (
    ("clean", 0, 0.0),
    ("bitflip_0.1", 1, 0.1),
    ("bitflip_0.3", 1, 0.3),
    ("brownian_0.1", 2, 0.1),
    ("brownian_0.3", 2, 0.3),
    ("resource_0.01", 3, 0.01),
    ("resource_0.05", 3, 0.05),
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (11, 29, 47, 71, 101)
    train_states: int = 6
    test_states: int = 24
    train_shots: int = 16
    test_shots: int = 256
    gate_steps: int = 24
    entangler_steps: int = 24
    discrete_budget: int = 97
    classifier_train_per_family: int = 6
    classifier_test_per_family: int = 16
    classifier_steps: int = 12
    classifier_shots: int = 32
    teleport_steps: int = 16
    teleport_full_steps: int = 12
    teleport_shots: int = 8
    teleport_test_shots: int = 128
    noise_steps: int = 12
    kernel_train_per_class: int = 4
    kernel_test_per_class: int = 8
    kernel_shots: int = 128
    shot_sweep: tuple[int, ...] = (32, 128, 512)
    resource_noise: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1, 0.2)

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide nonempty, distinct experiment seeds.")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be nonnegative integers.")
        for name, value in asdict(self).items():
            if name in {"seeds", "shot_sweep", "resource_noise"}:
                continue
            minimum = 0 if name.endswith("_steps") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for count in (self.classifier_train_per_family, self.classifier_test_per_family):
            if count < 2 or count % 2:
                raise ValueError("Classifier counts must be even and at least two.")
        if self.discrete_budget < 10:
            raise ValueError("The discrete comparison needs at least one Adam update.")
        if not self.shot_sweep or any(type(s) is not int or s < 1 for s in self.shot_sweep):
            raise ValueError("The shot sweep requires positive integer shot counts.")
        if len(set(self.shot_sweep)) != len(self.shot_sweep):
            raise ValueError("Shot sweep values must be distinct.")
        if not self.resource_noise or any(
            not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in self.resource_noise
        ):
            raise ValueError("Resource noise probabilities must lie in [0, 1].")


def initialize():
    if version("qdk") != "1.32.0":
        raise RuntimeError("Use the pinned QDK in muta/requirements-notebook.txt.")
    qsharp.init(project_root=str(PROJECT))


def seed_value(seed, stream):
    key = f"muta-reference-v1:{seed}:{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def set_seeds(seed, stream):
    qsharp.set_quantum_seed(seed_value(seed, f"{stream}:quantum"))
    qsharp.set_classical_seed(seed_value(seed, f"{stream}:classical"))


def literal(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def tuples(rows):
    return "[" + ",".join("(" + ",".join(literal(v) for v in row) + ")" for row in rows) + "]"


def call(expression, seed, stream):
    set_seeds(seed, stream)
    return qsharp.eval(expression)


def draw_data(config):
    """Draw train/test populations with independent named RNG streams."""
    datasets = {}
    for seed in config.seeds:
        run = {}
        for name, width in (("one_qubit", 1), ("two_qubit", 2)):
            run[name] = {
                split: call(
                    f"{NAMESPACE}.HaarData({width}, {count})", seed, f"data:{name}:{split}"
                )
                for split, count in (("train", config.train_states), ("test", config.test_states))
            }
        run["classifier"] = {
            split: call(
                f"{NAMESPACE}.ClassifierData({count})", seed, f"data:classifier:{split}"
            )
            for split, count in (
                ("train", config.classifier_train_per_family),
                ("test", config.classifier_test_per_family),
            )
        }
        run["kernels"] = {}
        for task, name in enumerate(("clusters", "xor")):
            run["kernels"][name] = {}
            for split, count in (
                ("train", config.kernel_train_per_class),
                ("test", config.kernel_test_per_class),
            ):
                points, labels = call(
                    f"{NAMESPACE}.KernelData({task}, {count})",
                    seed,
                    f"data:kernel:{name}:{split}",
                )
                run["kernels"][name][split] = {"points": points, "labels": labels}
        datasets[str(seed)] = run
    validate_datasets(datasets, config)
    return datasets


def validate_datasets(datasets, config):
    if set(datasets) != {str(seed) for seed in config.seeds}:
        raise ValueError("Dataset seeds differ from the configured experiment seeds.")
    for run in datasets.values():
        for name in ("one_qubit", "two_qubit", "classifier"):
            train = {literal(row) for row in run[name]["train"]}
            test = {literal(row) for row in run[name]["test"]}
            if train & test:
                raise ValueError(f"{name} contains train/test overlap.")
            if len(train) != len(run[name]["train"]) or len(test) != len(run[name]["test"]):
                raise ValueError(f"{name} contains duplicate samples.")
        for name in ("one_qubit", "two_qubit"):
            if len(run[name]["train"]) != config.train_states or len(run[name]["test"]) != config.test_states:
                raise ValueError(f"{name} sizes differ from the configured protocol.")
        for split, count in (
            ("train", config.classifier_train_per_family),
            ("test", config.classifier_test_per_family),
        ):
            samples = run["classifier"][split]
            for family in (0, 1):
                for label in (0, 1):
                    if sum(s[0] == family and s[4] == label for s in samples) != count // 2:
                        raise ValueError("Classifier data must be balanced within each family.")
        for task in run["kernels"].values():
            train = {literal(row) for row in task["train"]["points"]}
            test = {literal(row) for row in task["test"]["points"]}
            if train & test:
                raise ValueError("Kernel data contains train/test overlap.")
            for split, count in (
                ("train", config.kernel_train_per_class), ("test", config.kernel_test_per_class)
            ):
                data = task[split]
                if len(data["points"]) != 2 * count or len({literal(p) for p in data["points"]}) != 2 * count:
                    raise ValueError("Kernel sample counts differ from the configured protocol.")
                if sorted(data["labels"]) != [-1] * count + [1] * count:
                    raise ValueError("Kernel labels must be balanced and match the point count.")


def fit(expression, seed, name, initial, samples, shots, patterns_per_shot=1):
    start = perf_counter()
    parameters, loss, history, evaluations = call(
        f"let fit = {expression}; (fit.Parameters, fit.Loss, fit.History, fit.Evaluations)",
        seed,
        f"fit:{name}",
    )
    if (
        len(history) != evaluations
        or len(parameters) != len(initial)
        or not all(math.isfinite(x) for x in [*parameters, *history, loss])
        or any(b > a + 1e-12 for a, b in zip(history, history[1:]))
        or not math.isclose(history[-1], loss, abs_tol=1e-12)
    ):
        raise RuntimeError(f"Invalid training result for {name}, seed {seed}.")
    seconds = perf_counter() - start
    print(f"  {name}, seed {seed}: {evaluations} evaluations, {seconds:.1f} s", flush=True)
    return {
        "name": name,
        "seed": seed,
        "initial": initial,
        "parameters": parameters,
        "training_best_loss": loss,
        "history": history,
        "evaluations": evaluations,
        "fit_seconds": seconds,
        "sample_shots": samples * shots * evaluations,
        "mbqc_executions": samples * shots * evaluations * patterns_per_shot,
    }


def train_gates(datasets, config):
    models = []
    settings = (
        ("single", 0, 0, config.gate_steps),
        ("entangling", 1, 0, config.entangler_steps),
        ("restricted_adam", 2, 0, (config.discrete_budget - 1) // 9),
        ("restricted_greedy", 2, 1, 0),
        ("restricted_dqn", 2, 2, 0),
    )
    for seed in config.seeds:
        for name, task, optimizer, steps in settings:
            data_name = "two_qubit" if task == 1 else "one_qubit"
            data = datasets[str(seed)][data_name]["train"]
            initial = [0.0] * (8 if task == 1 else 4)
            model = fit(
                f"{NAMESPACE}.FitGate({task}, {optimizer}, {tuples(data)}, {literal(initial)}, "
                f"{config.train_shots}, {steps}, {config.discrete_budget})",
                seed, name, initial, len(data), config.train_shots,
            )
            models.append({**model, "task": task, "optimizer": optimizer, "data_name": data_name})
    return models


def train_classifiers(datasets, config):
    models = []
    for seed in config.seeds:
        data = datasets[str(seed)]["classifier"]["train"]
        models.append(fit(
            f"{NAMESPACE}.FitClassifier({tuples(data)}, {literal(CLASSIFIER_INITIAL)}, "
            f"{config.classifier_shots}, {config.classifier_steps})",
            seed, "classifier", CLASSIFIER_INITIAL.copy(), len(data), config.classifier_shots,
        ))
    return models


def train_teleportation(datasets, config):
    models = []
    for seed in config.seeds:
        data = datasets[str(seed)]["one_qubit"]["train"]
        for full, steps in ((False, config.teleport_steps), (True, config.teleport_full_steps)):
            name = "teleport_full" if full else "teleport_corrections"
            initial = (
                call(f"{NAMESPACE}.RandomAngles(20)", seed, "initial:teleport_full")
                if full else [-2.2, -2.2, 0.0, 0.0]
            )
            model = fit(
                f"{NAMESPACE}.FitTeleport({tuples(data)}, {literal(initial)}, "
                f"{config.teleport_shots}, {steps}, {literal(full)})",
                seed, name, initial, len(data), config.teleport_shots, 3,
            )
            expanded = qsharp.eval(
                f"({NAMESPACE}.TeleportParameters({literal(initial)}, {literal(full)}), "
                f"{NAMESPACE}.TeleportParameters({literal(model['parameters'])}, {literal(full)}))"
            )
            models.append({**model, "full": full, "initial_instrument": expanded[0],
                           "instrument": expanded[1]})
    return models


def train_noise(datasets, config):
    models = []
    for seed in config.seeds:
        data = datasets[str(seed)]["one_qubit"]["train"]
        for name, mode, level in NOISE_CASES:
            noise = call(
                f"{NAMESPACE}.LabelNoise({len(data)}, {mode}, {level})",
                seed, f"labels:{name}",
            )
            initial = [0.0] * 4
            model = fit(
                f"{NAMESPACE}.FitNoisyGate({tuples(data)}, {literal(initial)}, {mode}, "
                f"{level}, {tuples(noise)}, {config.train_shots}, {config.noise_steps})",
                seed, f"noise:{name}", initial, len(data), config.train_shots,
            )
            models.append({**model, "condition": name, "mode": mode, "level": level,
                           "fixed_label_programs": noise})
    return models


def train_kernels(datasets, config):
    models = []
    for seed in config.seeds:
        for task in ("clusters", "xor"):
            data = datasets[str(seed)]["kernels"][task]["train"]
            for exact in (False, True):
                name = f"kernel:{task}:{'exact' if exact else 'shots'}"
                start = perf_counter()
                alpha, bias, iterations, stable, matrix = call(
                    f"let (model, matrix) = {NAMESPACE}.FitKernel("
                    f"{literal(data['points'])}, {literal(data['labels'])}, "
                    f"{config.kernel_shots}, {literal(exact)}); "
                    "(model.Alpha, model.Bias, model.Iterations, model.StablePasses, matrix)",
                    seed, f"fit:{name}",
                )
                seconds = perf_counter() - start
                count = len(data["points"])
                shots = 0 if exact else count * (count - 1) // 2 * config.kernel_shots
                models.append({
                    "name": name, "seed": seed, "task": task, "exact": exact,
                    "alpha": alpha, "bias": bias, "iterations": iterations,
                    "stable_passes": stable, "gram": matrix,
                    "training_points": data["points"], "training_labels": data["labels"],
                    "fit_seconds": seconds, "sample_shots": shots, "mbqc_executions": 2 * shots,
                })
                print(f"  {name}, seed {seed}: {iterations} SMO passes, {seconds:.1f} s", flush=True)
    return models


def source_hashes():
    files = sorted((PROJECT / "src").glob("*.qs")) + [
        Path(__file__), PROJECT / "qsharp.json",
        PROJECT / "requirements-dev.txt", PROJECT / "requirements-notebook.txt"
    ]
    return {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def save_training(directory, datasets, models, config):
    directory = Path(directory)
    data_path = write_json(directory / "datasets.json", datasets)
    return write_json(directory / "models.json", {
        "schema_version": 1,
        "config": asdict(config),
        "source_sha256": source_hashes(),
        "datasets_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "models": models,
    })


def load_training(directory, config):
    directory = Path(directory)
    bundle = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    raw_data = (directory / "datasets.json").read_bytes()
    if bundle["schema_version"] != 1 or literal(bundle["config"]) != literal(asdict(config)):
        raise ValueError("Saved-model schema or configuration differs from this notebook.")
    if bundle["source_sha256"] != source_hashes():
        raise ValueError("MuTA sources changed after training; rerun training before inference.")
    if bundle["datasets_sha256"] != hashlib.sha256(raw_data).hexdigest():
        raise ValueError("Saved datasets changed after training.")
    data = json.loads(raw_data)
    validate_datasets(data, config)
    return data, bundle["models"]


def balanced_accuracy(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Accuracy requires matching nonempty label/prediction arrays.")
    classes = sorted(set(labels))
    if len(classes) != 2:
        raise ValueError("Balanced accuracy requires both classes.")
    return mean(
        sum(y == label and p == label for y, p in zip(labels, predictions)) / labels.count(label)
        for label in classes
    )


def classifier_metrics(data, raw):
    labels = [row[4] for row in data]
    predictions = [row[3] for row in raw]
    by_family = []
    for family in (0, 1):
        indices = [i for i, row in enumerate(data) if row[0] == family]
        by_family.append(balanced_accuracy(
            [labels[i] for i in indices], [predictions[i] for i in indices]
        ))
    return {
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "by_family": by_family,
        "predictions": predictions,
        "scores": [row[2] for row in raw],
        "probabilities": [list(row[:2]) for row in raw],
    }


def gate_inference(model, data, config, parameters, noise, stream):
    return call(
        f"{NAMESPACE}.GateInference({model.get('task', 0)}, {literal(parameters)}, "
        f"{tuples(data)}, {config.test_shots}, {float(noise)})",
        model["seed"], stream,
    )


def infer_gates(models, datasets, config):
    results = []
    for model in models:
        start = perf_counter()
        data = datasets[str(model["seed"])][model["data_name"]]
        stream = f"inference:{model['name']}"
        before = gate_inference(model, data["test"], config, model["initial"], 0.0, stream + ":initial")
        after = gate_inference(model, data["test"], config, model["parameters"], 0.0, stream + ":fit")
        train = gate_inference(model, data["train"], config, model["parameters"], 0.0, stream + ":train")
        compiled = call(
            f"{NAMESPACE}.CompilationInfidelity({model['task']}, {literal(model['parameters'])}, "
            f"{tuples(data['test'])}, 4)", model["seed"], stream + ":compiled",
        )
        results.append({
            "name": model["name"], "seed": model["seed"], "initial_infidelity": mean(before),
            "test_infidelity": mean(after), "train_infidelity": mean(train),
            "sample_infidelities": after, "initial_sample_infidelities": before,
            "compiled_infidelity": compiled, "inference_seconds": perf_counter() - start,
        })
        print(f"  {model['name']}, seed {model['seed']}: held-out infidelity {mean(after):.4f}", flush=True)
    return results


def infer_classifiers(models, datasets, config):
    results = []
    for model in models:
        start = perf_counter()
        data = datasets[str(model["seed"])]["classifier"]
        result = {"seed": model["seed"]}
        for name, parameters, reference in (
            ("initial", model["initial"], False),
            ("trained", model["parameters"], False),
            ("qfi_reference", model["initial"], True),
        ):
            raw = call(
                f"{NAMESPACE}.ClassifierInference({tuples(data['test'])}, {literal(parameters)}, "
                f"{config.test_shots}, {literal(reference)})",
                model["seed"], f"inference:classifier:{name}",
            )
            result[name] = classifier_metrics(data["test"], raw)
        result["train_margin_reevaluated"] = call(
            f"{NAMESPACE}.ClassifierObjective({tuples(data['train'])}, {config.test_shots}, "
            f"{literal(model['parameters'])})", model["seed"], "inference:classifier:train",
        )
        result["inference_seconds"] = perf_counter() - start
        results.append(result)
        print(f"  classifier, seed {model['seed']}: held-out balanced accuracy "
              f"{result['trained']['balanced_accuracy']:.3f}", flush=True)
    return results


def infer_teleportation(models, datasets, config):
    results = []
    for model in models:
        start = perf_counter()
        data = datasets[str(model["seed"])]["one_qubit"]["test"]
        result = {"name": model["name"], "seed": model["seed"]}
        for name, parameters in (("initial", model["initial_instrument"]), ("trained", model["instrument"])):
            losses, trials, failures = call(
                f"{NAMESPACE}.TeleportInference({tuples(data)}, {literal(parameters)}, "
                f"{config.teleport_test_shots})",
                model["seed"], f"inference:{model['name']}:{name}",
            )
            result[name] = {"infidelity": mean(losses), "sample_infidelities": losses,
                            "branch_trials": trials, "branch_failures": failures}
        result["inference_seconds"] = perf_counter() - start
        results.append(result)
        print(f"  {model['name']}, seed {model['seed']}: held-out infidelity "
              f"{result['trained']['infidelity']:.4f}", flush=True)
    references = []
    for seed in config.seeds:
        data = datasets[str(seed)]["one_qubit"]["test"]
        losses, trials, failures = call(
            f"{NAMESPACE}.TeleportInference({tuples(data)}, "
            f"MuTA.Applications.ExactTeleportationParameters(), {config.teleport_test_shots})",
            seed, "inference:teleport:exact",
        )
        references.append({"seed": seed, "infidelity": mean(losses),
                           "branch_trials": trials, "branch_failures": failures})
    return results, references


def infer_kernels(models, datasets, config):
    import numpy as np

    results = []
    for model in models:
        start = perf_counter()
        data = datasets[str(model["seed"])]["kernels"][model["task"]]["test"]
        predictions, scores, similarities = call(
            f"{NAMESPACE}.KernelInference({literal(model['training_points'])}, "
            f"{literal(data['points'])}, {literal(model['alpha'])}, "
            f"{literal(model['training_labels'])}, {model['bias']}, "
            f"{config.kernel_shots}, {literal(model['exact'])})",
            model["seed"], f"inference:{model['name']}",
        )
        accuracy = balanced_accuracy(data["labels"], predictions)
        results.append({
            "name": model["name"], "seed": model["seed"], "task": model["task"],
            "exact": model["exact"], "balanced_accuracy": accuracy,
            "predictions": predictions, "scores": scores, "test_train_kernel": similarities,
            "gram_min_eigenvalue": float(np.linalg.eigvalsh(model["gram"])[0]),
            "svm_stable": model["stable_passes"] >= 8,
            "inference_seconds": perf_counter() - start,
        })
        print(f"  {model['name']}, seed {model['seed']}: held-out balanced accuracy {accuracy:.3f}", flush=True)
    return results


def infer_noise(models, datasets, config):
    results = []
    for model in models:
        start = perf_counter()
        data = datasets[str(model["seed"])]["one_qubit"]["test"]
        stream = f"inference:{model['name']}"
        losses = gate_inference(model, data, config, model["parameters"], 0.0, stream + ":clean")
        curve = []
        if model["condition"] in {"clean", "resource_0.05"}:
            for p in config.resource_noise:
                noisy = losses if p == 0.0 else gate_inference(
                    model, data, config, model["parameters"], p, stream + f":resource:{p}"
                )
                curve.append({"p": p, "infidelity": mean(noisy), "sample_infidelities": noisy})
        results.append({
            "seed": model["seed"], "condition": model["condition"],
            "clean_test_infidelity": mean(losses), "sample_infidelities": losses,
            "resource_curve": curve, "inference_seconds": perf_counter() - start,
        })
        print(f"  {model['name']}, seed {model['seed']}: clean held-out infidelity {mean(losses):.4f}", flush=True)
    return results


def shot_diagnostics(classifiers, datasets, config):
    results = []
    for model in classifiers:
        seed = model["seed"]
        data = datasets[str(seed)]
        points = data["kernels"]["clusters"]["test"]["points"]
        half = len(points) // 2
        pairs = points[:min(4, half)] + points[half:half + min(4, half)]
        for shots in config.shot_sweep:
            raw = call(
                f"{NAMESPACE}.ClassifierInference({tuples(data['classifier']['test'])}, "
                f"{literal(model['parameters'])}, {shots}, false)",
                seed, f"shots:classifier:{shots}",
            )
            measured, exact = call(
                f"{NAMESPACE}.KernelEstimates({literal(pairs)}, {shots})",
                seed, f"shots:kernel:{shots}",
            )
            results.append({
                "seed": seed, "shots": shots,
                "classifier_balanced_accuracy": classifier_metrics(data["classifier"]["test"], raw)["balanced_accuracy"],
                "kernel_measured": measured, "kernel_exact": exact,
                "kernel_rmse": math.sqrt(mean((a - b) ** 2 for a, b in zip(measured, exact))),
                "kernel_rms_standard_error": math.sqrt(mean((1.0 - k * k) / shots for k in exact)),
            })
    return results


def logical_resources():
    rows = []
    for width in (1, 2, 3, 4):
        for depth in (1, 2, 3):
            vertices, edges, angles, slices, gates = qsharp.eval(
                f"{NAMESPACE}.LogicalResources({width}, {depth})"
            )
            rows.append({"width": width, "depth": depth, "vertices": vertices, "edges": edges,
                         "angles": angles, "flow_slices": slices, "compiled_gates": gates})
    return rows


def provenance(config, wall_seconds):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", "muta"], cwd=PROJECT.parent,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "schema_version": 1, "config": asdict(config), "git_head": head,
        "git_status": status, "source_sha256": source_hashes(),
        "python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in
                     ("qdk", "numpy", "matplotlib", "nbformat", "nbclient", "nbconvert", "ipykernel")},
        "seed_scheme": "First 32 SHA-256 bits of muta-reference-v1:<seed>:<stream>:quantum|classical",
        "wall_seconds": wall_seconds,
        "backend": "QDK local simulator; physical outcome-corrected MBQC; no hardware jobs",
        "uncertainty": "Mean and sample SD across independent seeds, not confidence intervals.",
    }


def validate_results(models, results, datasets, config):
    """Check completeness and estimator contracts, never require learning success."""
    validate_datasets(datasets, config)
    expected = {
        "gates": {"single", "entangling", "restricted_adam", "restricted_greedy", "restricted_dqn"},
        "classifier": {"classifier"},
        "teleportation": {"teleport_corrections", "teleport_full"},
        "noise": {f"noise:{name}" for name, _, _ in NOISE_CASES},
        "kernel": {f"kernel:{task}:{method}" for task in ("clusters", "xor") for method in ("exact", "shots")},
    }
    for group, names in expected.items():
        wanted = {(seed, name) for seed in config.seeds for name in names}
        actual = {(m["seed"], m["name"]) for m in models[group]}
        if actual != wanted or len(models[group]) != len(wanted):
            raise ValueError(f"Incomplete or duplicate {group} fits.")
        inferred = set()
        for row in results[group]:
            name = "classifier" if group == "classifier" else (
                f"noise:{row['condition']}" if group == "noise" else row["name"]
            )
            inferred.add((row["seed"], name))
        if inferred != wanted or len(results[group]) != len(wanted):
            raise ValueError(f"Incomplete or duplicate {group} inference.")
    if len(results["teleportation_reference"]) != len(config.seeds):
        raise ValueError("Missing exact teleportation controls.")
    wanted_shots = {(seed, shots) for seed in config.seeds for shots in config.shot_sweep}
    actual_shots = {(row["seed"], row["shots"]) for row in results["shots"]}
    if actual_shots != wanted_shots or len(results["shots"]) != len(wanted_shots):
        raise ValueError("Incomplete or duplicate shot-sweep results.")
    literal(results)
    for row in results["gates"]:
        if row["compiled_infidelity"] != 0.0:
            raise ValueError("A learned MBQC channel disagrees with its compiled circuit.")
        for key in ("initial_infidelity", "test_infidelity", "train_infidelity"):
            if not 0.0 <= row[key] <= 1.0:
                raise ValueError("Gate infidelity is outside [0, 1].")
    for row in results["teleportation"]:
        for stage in ("initial", "trained"):
            measurement = row[stage]
            if sum(measurement["branch_trials"]) != config.test_states * config.teleport_test_shots:
                raise ValueError("Teleportation discarded instrument outcomes.")
            if any(f < 0 or f > n for f, n in zip(
                measurement["branch_failures"], measurement["branch_trials"]
            )):
                raise ValueError("Teleportation has inconsistent branch statistics.")
    for row in results["teleportation_reference"]:
        if row["infidelity"] != 0.0 or not all(n > 0 for n in row["branch_trials"]):
            raise ValueError("The exact teleportation control failed or missed an outcome.")
    for row in results["noise"]:
        expected_curve = len(config.resource_noise) if row["condition"] in {"clean", "resource_0.05"} else 0
        if len(row["resource_curve"]) != expected_curve:
            raise ValueError("A frozen-model resource-noise sweep is incomplete.")


def summary_rows(results):
    rows = []

    def add(group, name, metric, values, initial=None):
        if len(values) < 2:
            raise ValueError("Between-seed SD requires at least two independent runs.")
        rows.append({
            "group": group, "experiment": name, "metric": metric, "runs": len(values),
            "mean": mean(values), "sd": stdev(values),
            "initial_mean": mean(initial) if initial is not None else None,
            "initial_sd": stdev(initial) if initial is not None else None,
        })

    for name in ("single", "entangling", "restricted_adam", "restricted_greedy", "restricted_dqn"):
        selected = [r for r in results["gates"] if r["name"] == name]
        add("gate", name, "held-out infidelity", [r["test_infidelity"] for r in selected],
            [r["initial_infidelity"] for r in selected])
    for stage in ("initial", "trained", "qfi_reference"):
        for family in (0, 1):
            add("classifier", f"{stage}:S{family + 1}", "balanced accuracy",
                [r[stage]["by_family"][family] for r in results["classifier"]])
    for name in ("teleport_corrections", "teleport_full"):
        selected = [r for r in results["teleportation"] if r["name"] == name]
        add("teleportation", name, "held-out infidelity", [r["trained"]["infidelity"] for r in selected],
            [r["initial"]["infidelity"] for r in selected])
    for task in ("clusters", "xor"):
        for exact in (False, True):
            selected = [r for r in results["kernel"] if r["task"] == task and r["exact"] == exact]
            add("kernel", f"{task}:{'exact' if exact else 'shots'}", "balanced accuracy",
                [r["balanced_accuracy"] for r in selected])
    for name, _, _ in NOISE_CASES:
        add("noise", name, "clean held-out infidelity",
            [r["clean_test_infidelity"] for r in results["noise"] if r["condition"] == name])
    return rows


def write_summary(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(models, results, datasets, config):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300,
    })
    blue, orange, green, purple, gray = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#777777"

    def band(ax, rows, label, color):
        support = min(len(row["history"]) for row in rows)
        values = np.array([row["history"][:support] for row in rows])
        center, spread = values.mean(axis=0), values.std(axis=0, ddof=1)
        x = np.arange(1, support + 1)
        ax.plot(x, center, color=color, label=label)
        ax.fill_between(x, np.maximum(0.0, center - spread), center + spread, color=color, alpha=0.16)

    def dots(ax, x, values, color, label=None):
        ax.errorbar(x, mean(values), yerr=stdev(values), fmt="o", color=color,
                    capsize=3, markersize=6, label=label)
        offsets = np.linspace(-0.035, 0.035, len(values))
        ax.scatter(x + offsets, values, color=color, alpha=0.45, s=14, zorder=3)

    def title(ax, letter, text):
        ax.set_title(f"{letter}  {text}", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    figure, axes = plt.subplots(2, 3, figsize=(12.6, 7.8), layout="constrained")
    a, b, c, d, e, f = axes.flat
    for name, label, color in (("single", "1-qubit, 4 angles", blue), ("entangling", "XX + local, 8 angles", orange)):
        band(a, [m for m in models["gates"] if m["name"] == name], label, color)
    title(a, "a", "Continuous training")
    a.set(xlabel="Objective evaluations (including probes)", ylabel="Best observed training infidelity", ylim=(0, None))
    a.legend()

    gate_names = ("single", "entangling", "restricted_adam", "restricted_greedy", "restricted_dqn")
    for x, name in enumerate(gate_names):
        rows = [r for r in results["gates"] if r["name"] == name]
        dots(b, x - 0.12, [r["initial_infidelity"] for r in rows], gray, "Initial" if x == 0 else None)
        dots(b, x + 0.12, [r["test_infidelity"] for r in rows], blue, "Fitted" if x == 0 else None)
    title(b, "b", "Gate inference on new Haar states")
    b.set_xticks(range(5), ("1q", "XX+local", "Adam\nrestricted\n target", "Greedy", "DQN"))
    b.set(ylabel="Held-out infidelity", ylim=(-0.02, None))
    b.legend()

    for family in (0, 1):
        for offset, stage, label, color in (
            (-0.23, "initial", "Initial", gray),
            (0.0, "trained", "Jointly fitted", blue),
            (0.23, "qfi_reference", "Known-QFI head", green),
        ):
            dots(c, family + offset, [r[stage]["by_family"][family] for r in results["classifier"]],
                 color, label if family == 0 else None)
    c.axhline(0.5, color=gray, linestyle="--", linewidth=1)
    title(c, "c", "Quantum-state classification")
    c.set_xticks((0, 1), ("S1", "S2"))
    c.set(ylabel="Held-out balanced accuracy", ylim=(0.0, 1.07))
    c.legend(loc="lower left")

    for x, name in enumerate(("teleport_corrections", "teleport_full")):
        rows = [r for r in results["teleportation"] if r["name"] == name]
        for offset, stage, color in ((-0.12, "initial", gray), (0.12, "trained", blue)):
            dots(d, x + offset, [r[stage]["infidelity"] for r in rows],
                 color, stage.capitalize() if x == 0 else None)
    d.axhline(0.0, color=green, linestyle=":", label="Exact instrument")
    title(d, "d", "Teleportation, all outcomes retained")
    d.set_xticks((0, 1), ("4 corrections\nBell stages fixed", "20 angles\nrandom initialization"))
    d.set(ylabel="Held-out infidelity", ylim=(-0.025, None))
    d.legend()

    for x, task in enumerate(("clusters", "xor")):
        for offset, exact, label, color in (
            (-0.12, False, "Finite-shot MuTA", blue), (0.12, True, "Exact same kernel", green)
        ):
            values = [r["balanced_accuracy"] for r in results["kernel"] if r["task"] == task and r["exact"] == exact]
            dots(e, x + offset, values, color, label if x == 0 else None)
    e.axhline(0.5, color=gray, linestyle="--", linewidth=1)
    title(e, "e", "Kernel SVM on unseen feature vectors")
    e.set_xticks((0, 1), ("Two clusters", "XOR clusters"))
    e.set(ylabel="Held-out balanced accuracy", ylim=(0.0, 1.07))
    e.legend(loc="lower left")

    for name, label, color in (
        ("clean", "Clean-trained", blue), ("resource_0.05", "Trained at p=0.05", orange)
    ):
        curves = [r["resource_curve"] for r in results["noise"] if r["condition"] == name]
        values = np.array([[p["infidelity"] for p in curve] for curve in curves])
        center, spread = values.mean(axis=0), values.std(axis=0, ddof=1)
        f.errorbar(config.resource_noise, center, yerr=spread, marker="o", capsize=3, color=color, label=label)
    title(f, "f", "Resource-noise inference, frozen angles")
    f.set(xlabel="Nonidentity Pauli probability p per vertex", ylabel="Held-out infidelity", ylim=(0.0, None))
    f.legend()
    figure.suptitle(f"Hardware-agnostic MuTA | local MBQC simulation | {len(config.seeds)} seeds, mean +/- SD", fontsize=12)

    diagnostic, axes = plt.subplots(2, 3, figsize=(12.6, 8.0), layout="constrained")
    a, b, c, d, e, f = axes.flat
    for name, label, color in (
        ("restricted_adam", "Adam (continuous angles)", gray),
        ("restricted_greedy", "Slicewise greedy", blue),
        ("restricted_dqn", "Neural DQN", orange),
    ):
        band(a, [m for m in models["gates"] if m["name"] == name], label, color)
    title(a, "a", "Four-slot search, same target")
    a.set(xlabel="Objective evaluations", ylabel="Best observed training infidelity", ylim=(0, None))
    a.legend()

    for x, (name, mode, _) in enumerate(NOISE_CASES):
        values = [r["clean_test_infidelity"] for r in results["noise"] if r["condition"] == name]
        dots(b, x, values, (gray, blue, purple, orange)[mode])
    title(b, "b", "Training corruption; clean inference")
    b.set_xticks(range(7), ("Clean", "Flip\n0.1", "Flip\n0.3", "Brownian\n0.1", "Brownian\n0.3", "Resource\n0.01", "Resource\n0.05"))
    b.set(ylabel="Clean held-out infidelity", ylim=(0, None))
    b.tick_params(axis="x", labelsize=8)

    for family, color in ((0, blue), (1, orange)):
        truth, scores = [], []
        for row in results["classifier"]:
            data = datasets[str(row["seed"])]["classifier"]["test"]
            for sample, score in zip(data, row["trained"]["scores"]):
                if sample[0] == family:
                    truth.append(sample[3])
                    scores.append(score)
        c.scatter(truth, scores, s=13, alpha=0.5, color=color, label=f"S{family + 1}")
    c.axvline(2.0, color=gray, linestyle="--")
    c.axhline(2.0, color=gray, linestyle="--")
    title(c, "c", "Classifier boundary, not QFI regression")
    c.set(xlabel="Analytic ground-truth QFI", ylabel="Learned polynomial score")
    c.legend()

    shots = sorted(config.shot_sweep)
    for key, color, ax, ylabel in (
        ("classifier_balanced_accuracy", blue, d, "Held-out balanced accuracy"),
        ("kernel_rmse", orange, e, "SWAP kernel RMSE"),
    ):
        grouped = [[r[key] for r in results["shots"] if r["shots"] == n] for n in shots]
        ax.errorbar(shots, [mean(v) for v in grouped], yerr=[stdev(v) for v in grouped],
                    color=color, marker="o", capsize=3)
        ax.set(xscale="log", xlabel="Shots per state / kernel pair", ylabel=ylabel)
        ax.set_xticks(shots, [str(n) for n in shots])
    d.set_ylim(0.0, 1.05)
    title(d, "d", "Shot sensitivity without retraining")
    theory = [mean(r["kernel_rms_standard_error"] for r in results["shots"] if r["shots"] == n) for n in shots]
    e.plot(shots, theory, "--", color=gray, label="SWAP estimator RMS SE")
    e.set_ylim(bottom=0.0)
    title(e, "e", "Measured kernel vs exact expression")
    e.legend()

    for width, color in zip((1, 2, 3, 4), (blue, orange, green, purple)):
        rows = [r for r in results["resources"] if r["width"] == width]
        f.plot([r["depth"] for r in rows], [r["vertices"] for r in rows],
               marker="o", color=color, label=f"{width} wires")
    title(f, "f", "Logical graphs, not physical qubit costs")
    f.set(xlabel="Concatenated layers", ylabel="Graph vertices")
    f.set_xticks((1, 2, 3))
    f.legend()
    diagnostic.suptitle("Baseline diagnostics | all seeds retained; no test-set model selection", fontsize=12)
    return {"figure-1": figure, "diagnostics": diagnostic}


def save_figures(directory, figures):
    directory = Path(directory)
    paths = []
    for name, figure in figures.items():
        for suffix in ("pdf", "svg", "png"):
            path = directory / f"{name}.{suffix}"
            figure.savefig(path, bbox_inches="tight")
            paths.append(path)
    return paths


def evaluation_markdown(models, results, config):
    rows = summary_rows(results)
    text = [
        "## Evaluation of this executed baseline",
        "",
        f"All **{sum(len(v) for v in models.values())} fits** across **{len(config.seeds)} seeds** "
        "are retained. Inference uses reloaded models, independent held-out data, and fresh "
        "quantum/classical RNG streams. Values below are means +/- sample SD across seeds.",
        "",
        "| Experiment | Held-out metric | Initial | Fitted / reference |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        initial = "-" if row["initial_mean"] is None else f"{row['initial_mean']:.3f} +/- {row['initial_sd']:.3f}"
        text.append(f"| {row['experiment']} | {row['metric']} | {initial} | "
                    f"{row['mean']:.3f} +/- {row['sd']:.3f} |")
    text += ["", "### What the results support", ""]
    for name, label in (("single", "Single-qubit Adam"), ("entangling", "Entangling-gate Adam")):
        selected = [r for r in results["gates"] if r["name"] == name]
        improved = sum(r["test_infidelity"] < r["initial_infidelity"] for r in selected)
        text.append(f"- {label} improves held-out infidelity in **{improved}/{len(selected)}** runs. "
                    f"Mean held-out infidelity is **{mean(r['test_infidelity'] for r in selected):.4f}**.")
    for name, label in (("teleport_corrections", "Four-correction teleportation"), ("teleport_full", "Cold-start, 20-angle teleportation")):
        selected = [r for r in results["teleportation"] if r["name"] == name]
        improved = sum(r["trained"]["infidelity"] < r["initial"]["infidelity"] for r in selected)
        text.append(f"- {label} improves in **{improved}/{len(selected)}** runs; mean held-out "
                    f"infidelity is **{mean(r['trained']['infidelity'] for r in selected):.4f}**. "
                    "The exact instrument has no observed projection failures.")
    learned = mean(r["trained"]["balanced_accuracy"] for r in results["classifier"])
    reference = mean(r["qfi_reference"]["balanced_accuracy"] for r in results["classifier"])
    text.append(f"- The jointly trained state classifier reaches **{learned:.3f}** pooled balanced "
                f"accuracy versus **{reference:.3f}** for the task-informed, known-QFI measurement "
                "baseline. The fitted score is not a calibrated QFI estimate or probability.")
    indefinite = sum(r["gram_min_eigenvalue"] < -1e-8 for r in results["kernel"] if not r["exact"])
    unstable = sum(not r["svm_stable"] for r in results["kernel"])
    minimum = min(r["gram_min_eigenvalue"] for r in results["kernel"] if not r["exact"])
    text.append(f"- **{indefinite}/{2 * len(config.seeds)}** shot-estimated training Gram matrices "
                f"have a negative eigenvalue (minimum **{minimum:.4g}**); **{unstable}** SVM fits "
                "reach the iteration cap without eight stable passes. Matrices are not clipped "
                "or silently projected to PSD; even a stable pass count is not an optimality proof.")
    comparisons = []
    for model in models["gates"]:
        row = next(r for r in results["gates"] if r["seed"] == model["seed"] and r["name"] == model["name"])
        comparisons.append(row["train_infidelity"] - model["training_best_loss"])
    text.append(f"- Fresh, higher-shot training-set reevaluation exceeds the selected training "
                f"incumbent by **{mean(comparisons):.4f}** infidelity on average. Best-observed "
                "training curves are selection-biased and must not be reported as test fidelity.")
    clean = {r["seed"]: r["clean_test_infidelity"] for r in results["noise"] if r["condition"] == "clean"}
    resource = {r["seed"]: r["clean_test_infidelity"] for r in results["noise"] if r["condition"] == "resource_0.05"}
    delta = mean(resource[s] - clean[s] for s in config.seeds)
    text.append(f"- Training with resource p=0.05 changes clean held-out infidelity by "
                f"**{delta:+.4f}** relative to the matched-budget clean control. "
                "The noise curve evaluates these same frozen models; it does not retrain at each p.")
    text += [
        "",
        "### Limits and interpretation for Figure 1",
        "",
        "- This is a small, budgeted simulator reference, not a reproduction of the paper's "
        "numerical experiments, a convergence guarantee, or evidence of quantum learning advantage. "
        "Five seeds characterize this run, not a publication-grade confidence interval.",
        "- The XX target is attainable in one connected layer; all eight angles are trainable. "
        "Greedy/DQN use all four slots and alphabet {0, pi/4, pi/2}; their Adam control "
        "uses continuous angles on the same target. No claim of equal wall-clock work is made.",
        "- The four-correction teleportation experiment assumes exact Bell stages. Only the "
        "20-angle condition is a random cold start; neither the known solution nor the test "
        "loss is used to select its model. No instrument outcomes are postselected.",
        "- Classifier states are class-balanced within each family, including states near SQL=2. "
        "The kernel datasets are synthetic balanced clusters/XOR, not the paper's datasets; "
        "the MuTA feature map is fixed and only the SVM is fitted.",
        "- Bit-flip programs and Brownian programs are sampled once per training label and saved. "
        "Brownian levels are time steps with three increments and two Trotter steps, not error "
        "probabilities. Resource p is total nonidentity Pauli probability per graph vertex "
        "(p/3 each for X, Y, Z), sampled anew on every execution.",
        "- Zero sampled errors do not establish a zero true error rate. Error bars describe "
        "between-seed variation, combining training randomness, dataset draws, and finite shots; "
        "they are not independent per-shot confidence intervals.",
        "- Graph counts, MBQC execution counts, and simulator wall times are not physical "
        "qubit costs or QPU latency. No topological encoding, Majorana operations, logical "
        "error correction, device connectivity, or hardware execution is included.",
        "",
        "### Recorded training work",
        "",
        "| Family | Fits | MBQC pattern executions | Simulator fit time (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group, fits in models.items():
        text.append(f"| {group} | {len(fits)} | {sum(m['mbqc_executions'] for m in fits):,} | "
                    f"{sum(m['fit_seconds'] for m in fits):.1f} |")
    return "\n".join(text)


def figure_caption(config):
    return (
        f"Figure 1. Hardware-agnostic MuTA baseline under local QDK MBQC simulation "
        f"({len(config.seeds)} independent seeds; means and sample standard deviations). "
        "(a) Best observed continuous-training infidelity versus objective evaluations, including "
        "derivative probes. (b) Initial and fitted gate infidelity on independent Haar states; "
        "the last three conditions share an attainable four-slot restricted-angle target, "
        "but the Adam control has continuous angles. (c) Held-out balanced accuracy for both "
        "quantum-state families, compared with initialization and a task-informed known-QFI head. "
        "(d) Teleportation with four fitted correction angles and fixed Bell stages, or all "
        "twenty angles fitted from a random initialization; all measurement outcomes are retained. "
        "(e) SVM prediction on unseen cluster/XOR points using the finite-shot MuTA kernel or "
        "its exact expression. (f) Frozen clean-trained and noise-trained gates under resource "
        "depolarization, with total nonidentity Pauli probability p per vertex. Individual dots "
        "in (b)-(e) show all seeds. Budgets and data sizes differ by task and are specified in the "
        "notebook and machine-readable configuration. Training incumbents are noisy and "
        "selection-biased; panels (b)-(f) use fresh inference shots without test-set model "
        "selection. These small demonstrations do not establish quantum advantage, reproduce "
        "published numerical figures, or model topological-qubit hardware."
    )
