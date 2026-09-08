"""QDK harness: mathematical/algorithm checks are implemented in Q#.

Python manages compiler contexts, reproducible seeds, and expected failures.
"""

from pathlib import Path
import importlib.util

import pytest
from qdk import qsharp

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def qs():
    qsharp.init(project_root=str(PROJECT))
    qsharp.set_quantum_seed(2026)
    qsharp.set_classical_seed(2026)
    return qsharp


@pytest.mark.parametrize("name", [
    "TestLayers", "TestConcatenation", "TestBoundaryGraphs",
    "TestCliffordControl", "TestExpressivity",
])
def test_core_channels(qs, name):
    qs.eval(f"MuTA.CoreTests.{name}()")


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_measurement_branches(qs, seed):
    qs.set_quantum_seed(seed)
    qs.eval("MuTA.CoreTests.LayerEquivalence(2,0,[1],0.38)")


def test_angle_and_slice_conventions(qs):
    order = qs.eval("MuTA.Core.Layer(2,0,[1]).Order")
    slices = qs.eval("MuTA.Core.ParameterSlices(MuTA.Core.Layer(2,0,[1]))")
    assert len(order) == 8
    assert set(order) == {0, 1, 2, 3, 5, 6, 7, 8}
    assert sorted(i for group in slices for i in group) == list(range(8))
    angles = qs.eval(
        "MuTA.Core.AnglesFromVertices(MuTA.Core.Layer(2,0,[1]),"
        "[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])"
    )
    assert angles == [float(v) for v in order]


@pytest.mark.parametrize("expression,message", [
    ("MuTA.Core.Layer(0,-1,[])", "width must be positive"),
    ("MuTA.Core.Layer(2,0,[0])", "Invalid triangle targets"),
    ("MuTA.Core.Layer(2,0,[1,1])", "Duplicate vertex"),
    ("MuTA.Core.CreatePattern(2,[(0,1),(1,0)],[0],[1],[1,-1])", "Duplicate undirected edge"),
    ("MuTA.Core.CreatePattern(2,[(0,0)],[0],[1],[1,-1])", "Self edges"),
    ("MuTA.Core.CreatePattern(3,[(0,1),(1,2),(2,0)],[0],[2],[1,2,-1])", "cyclic causal"),
    ("MuTA.Core.Compile(MuTA.Core.Layer(1,-1,[]),[0.0])", "One angle"),
    ("MuTA.Core.CreatePattern(2,[(0,1)],[1],[1],[1,-1])", "successor cannot be an input"),
    ("MuTA.Core.Concatenate(MuTA.Core.Layer(1,-1,[]),MuTA.Core.Layer(1,-1,[]),[(3,0)])", "Joins must match"),
    ("MuTA.Core.Compile(MuTA.Core.CreatePattern(2,[(0,1)],[0],[0,1],[-1,-1]),[])", "equal input/output"),
    ("MuTA.Expressivity.PauliLieClosure([[PauliX],[PauliZ]],2)", "exceeded maxTerms"),
])
def test_invalid_graphs_and_budgets(qs, expression, message):
    with pytest.raises(Exception, match=message):
        qs.eval(expression)


@pytest.mark.parametrize("name", [
    "TestMeasurementOrder", "TestGreedyCanonicalOrder", "TestGreedyJointWindow",
    "TestGreedyExplorationAndBudget", "TestGradients", "TestAdam",
    "TestDqnStateEncoding", "TestDqnEpisodeOrderAndRewards", "TestDqnForwardAndBellman",
    "TestDqnHandCalculatedSgd", "TestDqnBackpropagationNumerically",
    "TestDqnReplayAndSchedules", "TestDqnDriver", "TestEmptyOptimizers",
])
def test_training_algorithms(qs, name):
    qs.eval(f"MuTA.TrainingTests.{name}()")


@pytest.mark.parametrize("name", [
    "RejectRepeatedSliceIndex", "RejectMissingSliceIndex", "RejectOutputVertexAsParameter",
    "RejectEmptySlice", "RejectRepeatedAngles", "RejectInitialAngle", "RejectGreedyBudget",
    "RejectDifferenceStep", "RejectDqnNetworkShape", "RejectReplayAction",
])
def test_invalid_training_inputs(qs, name):
    with pytest.raises(Exception, match="Qdk.Qsc.Eval.UserFail"):
        qs.eval(f"MuTA.TrainingTests.{name}()")


@pytest.mark.parametrize("name", [
    "TestStatePreparationAndGateLoss", "TestNoiseModels", "TestClassifier",
    "TestTeleportation", "TestKernelsAndSvm",
])
def test_applications(qs, name):
    qs.eval(f"MuTA.ApplicationTests.{name}()")


def test_quantum_objective_greedy(qs):
    theta, loss, evaluations = qs.eval("MuTA.Examples.GreedyLearning()")
    assert theta[0] == pytest.approx(0.7853981633974483)
    assert loss == 0.0
    assert evaluations <= 20


def test_quantum_entry_compiles_to_adaptive_qir():
    path = PROJECT / "export_qir.py"
    spec = importlib.util.spec_from_file_location("muta_qir_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qir = module.compile_quantum_entry()
    assert "__quantum__qis__m" in qir
    assert "entry_point" in qir
    assert "br i1" in qir  # outcome-dependent correction survives compilation
