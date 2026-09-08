namespace MuTA.TrainingTests {
    import Std.Math.*;
    import MuTA.Training.*;

    function Check(condition : Bool, message : String) : Unit {
        if not condition { fail message; }
    }

    function Near(actual : Double, expected : Double, tolerance : Double, label : String) : Unit {
        Check(IsFiniteTrainingValue(actual) and AbsD(actual - expected) <= tolerance,
            $"{label}: expected {expected}, got {actual}.");
    }

    function CheckHistory(result : TrainingResult) : Unit {
        Check(Length(result.History) == result.Evaluations, "History must record every objective evaluation.");
        for k in 1..Length(result.History) - 1 {
            Check(result.History[k] <= result.History[k - 1], "Incumbent history must be nonincreasing.");
        }
        Near(result.History[Length(result.History) - 1], result.Loss, 0.0, "Final history");
    }

    operation SeparableLoss(theta : Double[]) : Double {
        return 4.0 * (theta[0] - 1.0) * (theta[0] - 1.0) +
            2.0 * theta[1] * theta[1] + (theta[2] - 1.0) * (theta[2] - 1.0);
    }

    operation CoupledLoss(theta : Double[]) : Double {
        if theta[0] == 1.0 and theta[1] == 1.0 { return 0.0; }
        if theta[0] == 0.0 and theta[1] == 0.0 { return 1.0; }
        return 2.0;
    }

    operation IncreasingLoss(theta : Double[]) : Double { return theta[0]; }
    operation DecreasingLoss(theta : Double[]) : Double { return 1.0 - theta[0]; }
    operation EmptyLoss(theta : Double[]) : Double { Check(Length(theta) == 0, "Expected no parameters."); return 0.0; }
    operation PeriodicLoss(theta : Double[]) : Double {
        return (1.0 - Cos(theta[0])) / 2.0 + (1.0 - Sin(theta[1])) / 4.0;
    }
    operation PolynomialLoss(theta : Double[]) : Double {
        return theta[0] * theta[0] * theta[0] + theta[0] * theta[1] + 2.0 * theta[1] * theta[1];
    }
    operation QuadraticLoss(theta : Double[]) : Double {
        return (theta[0] - 0.3) * (theta[0] - 0.3) + (theta[1] + 0.4) * (theta[1] + 0.4);
    }

    function DefaultAdam(shift : Bool) : AdamConfig {
        return new AdamConfig {
            Steps = 160, LearningRate = 0.08, Beta1 = 0.9, Beta2 = 0.999,
            Epsilon = 1e-8, Threshold = 1e-7, UseParameterShift = shift,
            DifferenceStep = 1e-5, MaxEvaluations = 1000
        };
    }

    function DefaultDqn() : DqnConfig {
        return new DqnConfig {
            Episodes = 64, HiddenSize = 8, LearningRate = 0.02, Gamma = 0.9,
            EpsilonStart = 1.0, EpsilonMin = 0.1, EpsilonDecay = 0.95,
            ReplayCapacity = 32, BatchSize = 4, UpdatesPerEpisode = 2,
            TargetPeriod = 3, Threshold = -1.0, MaxEvaluations = 100
        };
    }

    operation TestMeasurementOrder() : Unit {
        let order = MeasurementOrder(5, [[3], [0, 4], [2], [1]]);
        Check(order == [3, 0, 4, 2, 1], "Temporal order was sorted or changed.");
        Check(MeasurementOrder(0, []) == [], "Empty parameter set must have empty order.");
    }

    operation TestGreedyCanonicalOrder() : Unit {
        let result = GreedyOptimize(SeparableLoss, [0.0, 1.0, 0.0], [[2], [0, 1]],
            [0.0, 1.0], 0.0, 0, 1, 0.01, 100);
        Check(result.Parameters == [1.0, 0.0, 1.0], "Greedy changed canonical parameter indexing.");
        Check(result.Success, "Zero-restart greedy search must still run.");
        Near(result.Loss, 0.0, 0.0, "Greedy optimum");
        CheckHistory(result);
    }

    operation TestGreedyJointWindow() : Unit {
        let local = GreedyOptimize(CoupledLoss, [0.0, 0.0], [[1], [0]],
            [0.0, 1.0], 0.0, 0, 1, 0.1, 100);
        Near(local.Loss, 1.0, 0.0, "Single-slice local minimum");
        Check(not local.Success, "Single slices cannot escape this strict local minimum.");
        let joint = GreedyOptimize(CoupledLoss, [0.0, 0.0], [[1], [0]],
            [0.0, 1.0], 0.0, 0, 2, 0.1, 100);
        Check(joint.Parameters == [1.0, 1.0], "A two-slice window must enumerate the joint optimum.");
        Check(joint.Success and joint.Evaluations == 9, "Joint enumeration count is incorrect.");
        CheckHistory(joint);
    }

    operation TestGreedyExplorationAndBudget() : Unit {
        // epsilon = 1 accepts the final (worse) enumerated candidate, but the
        // returned incumbent must retain the initial optimum.
        let explore = GreedyOptimize(IncreasingLoss, [0.0], [[0]], [0.0, 1.0],
            1.0, 0, 1, -1.0, 100);
        Check(explore.Parameters == [0.0] and explore.Evaluations == 3,
            "Exploration discarded the best pattern or evaluated a wrong number of candidates.");
        CheckHistory(explore);
        let budget = GreedyOptimize(CoupledLoss, [0.0, 0.0], [[1], [0]],
            [0.0, 1.0], 0.0, 100, 2, 0.1, 5);
        Check(budget.Evaluations == 5 and not budget.Success, "Greedy evaluation budget was exceeded.");
        CheckHistory(budget);
        let initialOnly = GreedyOptimize(IncreasingLoss, [1.0], [[0]], [0.0, 1.0],
            0.0, 3, 1, -1.0, 1);
        Check(initialOnly.Evaluations == 1, "Budget one must evaluate only the initial pattern.");
    }

    operation TestGradients() : Unit {
        let theta = [0.7, -0.3];
        let shift = ParameterShiftGradient(PeriodicLoss, theta);
        Near(shift[0], Sin(theta[0]) / 2.0, 1e-12, "Parameter-shift X derivative");
        Near(shift[1], -Cos(theta[1]) / 4.0, 1e-12, "Parameter-shift Y derivative");
        let finite = CentralDifferenceGradient(PolynomialLoss, theta, 1e-5);
        Near(finite[0], 3.0 * theta[0] * theta[0] + theta[1], 1e-8, "Finite-difference first derivative");
        Near(finite[1], theta[0] + 4.0 * theta[1], 1e-8, "Finite-difference second derivative");
    }

    operation TestAdam() : Unit {
        let continuous = TrainAdam(PeriodicLoss, [1.4, -0.2], DefaultAdam(true));
        Check(continuous.Loss < 1e-5, "Parameter-shift Adam did not optimize a valid periodic objective.");
        CheckHistory(continuous);
        let nonlinear = TrainAdam(QuadraticLoss, [1.1, 0.8], DefaultAdam(false));
        Check(nonlinear.Loss < 1e-5, "Finite-difference Adam did not optimize a nonlinear objective.");
        CheckHistory(nonlinear);
        let budgetConfig = new AdamConfig {
            Steps = 20, LearningRate = 0.1, Beta1 = 0.9, Beta2 = 0.999,
            Epsilon = 1e-8, Threshold = -1.0, UseParameterShift = false,
            DifferenceStep = 1e-5, MaxEvaluations = 5
        };
        let budget = TrainAdam(QuadraticLoss, [1.0, 1.0], budgetConfig);
        Check(budget.Evaluations == 1, "Adam must reserve a complete gradient plus update evaluation.");
    }

    operation TestDqnStateEncoding() : Unit {
        let initial = EncodeDqnState([0.0, 0.0], [false, false], 1);
        Check(initial == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "Wrong initial DQN features.");
        let partial = EncodeDqnState([0.0, 0.0], [false, true], 0);
        Check(partial == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            "A chosen zero angle must differ from an unassigned parameter.");
        let terminal = EncodeDqnState([PI(), 0.0], [true, true], -1);
        Check(terminal == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], "Wrong terminal DQN features.");
    }

    /// This fixed policy selects angle one for CANONICAL slot zero and angle
    /// zero for slot one. Executing order [1,0] catches append-order corruption.
    function PermutationPolicy() : QNetwork {
        return new QNetwork {
            InputSize = 6, HiddenSize = 2, OutputSize = 2,
            InputWeights = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            HiddenBias = [0.0, 0.0], OutputWeights = [0.0, 1.0, 1.0, 0.0],
            OutputBias = [0.0, 0.0]
        };
    }

    operation PermutationLoss(theta : Double[]) : Double {
        return 2.0 + 3.0 * (theta[0] - 1.0) * (theta[0] - 1.0) + theta[1] * theta[1];
    }

    operation TestDqnEpisodeOrderAndRewards() : Unit {
        let (theta, value, replay) = SampleDqnEpisode(PermutationLoss, 2, [[1], [0]],
            [0.0, 1.0], PermutationPolicy(), 0.0, [], 4);
        Check(theta == [1.0, 0.0], "DQN actions were appended instead of assigned to canonical slots.");
        Near(value, 2.0, 0.0, "Canonical-order episode loss");
        Check(Length(replay) == 2, "Episode must emit one transition per measured parameter.");
        Near(replay[0].Reward, 0.0, 0.0, "Intermediate reward");
        Near(replay[1].Reward, -2.0, 0.0, "Terminal negative-loss reward");
        Check(not replay[0].Terminal and replay[1].Terminal, "Episode termination flags are incorrect.");
        Check(replay[0].Action == 0 and replay[1].Action == 1, "Wrong policy actions.");
        Check(replay[0].NextState == replay[1].State, "Adjacent replay states do not match.");
        let (_, _, bounded) = SampleDqnEpisode(PermutationLoss, 2, [[1], [0]],
            [0.0, 1.0], PermutationPolicy(), 0.0, [], 1);
        Check(Length(bounded) == 1 and bounded[0].Terminal, "FIFO capacity one did not retain the terminal transition.");
    }

    function SmallNetwork() : QNetwork {
        return new QNetwork {
            InputSize = 1, HiddenSize = 1, OutputSize = 2, InputWeights = [1.0],
            HiddenBias = [0.5], OutputWeights = [2.0, -1.0], OutputBias = [0.1, 0.2]
        };
    }

    operation TestDqnForwardAndBellman() : Unit {
        let network = SmallNetwork();
        let (hidden, output) = QNetworkForward(network, [3.0]);
        Near(hidden[0], 3.5, 1e-12, "Hidden activation");
        Near(output[0], 7.1, 1e-12, "First Q value");
        Near(output[1], -3.3, 1e-12, "Second Q value");
        let (inactive, inactiveOutput) = QNetworkForward(network, [-2.0]);
        Near(inactive[0], 0.0, 0.0, "ReLU inactive unit");
        Check(inactiveOutput == network.OutputBias, "Inactive ReLU should leave only output bias.");
        let terminal = new ReplayTransition { State = [3.0], Action = 0, Reward = -2.0, NextState = [], Terminal = true };
        Near(BellmanTarget(terminal, network, 0.9), -2.0, 0.0, "Terminal target must not bootstrap");
        let nonterminal = new ReplayTransition { State = [1.0], Action = 1, Reward = 0.25, NextState = [3.0], Terminal = false };
        Near(BellmanTarget(nonterminal, network, 0.5), 3.8, 1e-12, "Nonterminal Bellman target");
    }

    operation TestDqnHandCalculatedSgd() : Unit {
        let network = SmallNetwork();
        let sample = new ReplayTransition { State = [3.0], Action = 0, Reward = 1.0, NextState = [], Terminal = true };
        let (updated, mse) = DqnMinibatchStep(network, network, [sample], 0.9, 0.01);
        Near(mse, 37.21, 1e-10, "Minibatch MSE");
        Near(updated.InputWeights[0], 0.268, 1e-12, "Input-weight SGD");
        Near(updated.HiddenBias[0], 0.256, 1e-12, "Hidden-bias SGD");
        Near(updated.OutputWeights[0], 1.573, 1e-12, "Output-weight SGD");
        Near(updated.OutputBias[0], -0.022, 1e-12, "Output-bias SGD");
        Near(updated.OutputWeights[1], -1.0, 0.0, "Unselected output weight");
        Near(updated.OutputBias[1], 0.2, 0.0, "Unselected output bias");
        let (duplicate, duplicateMse) = DqnMinibatchStep(network, network, [sample, sample], 0.9, 0.01);
        Check(NetworkParameters(updated) == NetworkParameters(duplicate), "Duplicating a minibatch must preserve its mean-gradient update.");
        Near(duplicateMse, mse, 1e-12, "Minibatch normalization");
        // A frozen target must not be silently mutated by the online update.
        Near(network.InputWeights[0], 1.0, 0.0, "Target snapshot immutability");
    }

    function GradientNetwork(parameters : Double[]) : QNetwork {
        return new QNetwork {
            InputSize = 2, HiddenSize = 2, OutputSize = 2,
            InputWeights = parameters[0..3], HiddenBias = parameters[4..5],
            OutputWeights = parameters[6..9], OutputBias = parameters[10..11]
        };
    }

    function NetworkParameters(network : QNetwork) : Double[] {
        return network.InputWeights + network.HiddenBias + network.OutputWeights + network.OutputBias;
    }

    function NetworkMse(network : QNetwork, target : QNetwork, batch : ReplayTransition[]) : Double {
        mutable value = 0.0;
        for sample in batch {
            let (_, output) = QNetworkForward(network, sample.State);
            let error = output[sample.Action] - BellmanTarget(sample, target, 0.8);
            set value += error * error / 2.0;
        }
        return value;
    }

    operation TestDqnBackpropagationNumerically() : Unit {
        let parameters = [0.4, -0.2, -0.3, 0.5, 0.7, -0.8, 0.2, -0.4, 0.8, 0.1, -0.1, 0.3];
        let network = GradientNetwork(parameters);
        let batch = [
            new ReplayTransition { State = [0.6, -0.1], Action = 0, Reward = 0.25, NextState = [], Terminal = true },
            new ReplayTransition { State = [-0.3, 2.0], Action = 1, Reward = 0.1, NextState = [0.2, 0.4], Terminal = false }
        ];
        let (updated, _) = DqnMinibatchStep(network, network, batch, 0.8, 0.01);
        let after = NetworkParameters(updated);
        for k in 0..Length(parameters) - 1 {
            let plus = GradientNetwork(parameters w/ k <- parameters[k] + 1e-6);
            let minus = GradientNetwork(parameters w/ k <- parameters[k] - 1e-6);
            let numerical = (NetworkMse(plus, network, batch) - NetworkMse(minus, network, batch)) / 2e-6;
            let analytic = (parameters[k] - after[k]) / 0.01;
            Near(analytic, numerical, 1e-7, $"Backpropagation parameter {k}");
        }
    }

    operation TestDqnReplayAndSchedules() : Unit {
        let first = new ReplayTransition { State = [0.0], Action = 0, Reward = 1.0, NextState = [], Terminal = true };
        let second = new ReplayTransition { State = [0.0], Action = 0, Reward = 2.0, NextState = [], Terminal = true };
        let third = new ReplayTransition { State = [0.0], Action = 0, Reward = 3.0, NextState = [], Terminal = true };
        let buffer = PushReplay(PushReplay(PushReplay([], first, 2), second, 2), third, 2);
        Check(Length(buffer) == 2 and buffer[0].Reward == 2.0 and buffer[1].Reward == 3.0,
            "Replay must evict the oldest transition.");
        let config = DefaultDqn();
        Near(DqnEpsilon(config, 0), 1.0, 0.0, "Initial epsilon");
        Near(DqnEpsilon(config, 1), 0.95, 1e-12, "Epsilon decay");
        Near(DqnEpsilon(config, 1000), 0.1, 1e-12, "Epsilon floor");
        let target = SmallNetwork();
        let (online, _) = DqnMinibatchStep(target, target, [first], 0.9, 0.01);
        Check(NetworkParameters(UpdatedTarget(online, target, 1, 2)) == NetworkParameters(target), "Target copied before its scheduled period.");
        Check(NetworkParameters(UpdatedTarget(online, target, 2, 2)) == NetworkParameters(online), "Target was not copied at its scheduled period.");
    }

    /// The runner seeds QDK's classical RNG. Gradient correctness and actual
    /// policy execution are covered above independently of random convergence.
    operation TestDqnDriver() : Unit {
        let result = TrainDqn(DecreasingLoss, [0.0], [[0]], [0.0, 1.0], DefaultDqn());
        Check(result.Parameters == [1.0] and result.Loss == 0.0, "DQN driver never sampled the optimum.");
        Check(result.Evaluations == 65 and not result.Success, "DQN episode or strict threshold semantics are incorrect.");
        CheckHistory(result);
        let bounded = new DqnConfig {
            Episodes = 20, HiddenSize = 2, LearningRate = 0.01, Gamma = 0.9,
            EpsilonStart = 1.0, EpsilonMin = 0.0, EpsilonDecay = 0.0,
            ReplayCapacity = 2, BatchSize = 1, UpdatesPerEpisode = 1,
            TargetPeriod = 1, Threshold = -1.0, MaxEvaluations = 2
        };
        let limited = TrainDqn(IncreasingLoss, [0.0], [[0]], [0.0, 1.0], bounded);
        Check(limited.Evaluations == 2, "DQN exceeded its objective budget.");
        Near(DqnEpsilon(bounded, 0), 1.0, 0.0, "Zero-decay initial epsilon");
        Near(DqnEpsilon(bounded, 1), 0.0, 0.0, "Zero-decay next epsilon");
    }

    operation TestEmptyOptimizers() : Unit {
        let greedy = GreedyOptimize(EmptyLoss, [], [], [0.0], 0.0, 0, 1, 0.1, 1);
        let adam = TrainAdam(EmptyLoss, [], DefaultAdam(true));
        let dqn = TrainDqn(EmptyLoss, [], [], [0.0], DefaultDqn());
        Check(greedy.Evaluations == 1 and adam.Evaluations == 1 and dqn.Evaluations == 1,
            "Empty models should evaluate once without attempting parameter updates.");
        CheckHistory(greedy);
        CheckHistory(adam);
        CheckHistory(dqn);
    }

    // Negative tests are invoked separately by the Python runner and MUST fail.
    operation RejectRepeatedSliceIndex() : Unit { let _ = MeasurementOrder(2, [[0], [0]]); }
    operation RejectMissingSliceIndex() : Unit { let _ = MeasurementOrder(2, [[0]]); }
    operation RejectOutputVertexAsParameter() : Unit { let _ = MeasurementOrder(2, [[0], [2]]); }
    operation RejectEmptySlice() : Unit { let _ = MeasurementOrder(1, [[], [0]]); }
    operation RejectRepeatedAngles() : Unit { ValidateDiscreteInputs([0.0], [[0]], [0.0, 0.0]); }
    operation RejectInitialAngle() : Unit { ValidateDiscreteInputs([0.5], [[0]], [0.0, 1.0]); }
    operation RejectGreedyBudget() : Unit {
        let _ = GreedyOptimize(IncreasingLoss, [0.0], [[0]], [0.0], 0.0, 0, 1, 0.1, 0);
    }
    operation RejectDifferenceStep() : Unit { let _ = CentralDifferenceGradient(IncreasingLoss, [0.0], 0.0); }
    operation RejectDqnNetworkShape() : Unit { let _ = QNetworkForward(SmallNetwork(), [0.0, 1.0]); }
    operation RejectReplayAction() : Unit {
        let sample = new ReplayTransition { State = [0.0], Action = 2, Reward = 0.0, NextState = [], Terminal = true };
        let _ = DqnMinibatchStep(SmallNetwork(), SmallNetwork(), [sample], 0.9, 0.1);
    }

    operation RunAll() : Int {
        TestMeasurementOrder();
        TestGreedyCanonicalOrder();
        TestGreedyJointWindow();
        TestGreedyExplorationAndBudget();
        TestGradients();
        TestAdam();
        TestDqnStateEncoding();
        TestDqnEpisodeOrderAndRewards();
        TestDqnForwardAndBellman();
        TestDqnHandCalculatedSgd();
        TestDqnBackpropagationNumerically();
        TestDqnReplayAndSchedules();
        TestDqnDriver();
        TestEmptyOptimizers();
        return 14;
    }
}
