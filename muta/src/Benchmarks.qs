namespace MuTA.Benchmarks {
    import MuTA.Core.*;
    import MuTA.Applications.*;
    import MuTA.Training.*;
    import Std.Arrays.*;
    import Std.Convert.*;
    import Std.Math.*;
    import Std.Measurement.*;
    import Std.Random.*;

    /// Serializable Haar samples retain their complex amplitudes, not just seeds.
    operation HaarData(width : Int, count : Int) : (Double[], Double[])[] {
        return Mapped(s -> (s.Real, s.Imaginary), SampleHaarDataset(width, count));
    }

    operation PrepareSavedState(
        data : (Double[], Double[])[], sample : Int, register : Qubit[]
    ) : Unit is Adj + Ctl {
        let (real, imaginary) = data[sample];
        PrepareVector(new StateVector { Real = real, Imaginary = imaginary }, register);
    }

    function GatePattern(task : Int) : Pattern {
        if task < 0 or task > 2 { fail "Benchmark gate task must be 0, 1, or 2."; }
        return task == 1 ? Layer(2, 0, [1]) | Layer(1, -1, []);
    }

    /// Independent circuit targets: ZYZ, a locally dressed XX gate, and an
    /// attainable four-slot restricted-angle task (not a one-free-angle search).
    operation GateTarget(task : Int, register : Qubit[]) : Unit is Adj + Ctl {
        if task == 0 {
            ApplySingleQubitTarget([0.47, -0.31, 0.83], register);
        } elif task == 1 {
            Rz(0.31, register[0]);
            Rz(-0.27, register[1]);
            Rx(0.19, register[0]);
            ApplyIsingXX(0.8, register);
            Rz(-0.42, register[0]);
            Rz(0.23, register[1]);
            Rx(0.15, register[0]);
            Rx(-0.35, register[1]);
        } elif task == 2 {
            Rz(-PI() / 4.0, register[0]);
            Rx(-PI() / 2.0, register[0]);
            Rz(-PI() / 4.0, register[0]);
        } else { fail "Unknown benchmark gate target."; }
    }

    function AdamBudget(steps : Int, rate : Double, shift : Bool, count : Int) : AdamConfig {
        return new AdamConfig {
            Steps = steps, LearningRate = rate, Beta1 = 0.9, Beta2 = 0.999,
            Epsilon = 1e-8, Threshold = -1.0, UseParameterShift = shift,
            DifferenceStep = 0.2, MaxEvaluations = 1 + steps * (2 * count + 1)
        };
    }

    operation FitGate(
        task : Int, optimizer : Int, data : (Double[], Double[])[],
        initial : Double[], shots : Int, steps : Int, discreteBudget : Int
    ) : TrainingResult {
        let pattern = GatePattern(task);
        ValidateAngles(pattern, initial);
        let objective = AverageGateInfidelity(pattern, _, PrepareSavedState(data, _, _),
            GateTarget(task, _), Length(data), shots);
        if optimizer == 0 {
            return TrainAdam(objective, initial, AdamBudget(steps, 0.18, true, Length(initial)));
        }
        let slices = ParameterSlices(pattern);
        let alphabet = [0.0, PI() / 4.0, PI() / 2.0];
        if optimizer == 1 {
            return GreedyOptimize(objective, initial, slices, alphabet,
                0.1, discreteBudget, 2, -1.0, discreteBudget);
        }
        if optimizer == 2 {
            let config = new DqnConfig {
                Episodes = discreteBudget - 1, HiddenSize = 16, LearningRate = 0.03,
                Gamma = 0.95, EpsilonStart = 1.0, EpsilonMin = 0.1,
                EpsilonDecay = 0.97, ReplayCapacity = 256, BatchSize = 16,
                UpdatesPerEpisode = 4, TargetPeriod = 5, Threshold = -1.0,
                MaxEvaluations = discreteBudget
            };
            return TrainDqn(objective, initial, slices, alphabet, config);
        }
        fail "Unknown benchmark optimizer.";
    }

    operation GateInference(
        task : Int, parameters : Double[], data : (Double[], Double[])[],
        shots : Int, resourceNoise : Double
    ) : Double[] {
        CheckSampling(Length(data), shots);
        CheckProbability(resourceNoise);
        let pattern = GatePattern(task);
        mutable losses = [];
        for sample in data {
            set losses += [AverageNoisyResourceInfidelity(pattern, parameters,
                PrepareSavedState([sample], _, _), GateTarget(task, _), 1, shots, resourceNoise)];
        }
        return losses;
    }

    operation CompilationInfidelity(
        task : Int, parameters : Double[], data : (Double[], Double[])[], shots : Int
    ) : Double {
        let pattern = GatePattern(task);
        return AverageGateInfidelity(pattern, parameters, PrepareSavedState(data, _, _),
            ApplyCompiled(pattern, parameters, _), Length(data), shots);
    }

    function StateQfi(family : Int, theta : Double, phi : Double) : Double {
        CheckFinite(theta, "State theta");
        CheckFinite(phi, "State phi");
        if family == 0 {
            return QfiFromProbabilities((Cos(theta) * Cos(theta), Sin(theta) * Sin(theta)));
        }
        if family == 1 {
            let p = (1.0 + Sin(2.0 * theta) * Cos(phi)) / 4.0;
            return QfiFromProbabilities((p, p));
        }
        fail "Unknown classifier state family.";
    }

    /// Uniform theta/phi draws, balanced by label within each family. No margin
    /// exclusion is applied: near-SQL states remain in the held-out population.
    operation ClassifierData(perFamily : Int) : (Int, Double, Double, Double, Int)[] {
        if perFamily < 2 or perFamily % 2 != 0 {
            fail "Classifier count per family must be positive and even.";
        }
        mutable data = [];
        for family in 0..1 {
            mutable counts = [0, 0];
            mutable attempts = 0;
            while counts[0] + counts[1] < perFamily {
                set attempts += 1;
                if attempts > 1000 * perFamily { fail "Classifier rejection sampler exhausted its budget."; }
                let theta = DrawRandomDouble(0.0, PI() / 2.0);
                let phi = DrawRandomDouble(0.0, 2.0 * PI());
                let qfi = StateQfi(family, theta, phi);
                let label = qfi > 2.0 ? 1 | 0;
                if counts[label] < perFamily / 2 {
                    set data += [(family, theta, phi, qfi, label)];
                    set counts w/= label <- counts[label] + 1;
                }
            }
        }
        return data;
    }

    operation PrepareClassifierState(
        data : (Int, Double, Double, Double, Int)[], sample : Int, register : Qubit[]
    ) : Unit is Adj + Ctl {
        let (family, theta, phi, _, _) = data[sample];
        PrepareStateFamily(family, theta, phi, register);
    }

    function ClassifierLabels(data : (Int, Double, Double, Double, Int)[]) : Int[] {
        mutable labels = [];
        for (family, theta, phi, qfi, label) in data {
            CheckFinite(qfi, "Stored QFI");
            if AbsD(qfi - StateQfi(family, theta, phi)) > 1e-10
                or label != (qfi > 2.0 ? 1 | 0) {
                fail "Classifier labels must agree with the analytic normalized QFI.";
            }
            set labels += [label];
        }
        return labels;
    }

    operation ClassifierObjective(
        data : (Int, Double, Double, Double, Int)[], shots : Int, parameters : Double[]
    ) : Double {
        if Length(parameters) != 10 { fail "Classifier needs four tied angles and six head coefficients."; }
        return QuantumClassifierLoss(Layer(2, -1, []), parameters[0..3], parameters[4..9],
            PrepareClassifierState(data, _, _), ClassifierLabels(data), shots, 0.5);
    }

    operation FitClassifier(
        data : (Int, Double, Double, Double, Int)[], initial : Double[], shots : Int, steps : Int
    ) : TrainingResult {
        return TrainAdam(ClassifierObjective(data, shots, _), initial,
            AdamBudget(steps, 0.12, false, 10));
    }

    /// The reference uses known QFI postprocessing and an untuned identity
    /// measurement basis; it is a task-informed baseline, not a learned model.
    operation ClassifierInference(
        data : (Int, Double, Double, Double, Int)[], parameters : Double[],
        shots : Int, reference : Bool
    ) : (Double, Double, Double, Int)[] {
        CheckSampling(Length(data), shots);
        if Length(parameters) != 10 { fail "Classifier needs ten saved parameters."; }
        let pattern = Layer(2, -1, []);
        let angles = TiedClassifierAngles(pattern, reference ? [0.0, size = 4] | parameters[0..3]);
        mutable predictions = [];
        for index in 0..Length(data) - 1 {
            let probabilities = ClassifierProbabilities(pattern, angles,
                PrepareClassifierState(data, index, _), shots);
            let score = reference ? QfiFromProbabilities(probabilities)
                | PolynomialHead(parameters[4..9], probabilities);
            let (p00, p11) = probabilities;
            set predictions += [(p00, p11, score, score > 2.0 ? 1 | 0)];
        }
        return predictions;
    }

    function TeleportParameters(parameters : Double[], full : Bool) : Double[] {
        if Length(parameters) != (full ? 20 | 4) { fail "Incorrect teleportation parameter count."; }
        return full ? parameters | ExactTeleportationParameters()[0..15] + parameters;
    }

    operation RandomAngles(count : Int) : Double[] {
        if count < 1 { fail "Random angle count must be positive."; }
        mutable angles = [];
        for _ in 1..count { set angles += [DrawRandomDouble(-PI(), PI())]; }
        return angles;
    }

    operation TeleportObjective(
        data : (Double[], Double[])[], shots : Int, full : Bool, parameters : Double[]
    ) : Double {
        return TeleportationInfidelity(TeleportParameters(parameters, full),
            PrepareSavedState(data, _, _), Length(data), shots);
    }

    operation FitTeleport(
        data : (Double[], Double[])[], initial : Double[], shots : Int, steps : Int, full : Bool
    ) : TrainingResult {
        return TrainAdam(TeleportObjective(data, shots, full, _), initial,
            AdamBudget(steps, 0.18, true, Length(initial)));
    }

    /// Retain every instrument outcome, with per-branch trials and failures.
    operation TeleportInference(
        data : (Double[], Double[])[], parameters : Double[], shots : Int
    ) : (Double[], Int[], Int[]) {
        CheckSampling(Length(data), shots);
        if Length(parameters) != 20 { fail "Inference requires all twenty instrument angles."; }
        mutable losses = [];
        mutable trials = [0, size = 4];
        mutable failures = [0, size = 4];
        for sample in 0..Length(data) - 1 {
            mutable failed = 0;
            for _ in 1..shots {
                use message = Qubit[1];
                use output = Qubit[1];
                PrepareSavedState(data, sample, message);
                let outcomes = TeleportWithMuTA(parameters[0..7], parameters[8..15],
                    parameters[16..19], message[0], output[0]);
                let branch = (outcomes[0] == One ? 2 | 0) + (outcomes[1] == One ? 1 | 0);
                set trials w/= branch <- trials[branch] + 1;
                Adjoint PrepareSavedState(data, sample, output);
                if not AllZeroAndReset(output) {
                    set failed += 1;
                    set failures w/= branch <- failures[branch] + 1;
                }
            }
            set losses += [IntAsDouble(failed) / IntAsDouble(shots)];
        }
        return (losses, trials, failures);
    }

    operation KernelData(task : Int, perClass : Int) : (Double[][], Int[]) {
        if task < 0 or task > 1 or perClass < 1 { fail "Invalid kernel dataset configuration."; }
        mutable points = [];
        mutable labels = [];
        for label in [-1, 1] {
            for _ in 1..perClass {
                if task == 0 {
                    let center = label == -1 ? 0.15 | 1.35;
                    set points += [[center + DrawRandomDouble(-0.2, 0.2),
                        center + DrawRandomDouble(-0.2, 0.2)]];
                } else {
                    let sign = DrawRandomInt(0, 1) == 0 ? -1.0 | 1.0;
                    set points += [[0.9 * sign + DrawRandomDouble(-0.25, 0.25),
                        0.9 * sign * IntAsDouble(label) + DrawRandomDouble(-0.25, 0.25)]];
                }
                set labels += [label];
            }
        }
        return (points, labels);
    }

    operation FitKernel(
        points : Double[][], labels : Int[], shots : Int, exact : Bool
    ) : (SvmModel, Double[][]) {
        let matrix = exact ? Mapped(x -> Mapped(y -> ExactKernel(x, y), points), points)
            | KernelMatrix(points, shots);
        return (TrainSvm(matrix, labels, 10.0, 1e-4, 8, 200), matrix);
    }

    operation KernelInference(
        training : Double[][], testing : Double[][], alpha : Double[], labels : Int[],
        bias : Double, shots : Int, exact : Bool
    ) : (Int[], Double[], Double[][]) {
        let model = new SvmModel {
            Alpha = alpha, Labels = labels, Bias = bias, Iterations = 0, StablePasses = 0
        };
        if Length(training) != Length(alpha) { fail "Saved SVM training vectors must retain their order."; }
        mutable predictions = [];
        mutable scores = [];
        mutable similarities = [];
        for point in testing {
            mutable row = [];
            for trainPoint in training {
                set row += [exact ? ExactKernel(point, trainPoint) | MuTAKernel(point, trainPoint, shots)];
            }
            set predictions += [PredictSvm(model, row)];
            set scores += [SvmScore(model, row)];
            set similarities += [row];
        }
        return (predictions, scores, similarities);
    }

    operation KernelEstimates(points : Double[][], shots : Int) : (Double[], Double[]) {
        if Length(points) < 2 { fail "Kernel diagnostics need at least two points."; }
        mutable measured = [];
        mutable exact = [];
        for i in 0..Length(points) - 1 {
            let j = (i + 1) % Length(points);
            set measured += [MuTAKernel(points[i], points[j], shots)];
            set exact += [ExactKernel(points[i], points[j])];
        }
        return (measured, exact);
    }

    function EncodeNoise(programs : PauliProgram[]) : (Int[][], Double[])[] {
        return Mapped(program -> (Mapped(word -> Mapped(p ->
            p == PauliI ? 0 | p == PauliX ? 1 | p == PauliY ? 2 | 3, word),
            program.Paulis), program.Angles), programs);
    }

    function DecodePauli(code : Int) : Pauli {
        if code < 0 or code > 3 { fail "Serialized Pauli code must be in 0..3."; }
        return code == 0 ? PauliI | code == 1 ? PauliX | code == 2 ? PauliY | PauliZ;
    }

    function DecodeNoise(encoded : (Int[][], Double[])[]) : PauliProgram[] {
        mutable programs = [];
        for (words, angles) in encoded {
            set programs += [new PauliProgram {
                Paulis = Mapped(word -> Mapped(DecodePauli, word), words), Angles = angles
            }];
        }
        return programs;
    }

    operation LabelNoise(count : Int, mode : Int, level : Double) : (Int[][], Double[])[] {
        if count < 1 or mode < 0 or mode > 3 { fail "Invalid noise benchmark configuration."; }
        CheckProbability(level);
        mutable programs = [];
        for _ in 1..count {
            let program = mode == 1 ? SampleBitFlipProgram(1, level)
                | mode == 2 ? SampleBrownianProgram(1, 3, level, 2) | EmptyNoise();
            set programs += [program];
        }
        return EncodeNoise(programs);
    }

    operation FitNoisyGate(
        data : (Double[], Double[])[], initial : Double[], mode : Int, level : Double,
        encodedNoise : (Int[][], Double[])[], shots : Int, steps : Int
    ) : TrainingResult {
        if mode < 0 or mode > 3 or Length(encodedNoise) != Length(data) {
            fail "Noise mode and fixed label program count must match the training data.";
        }
        CheckProbability(level);
        let pattern = GatePattern(0);
        let prepare = PrepareSavedState(data, _, _);
        let target = GateTarget(0, _);
        let labelObjective = AverageNoisyLabelInfidelity(pattern, _, prepare, target,
            DecodeNoise(encodedNoise), shots);
        let resourceObjective = AverageNoisyResourceInfidelity(pattern, _, prepare,
            target, Length(data), shots, level);
        return TrainAdam(mode == 3 ? resourceObjective | labelObjective, initial,
            AdamBudget(steps, 0.18, true, 4));
    }

    /// Logical graph sizes only; no allocation or physical-device estimation.
    function LogicalResources(width : Int, depth : Int) : (Int, Int, Int, Int, Int) {
        if width < 1 or depth < 1 { fail "Resource width and depth must be positive."; }
        mutable targets = [];
        for row in 1..width - 1 { set targets += [row]; }
        let layer = Layer(width, width == 1 ? -1 | 0, targets);
        mutable pattern = layer;
        for _ in 2..depth {
            set pattern = Concatenate(pattern, layer, Zipped(pattern.Outputs, layer.Inputs));
        }
        return (pattern.VertexCount, Length(pattern.Edges), Length(pattern.Order),
            Length(ParameterSlices(pattern)), Length(Compile(pattern, [0.0, size = Length(pattern.Order)])));
    }
}
