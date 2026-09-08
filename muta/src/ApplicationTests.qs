namespace MuTA.ApplicationTests {
    import Std.Arrays.*;
    import Std.Convert.*;
    import Std.Diagnostics.*;
    import Std.Math.*;
    import Std.Measurement.*;
    import MuTA.Core.*;
    import MuTA.Applications.*;

    function Near(actual : Double, expected : Double, tolerance : Double, description : String) : Unit {
        if IsNaN(actual) or IsInfinite(actual) or AbsD(actual - expected) > tolerance {
            fail $"{description}: expected {expected}, got {actual}.";
        }
    }

    operation ZeroSample(_sample : Int, _register : Qubit[]) : Unit is Adj + Ctl { }
    operation IdentityTarget(_register : Qubit[]) : Unit is Adj + Ctl { }
    operation XTarget(register : Qubit[]) : Unit is Adj + Ctl { X(register[0]); }

    operation ComplexSample(sample : Int, register : Qubit[]) : Unit is Adj + Ctl {
        Ry(0.27 * IntAsDouble(sample + 1), register[0]);
        Rz(0.47 * IntAsDouble(sample + 1), register[0]);
    }

    operation ClassifierSample(sample : Int, register : Qubit[]) : Unit is Adj + Ctl {
        if sample == 1 { PrepareStateFamily(0, PI() / 4.0, 0.37, register); }
    }

    operation TestStatePreparationAndGateLoss() : Unit {
        // Independent basis-state/phase assertions, not just U†U cancellation.
        use q = Qubit[2];
        PrepareVector(new StateVector { Real = [1.0, 0.0, 0.0, 0.0], Imaginary = [0.0, 1.0, 0.0, 0.0] }, q);
        if Measure([PauliY], [q[0]]) != Zero or M(q[1]) != Zero {
            fail "Complex state-vector preparation lost phase or little-endian ordering.";
        }
        ResetAll(q);
        PrepareVector(new StateVector { Real = [0.0, 0.0, 1.0, 0.0], Imaginary = [0.0, 0.0, 0.0, 0.0] }, q);
        if MResetZ(q[0]) != Zero or MResetZ(q[1]) != One {
            fail "State-vector preparation has incorrect basis ordering.";
        }
        let pattern = Layer(1, -1, []);
        let angles = [0.0, size = 4];
        let dataset = SampleHaarDataset(1, 6);
        let loss = AverageGateInfidelity(pattern, angles, PrepareDatasetState(dataset, _, _), IdentityTarget, 6, 8);
        Near(loss, 0.0, 1e-10, "Identity on Haar states");
        Near(AverageGateInfidelity(pattern, angles, ZeroSample, XTarget, 1, 8), 1.0, 1e-10, "Wrong gate must have nonzero loss");
        let nonzero = [0.31, -0.27, 0.49, 0.18];
        let reference = ApplyCompiled(pattern, nonzero, _);
        Near(AverageGateInfidelity(pattern, nonzero, PrepareDatasetState(dataset, _, _), reference, 6, 8),
            0.0, 1e-10, "Gate learning objective on a nontrivial unitary");
        Message("Application state preparation and gate fidelity tests passed.");
    }

    operation TestNoiseModels() : Unit {
        let pattern = Layer(1, -1, []);
        let angles = [0.0, size = 4];
        let flip = SampleBitFlipProgram(1, 1.0);
        Near(AverageNoisyLabelInfidelity(pattern, angles, ZeroSample, IdentityTarget, [flip], 8),
            1.0, 1e-10, "Bitflip affects labels rather than model resource");
        let noFlip = SampleBitFlipProgram(1, 0.0);
        Near(AverageNoisyLabelInfidelity(pattern, angles, ZeroSample, IdentityTarget, [noFlip], 8),
            0.0, 1e-10, "Zero label noise");
        Near(AverageNoisyResourceInfidelity(pattern, angles, ComplexSample, IdentityTarget, 3, 8, 0.0),
            0.0, 1e-10, "Zero resource noise");
        let noisyLoss = AverageNoisyResourceInfidelity(pattern, angles, ZeroSample, IdentityTarget, 1, 128, 0.75);
        Near(noisyLoss, 0.5, 0.18, "Completely depolarized resource gives mixed output fidelity");
        mutable ones = 0;
        for _ in 1..192 {
            use q = Qubit();
            ApplyDepolarizingResource(1.0, [q]);
            if MResetZ(q) == One { set ones += 1; }
        }
        Near(IntAsDouble(ones) / 192.0, 2.0 / 3.0, 0.16, "Appendix D depolarizing convention");
        let brownianZero = SampleBrownianProgram(1, 2, 0.0, 2);
        Near(AverageNoisyLabelInfidelity(pattern, angles, ComplexSample, IdentityTarget,
            [brownianZero, size = 3], 8), 0.0, 1e-10, "Zero Brownian time");
        let brownian = SampleBrownianProgram(2, 2, 0.3, 2);
        if Length(brownian.Paulis) != 120 or Length(brownian.Angles) != 120 {
            fail "Brownian program lost steps, Pauli terms, or symmetric Trotter factors.";
        }
        use register = Qubit[2];
        H(register[0]);
        ApplyNoiseProgram(brownian, register);
        Adjoint ApplyNoiseProgram(brownian, register);
        H(register[0]);
        if not AllZeroAndReset(register) { fail "A fixed Brownian target label must remain reversible."; }
        Message("Application target-noise and resource-noise tests passed.");
    }

    operation TestClassifier() : Unit {
        let pattern = Layer(2, -1, []);
        let columns = [0.0, size = 4];
        let angles = TiedClassifierAngles(pattern, columns);
        let ground = ClassifierProbabilities(pattern, angles, ClassifierSample(0, _), 32);
        let bell = ClassifierProbabilities(pattern, angles, ClassifierSample(1, _), 128);
        Near(QfiFromProbabilities(ground), 0.0, 1e-10, "Ground-state QFI");
        Near(QfiFromProbabilities(bell), 4.0, 0.25, "Bell-state QFI under normalized generator");
        let exactHead = [0.0, 4.0, 4.0, -4.0, 8.0, -4.0];
        Near(PolynomialHead(exactHead, (0.2, 0.3)), QfiFromProbabilities((0.2, 0.3)),
            1e-10, "Quadratic head spans QFI");
        Near(ClassifierMarginLoss([0.0, 4.0], [0, 1], 0.5), 0.0, 1e-10, "Correct classifier margin");
        Near(ClassifierMarginLoss([4.0, 0.0], [0, 1], 0.5), 2.5, 1e-10, "Incorrect classifier margin");
        Near(QuantumClassifierLoss(pattern, columns, exactHead, ClassifierSample, [0, 1], 128, 0.5),
            0.0, 1e-10, "Quantum classifier objective");
        Message("Application joint-probability classifier tests passed.");
    }

    operation TestTeleportation() : Unit {
        let parameters = ExactTeleportationParameters();
        let dataset = SampleHaarDataset(1, 8);
        Near(TeleportationInfidelity(parameters, PrepareDatasetState(dataset, _, _), 8, 8),
            0.0, 1e-10, "MuTA teleportation preserves arbitrary complex inputs");
        // All four classical branches should be reachable, even though flow
        // measurement outcomes are separately randomized within each stage.
        mutable branchCounts = [0, size = 4];
        for _ in 1..64 {
            use message = Qubit();
            use output = Qubit();
            H(message);
            S(message);
            let outcomes = TeleportWithMuTA(parameters[0..7], parameters[8..15], parameters[16..19], message, output);
            let index = (outcomes[0] == One ? 2 | 0) + (outcomes[1] == One ? 1 | 0);
            set branchCounts w/= index <- branchCounts[index] + 1;
            Adjoint S(output);
            H(output);
            if MResetZ(output) != Zero { fail "Teleportation branch correction failed."; }
        }
        for count in branchCounts { if count == 0 { fail "Teleportation did not exercise every instrument outcome."; } }
        let uncorrected = parameters[0..15] + [0.0, size = 4];
        let loss = TeleportationInfidelity(uncorrected, ComplexSample, 4, 32);
        if loss < 0.2 { fail "Removing outcome-dependent corrections should damage teleportation."; }
        Message("Application MuTA instrument and outcome-dependent correction tests passed.");
    }

    operation TestKernelsAndSvm() : Unit {
        let pattern = Layer(2, 0, [1]);
        let points = [[0.7, -0.3], [1.2, 0.4], [-0.2, 0.8]];
        for point in points {
            use output = Qubit[2];
            PrepareMuTAKernel(point, output);
            Adjoint PrepareKernel(point, output);
            if not AllZeroAndReset(output) { fail "MuTA graph feature map differs from Eq. (5)."; }
            Near(FidelityKernel(2, PrepareKernel(point, _), ApplyCompiled(pattern, KernelAngles(point), _), 8),
                1.0, 1e-10, "Compiled feature map");
            Near(MuTAKernel(point, point, 16), 1.0, 1e-10, "Kernel self-overlap");
        }
        let expected = ExactKernel(points[0], points[1]);
        Near(FidelityKernel(2, PrepareKernel(points[0], _), PrepareKernel(points[1], _), 192),
            expected, 0.16, "Fidelity kernel against closed form");
        Near(MuTAKernel(points[0], points[1], 256), expected, 0.20, "SWAP kernel against closed form");
        let matrix = KernelMatrix(points[0..1], 16);
        Near(matrix[0][0], 1.0, 1e-10, "Gram diagonal");
        Near(matrix[0][1], matrix[1][0], 1e-10, "Gram symmetry");
        // A separable nontrivial Gram matrix; zero alpha / constant output fails.
        let gram = [[4.0, 2.0, -2.0, -4.0], [2.0, 1.0, -1.0, -2.0],
            [-2.0, -1.0, 1.0, 2.0], [-4.0, -2.0, 2.0, 4.0]];
        let labels = [1, 1, -1, -1];
        let model = TrainSvm(gram, labels, 2.0, 1e-5, 4, 100);
        mutable equality = 0.0;
        for i in 0..3 {
            if PredictSvm(model, gram[i]) != labels[i] { fail "SVM failed on a separable dataset."; }
            if model.Alpha[i] < -1e-8 or model.Alpha[i] > 2.0 + 1e-8 { fail "SVM box constraint violated."; }
            set equality += model.Alpha[i] * IntAsDouble(labels[i]);
        }
        Near(equality, 0.0, 1e-8, "SVM dual equality constraint");
        let duplicate = TrainSvm([[1.0, 1.0], [1.0, 1.0]], [1, -1], 2.0, 1e-5, 4, 30);
        Near(duplicate.Alpha[0], 2.0, 1e-8, "SVM identical features require endpoint optimization");
        Near(duplicate.Alpha[1], 2.0, 1e-8, "SVM conflicting labels preserve the dual equality");
        // Train and infer with the actual paper kernel, independently of linear test.
        let data = [[0.0, 0.0], [0.1, 0.0], [1.57, 0.0], [1.47, 0.0]];
        let kernel = Mapped(x -> Mapped(y -> ExactKernel(x, y), data), data);
        let kernelModel = TrainSvm(kernel, labels, 100.0, 1e-5, 4, 200);
        for i in 0..3 {
            if PredictSvm(kernelModel, kernel[i]) != labels[i] { fail "SVM failed with the MuTA feature-map kernel."; }
        }
        Message("Application Eq. (5), measured kernels, and SVM optimization tests passed.");
    }

    operation ApplicationsSelfTest() : Unit {
        TestStatePreparationAndGateLoss();
        TestNoiseModels();
        TestClassifier();
        TestTeleportation();
        TestKernelsAndSvm();
        Message("All MuTA application self-tests passed.");
    }

    operation RunAll() : Unit {
        ApplicationsSelfTest();
    }
}
