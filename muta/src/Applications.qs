namespace MuTA.Applications {
    import Std.Arrays.*;
    import Std.Convert.*;
    import Std.Math.*;
    import Std.Measurement.*;
    import Std.Random.*;
    import MuTA.Core.*;

    /// A sampled unitary noise realization. Sampling is separate from application,
    /// so a noisy target label remains fixed across repeated shots and epochs.
    struct PauliProgram {
        Paulis : Pauli[][],
        Angles : Double[]
    }

    /// Complex amplitudes in little-endian computational-basis order.
    /// PrepareVector normalizes the vector; it must have nonzero norm.
    struct StateVector {
        Real : Double[],
        Imaginary : Double[]
    }

    struct SvmModel {
        Alpha : Double[],
        Labels : Int[],
        Bias : Double,
        Iterations : Int,
        StablePasses : Int
    }

    function CheckProbability(p : Double) : Unit {
        CheckFinite(p, "Probability");
        if p < 0.0 or p > 1.0 { fail "Probability must lie in [0, 1]."; }
    }

    function CheckFinite(value : Double, name : String) : Unit {
        if IsNaN(value) or IsInfinite(value) { fail $"{name} must be finite."; }
    }

    function CheckFiniteArray(values : Double[], name : String) : Unit {
        for value in values { CheckFinite(value, name); }
    }

    function CheckKernelPoint(data : Double[]) : Unit {
        if Length(data) != 2 { fail "The paper feature map is two-dimensional."; }
        CheckFiniteArray(data, "Feature-map coordinate");
    }

    function CheckJointProbabilities(probabilities : (Double, Double)) : Unit {
        let (plus, minus) = probabilities;
        CheckProbability(plus);
        CheckProbability(minus);
        if plus + minus > 1.0 + 1e-12 { fail "Joint outcome probabilities cannot sum above one."; }
    }

    function CheckSampling(sampleCount : Int, shots : Int) : Unit {
        if sampleCount <= 0 or shots <= 0 {
            fail "Dataset size and shots must be positive.";
        }
    }

    function EmptyNoise() : PauliProgram {
        return new PauliProgram { Paulis = [], Angles = [] };
    }

    operation ApplyNoiseProgram(program : PauliProgram, register : Qubit[]) : Unit is Adj + Ctl {
        if Length(program.Paulis) != Length(program.Angles) {
            fail "Noise program has inconsistent term and angle counts.";
        }
        CheckFiniteArray(program.Angles, "Noise angle");
        for j in 0..Length(program.Angles) - 1 {
            if Length(program.Paulis[j]) != Length(register) {
                fail "Noise Pauli word does not match the register width.";
            }
            Exp(program.Paulis[j], program.Angles[j], register);
        }
    }

    operation SampleBitFlipProgram(width : Int, probability : Double) : PauliProgram {
        CheckProbability(probability);
        if width <= 0 { fail "Noise width must be positive."; }
        mutable words = [];
        mutable angles = [];
        for q in 0..width - 1 {
            if DrawRandomDouble(0.0, 1.0) < probability {
                mutable word = [PauliI, size = width];
                set word w/= q <- PauliX;
                set words += [word];
                set angles += [PI() / 2.0];
            }
        }
        return new PauliProgram { Paulis = words, Angles = angles };
    }

    operation StandardNormal() : Double {
        // Box-Muller; avoid log(0) if the pseudorandom source returns its endpoint.
        let u = DrawRandomDouble(1e-15, 1.0);
        let v = DrawRandomDouble(0.0, 1.0);
        return Sqrt(-2.0 * Log(u)) * Cos(2.0 * PI() * v);
    }

    function PauliWord(index : Int, width : Int) : Pauli[] {
        mutable word = [];
        mutable remaining = index;
        for _ in 1..width {
            let digit = remaining % 4;
            set word += [digit == 0 ? PauliI | digit == 1 ? PauliX | digit == 2 ? PauliY | PauliZ];
            set remaining /= 4;
        }
        return word;
    }

    /// Discretized Brownian target noise, Sec. IV B. Each H_j is an independent
    /// isotropic Gaussian Pauli Hamiltonian with coefficient variance 1 / 2^n.
    /// exp(i H_j dt) is implemented with symmetric second-order product formulas.
    /// Increase trotterSteps to check convergence; this is an approximation, not
    /// an assertion that the paper's unpublished random seeds were reproduced.
    operation SampleBrownianProgram(
        width : Int, steps : Int, deltaTime : Double, trotterSteps : Int
    ) : PauliProgram {
        CheckFinite(deltaTime, "Brownian time step");
        // Full Gaussian Pauli Hamiltonians scale as 4^width. Keep this dataset
        // helper explicitly small; scalable structured noise can use PauliProgram.
        if width <= 0 or width > 4 or steps <= 0 or trotterSteps <= 0 or deltaTime < 0.0 {
            fail "Brownian noise supports 1..4 qubits, positive step counts, and nonnegative time.";
        }
        let dimension = 1 <<< width;
        let termCount = (1 <<< (2 * width)) - 1;
        let scale = deltaTime / (2.0 * IntAsDouble(trotterSteps) * Sqrt(IntAsDouble(dimension)));
        mutable words = [];
        mutable angles = [];
        for _ in 1..steps {
            mutable coefficients = [];
            for _ in 1..termCount { set coefficients += [scale * StandardNormal()]; }
            for _ in 1..trotterSteps {
                for j in 1..termCount {
                    set words += [PauliWord(j, width)];
                    set angles += [coefficients[j - 1]];
                }
                for j in termCount..-1..1 {
                    set words += [PauliWord(j, width)];
                    set angles += [coefficients[j - 1]];
                }
            }
        }
        return new PauliProgram { Paulis = words, Angles = angles };
    }

    /// Appendix D's convention: (1-p)rho + p/3 (XrhoX + YrhoY + ZrhoZ).
    /// ExecuteWithNoise calls this after preparation of the entire resource.
    /// Repeated trajectories retain every measurement branch in the mixed-state
    /// fidelity estimate; no measurement outcome is postselected.
    operation ApplyDepolarizingResource(probability : Double, resource : Qubit[]) : Unit {
        CheckProbability(probability);
        for q in resource {
            let r = DrawRandomDouble(0.0, 1.0);
            if r < probability / 3.0 { X(q); }
            elif r < 2.0 * probability / 3.0 { Y(q); }
            elif r < probability { Z(q); }
        }
    }

    operation SampleHaarVector(width : Int) : StateVector {
        if width <= 0 or width > 8 { fail "Full Haar-vector generation supports 1..8 qubits."; }
        mutable real = [];
        mutable imaginary = [];
        for _ in 1..(1 <<< width) {
            set real += [StandardNormal()];
            set imaginary += [StandardNormal()];
        }
        return new StateVector { Real = real, Imaginary = imaginary };
    }

    function SquaredAmplitude(vector : StateVector, index : Int) : Double {
        return vector.Real[index] * vector.Real[index]
            + vector.Imaginary[index] * vector.Imaginary[index];
    }

    function VectorNorm(vector : StateVector) : Double {
        mutable norm = 0.0;
        for j in 0..Length(vector.Real) - 1 { set norm += SquaredAmplitude(vector, j); }
        return norm;
    }

    function ConditionalMagnitudeAngle(vector : StateVector, depth : Int, prefix : Int) : Double {
        let prefixCount = 1 <<< depth;
        mutable zeroWeight = 0.0;
        mutable oneWeight = 0.0;
        for j in 0..Length(vector.Real) - 1 {
            if j % prefixCount == prefix {
                if (j >>> depth) % 2 == 0 { set zeroWeight += SquaredAmplitude(vector, j); }
                else { set oneWeight += SquaredAmplitude(vector, j); }
            }
        }
        return zeroWeight + oneWeight > 0.0
            ? 2.0 * ArcTan2(Sqrt(oneWeight), Sqrt(zeroWeight)) | 0.0;
    }

    /// Exact state preparation by conditional magnitude rotations followed by
    /// basis-state phases. The exponential classical work is intentional: this
    /// supplies small Haar datasets, not an efficient arbitrary-data loader.
    operation PrepareVector(vector : StateVector, register : Qubit[]) : Unit is Adj + Ctl {
        let width = Length(register);
        if width <= 0 or width > 8 { fail "Full state-vector preparation supports 1..8 qubits."; }
        let dimension = 1 <<< width;
        if width <= 0 or Length(vector.Real) != dimension or Length(vector.Imaginary) != dimension {
            fail "State-vector length must be 2^register width.";
        }
        CheckFiniteArray(vector.Real, "State-vector real amplitude");
        CheckFiniteArray(vector.Imaginary, "State-vector imaginary amplitude");
        CheckFinite(VectorNorm(vector), "State-vector squared norm");
        if VectorNorm(vector) <= 0.0 { fail "State vector must have nonzero norm."; }
        for depth in 0..width - 1 {
            let prefixCount = 1 <<< depth;
            for prefix in 0..prefixCount - 1 {
                let angle = ConditionalMagnitudeAngle(vector, depth, prefix);
                within {
                    for j in 0..depth - 1 {
                        if (prefix >>> j) % 2 == 0 { X(register[j]); }
                    }
                } apply {
                    Controlled Ry(register[0..depth - 1], (angle, register[depth]));
                }
            }
        }
        for basis in 0..dimension - 1 {
            let phase = SquaredAmplitude(vector, basis) > 0.0
                ? ArcTan2(vector.Imaginary[basis], vector.Real[basis]) | 0.0;
            within {
                for j in 0..width - 1 {
                    if (basis >>> j) % 2 == 0 { X(register[j]); }
                }
            } apply {
                Controlled R1(register[0..width - 2], (phase, register[width - 1]));
            }
        }
    }

    operation PrepareDatasetState(dataset : StateVector[], sample : Int, register : Qubit[]) : Unit is Adj + Ctl {
        PrepareVector(dataset[sample], register);
    }

    operation SampleHaarDataset(width : Int, count : Int) : StateVector[] {
        if count <= 0 { fail "Dataset size must be positive."; }
        mutable dataset = [];
        for _ in 1..count { set dataset += [SampleHaarVector(width)]; }
        return dataset;
    }

    /// Haar-distributed SU(2) up to global phase, represented by Z-Y-Z angles.
    operation SampleHaarSingleQubit() : Double[] {
        return [2.0 * PI() * DrawRandomDouble(0.0, 1.0),
            2.0 * ArcCos(Sqrt(DrawRandomDouble(0.0, 1.0))),
            2.0 * PI() * DrawRandomDouble(0.0, 1.0)];
    }

    operation ApplySingleQubitTarget(euler : Double[], register : Qubit[]) : Unit is Adj + Ctl {
        if Length(euler) != 3 or Length(register) < 1 { fail "Expected three Euler angles and a qubit."; }
        CheckFiniteArray(euler, "Euler angle");
        Rz(euler[0], register[0]);
        Ry(euler[1], register[0]);
        Rz(euler[2], register[0]);
    }

    operation ApplyIsingXX(angle : Double, register : Qubit[]) : Unit is Adj + Ctl {
        if Length(register) != 2 { fail "Ising XX needs two qubits."; }
        CheckFinite(angle, "Ising angle");
        Exp([PauliX, PauliX], -angle / 2.0, register);
    }

    operation AllZeroAndReset(register : Qubit[]) : Bool {
        mutable success = true;
        for q in register { if MResetZ(q) == One { set success = false; } }
        return success;
    }

    /// Eq. (2), estimated by projection onto the reversible target preparation.
    /// prepare(sample, q) must prepare sample |psi_i> on zeroed q.
    operation AverageGateInfidelity(
        pattern : Pattern, angles : Double[],
        prepare : (Int, Qubit[]) => Unit is Adj + Ctl,
        target : Qubit[] => Unit is Adj + Ctl,
        sampleCount : Int, shots : Int
    ) : Double {
        CheckSampling(sampleCount, shots);
        return AverageNoisyLabelInfidelity(pattern, angles, prepare, target,
            [EmptyNoise(), size = sampleCount], shots);
    }

    /// Sec. IV B: labels are V_i U|psi_i>, with one fixed V_i per sample.
    operation AverageNoisyLabelInfidelity(
        pattern : Pattern, angles : Double[],
        prepare : (Int, Qubit[]) => Unit is Adj + Ctl,
        target : Qubit[] => Unit is Adj + Ctl,
        labelNoise : PauliProgram[], shots : Int
    ) : Double {
        CheckSampling(Length(labelNoise), shots);
        let width = Length(pattern.Inputs);
        mutable accepted = 0;
        for sample in 0..Length(labelNoise) - 1 {
            for _ in 1..shots {
                use input = Qubit[width];
                use output = Qubit[width];
                prepare(sample, input);
                let _ = Execute(pattern, angles, input, output);
                Adjoint ApplyNoiseProgram(labelNoise[sample], output);
                Adjoint target(output);
                Adjoint prepare(sample, output);
                if AllZeroAndReset(output) { set accepted += 1; }
            }
        }
        return 1.0 - IntAsDouble(accepted) / IntAsDouble(Length(labelNoise) * shots);
    }

    operation AverageNoisyResourceInfidelity(
        pattern : Pattern, angles : Double[],
        prepare : (Int, Qubit[]) => Unit is Adj + Ctl,
        target : Qubit[] => Unit is Adj + Ctl,
        sampleCount : Int, shots : Int, probability : Double
    ) : Double {
        CheckSampling(sampleCount, shots);
        CheckProbability(probability);
        let width = Length(pattern.Inputs);
        mutable accepted = 0;
        for sample in 0..sampleCount - 1 {
            for _ in 1..shots {
                use input = Qubit[width];
                use output = Qubit[width];
                prepare(sample, input);
                let _ = ExecuteWithNoise(pattern, angles, input, output, ApplyDepolarizingResource(probability, _));
                Adjoint target(output);
                Adjoint prepare(sample, output);
                if AllZeroAndReset(output) { set accepted += 1; }
            }
        }
        return 1.0 - IntAsDouble(accepted) / IntAsDouble(sampleCount * shots);
    }

    /// Angles supplied per graph vertex, converted to the executor's order slots.
    function VertexAngles(pattern : Pattern, byVertex : Double[]) : Double[] {
        if Length(byVertex) != pattern.VertexCount { fail "One angle per vertex is required."; }
        CheckFiniteArray(byVertex, "Measurement angle");
        return Mapped(vertex -> byVertex[vertex], pattern.Order);
    }

    /// S1 (family=0) and S2 (family=1), Sec. IV C.
    operation PrepareStateFamily(family : Int, theta : Double, phi : Double, register : Qubit[]) : Unit is Adj + Ctl {
        if Length(register) != 2 or (family != 0 and family != 1) { fail "Expected S1/S2 and two qubits."; }
        CheckFinite(theta, "State-family theta");
        CheckFinite(phi, "State-family phi");
        Ry(2.0 * theta, register[0]);
        Rz(phi, register[0]);
        CNOT(register[0], register[1]);
        if family == 1 { H(register[0]); H(register[1]); }
    }

    function TiedClassifierAngles(pattern : Pattern, columnAngles : Double[]) : Double[] {
        if pattern.VertexCount != 10 or Length(columnAngles) != 4 {
            fail "The paper classifier has two five-vertex wires and four shared angles.";
        }
        CheckFiniteArray(columnAngles, "Classifier angle");
        return Mapped(vertex -> columnAngles[vertex % 5], pattern.Order);
    }

    /// Joint outcome probabilities p00 and p11, not independent local means.
    operation ClassifierProbabilities(
        pattern : Pattern, angles : Double[], prepare : Qubit[] => Unit,
        shots : Int
    ) : (Double, Double) {
        CheckSampling(1, shots);
        if Length(pattern.Inputs) != 2 { fail "The paper classifier needs two input qubits."; }
        mutable n00 = 0;
        mutable n11 = 0;
        for _ in 1..shots {
            use input = Qubit[2];
            use output = Qubit[2];
            prepare(input);
            let _ = Execute(pattern, angles, input, output);
            let r0 = MResetZ(output[0]);
            let r1 = MResetZ(output[1]);
            if r0 == Zero and r1 == Zero { set n00 += 1; }
            if r0 == One and r1 == One { set n11 += 1; }
        }
        return (IntAsDouble(n00) / IntAsDouble(shots), IntAsDouble(n11) / IntAsDouble(shots));
    }

    function PolynomialHead(beta : Double[], probabilities : (Double, Double)) : Double {
        if Length(beta) != 6 { fail "Degree-two head requires six coefficients."; }
        CheckFiniteArray(beta, "Classifier coefficient");
        CheckJointProbabilities(probabilities);
        let (plus, minus) = probabilities;
        return beta[0] + beta[1] * plus + beta[2] * minus
            + beta[3] * plus * plus + beta[4] * plus * minus + beta[5] * minus * minus;
    }

    /// F_Q=4 Var[(Z0+Z1)/2], hence SQL=2 and HL=4. The paper's
    /// later shorthand h=Z is inconsistent with its preceding normalization.
    function QfiFromProbabilities(probabilities : (Double, Double)) : Double {
        CheckJointProbabilities(probabilities);
        let (plus, minus) = probabilities;
        return 4.0 * (plus + minus - (plus - minus) * (plus - minus));
    }

    function ClassifierMarginLoss(predictions : Double[], labels : Int[], epsilon : Double) : Double {
        CheckFinite(epsilon, "Classifier margin");
        CheckFiniteArray(predictions, "Classifier prediction");
        if Length(predictions) == 0 or Length(predictions) != Length(labels) or epsilon <= 0.0 {
            fail "Classifier needs equally sized nonempty arrays and positive margin.";
        }
        mutable loss = 0.0;
        for j in 0..Length(labels) - 1 {
            if labels[j] != 0 and labels[j] != 1 { fail "Classifier labels must be 0 or 1."; }
            let violation = labels[j] == 1 ? 2.0 + epsilon - predictions[j] | predictions[j] - 2.0 + epsilon;
            if violation > 0.0 { set loss += violation; }
        }
        return loss / IntAsDouble(Length(labels));
    }

    operation QuantumClassifierLoss(
        pattern : Pattern, columnAngles : Double[], beta : Double[],
        prepare : (Int, Qubit[]) => Unit, labels : Int[], shots : Int, epsilon : Double
    ) : Double {
        let angles = TiedClassifierAngles(pattern, columnAngles);
        mutable predictions = [];
        for sample in 0..Length(labels) - 1 {
            let probabilities = ClassifierProbabilities(pattern, angles, prepare(sample, _), shots);
            set predictions += [PolynomialHead(beta, probabilities)];
        }
        return ClassifierMarginLoss(predictions, labels, epsilon);
    }

    /// Three trainable MuTA stages implement the same instrument-learning task as
    /// Fig. 6, with a staged layout rather than that figure's 23-vertex numbering.
    /// Resource and Bell-analysis each use a (2,0) layer; conditional corrections
    /// use a disconnected one-wire layer. Outcomes select angles beyond ordinary
    /// flow corrections. Only the two instrument outcomes are returned.
    operation TeleportWithMuTA(
        resourceAngles : Double[], bellAngles : Double[], conditionalAngles : Double[],
        message : Qubit, output : Qubit
    ) : Result[] {
        if Length(conditionalAngles) != 4 { fail "Teleportation needs four conditional angles."; }
        CheckFiniteArray(conditionalAngles, "Conditional angle");
        let pairPattern = Layer(2, 0, [1]);
        use initialPair = Qubit[2];
        use pair = Qubit[2];
        let _ = Execute(pairPattern, resourceAngles, initialPair, pair);
        use measured = Qubit[2];
        let _ = Execute(pairPattern, bellAngles, [message, pair[0]], measured);
        let first = MResetZ(measured[0]);
        let second = MResetZ(measured[1]);
        // Our Bell basis has labels Phi+, Psi-, Psi+, Phi- for 00,01,10,11.
        let zControl = second == One;
        let xControl = first != second;
        let branchAngles = [zControl ? conditionalAngles[0] | 0.0,
            xControl ? conditionalAngles[1] | 0.0,
            zControl ? conditionalAngles[2] | 0.0,
            xControl ? conditionalAngles[3] | 0.0];
        let _ = Execute(Layer(1, -1, []), branchAngles, [pair[1]], [output]);
        return [first, second];
    }

    function ExactTeleportationParameters() : Double[] {
        let pattern = Layer(2, 0, [1]);
        mutable resource = [0.0, size = 10];
        set resource w/= 6 <- PI() / 2.0;
        set resource w/= 7 <- PI() / 2.0;
        mutable bell = [0.0, size = 10];
        set bell w/= 5 <- -PI() / 2.0;
        set bell w/= 6 <- -PI() / 2.0;
        return VertexAngles(pattern, resource) + VertexAngles(pattern, bell) + [-PI(), -PI(), 0.0, 0.0];
    }

    operation TeleportationInfidelity(
        parameters : Double[], prepare : (Int, Qubit[]) => Unit is Adj + Ctl,
        sampleCount : Int, shots : Int
    ) : Double {
        CheckSampling(sampleCount, shots);
        if Length(parameters) != 20 { fail "Teleportation has 8+8+4 parameters."; }
        mutable accepted = 0;
        for sample in 0..sampleCount - 1 {
            for _ in 1..shots {
                use message = Qubit[1];
                use output = Qubit[1];
                prepare(sample, message);
                let _ = TeleportWithMuTA(parameters[0..7], parameters[8..15], parameters[16..19], message[0], output[0]);
                Adjoint prepare(sample, output);
                if AllZeroAndReset(output) { set accepted += 1; }
            }
        }
        return 1.0 - IntAsDouble(accepted) / IntAsDouble(sampleCount * shots);
    }

    /// Eq. (5), executable unitary reference; positive Exp angle gives exp(+i theta XX/2).
    operation PrepareKernel(data : Double[], register : Qubit[]) : Unit is Adj + Ctl {
        CheckKernelPoint(data);
        if Length(register) != 2 { fail "The paper feature map needs two qubits."; }
        Rz(-data[0], register[0]);
        Rz(-data[1], register[1]);
        Exp([PauliX, PauliX], Cos(data[0]) * Cos(data[1]) / 2.0, register);
        Rz(-data[0], register[0]);
        Rz(-data[1], register[1]);
    }

    function KernelAngles(data : Double[]) : Double[] {
        CheckKernelPoint(data);
        mutable byVertex = [0.0, size = 10];
        set byVertex w/= 0 <- data[0];
        set byVertex w/= 5 <- data[1];
        set byVertex w/= 6 <- Cos(data[0]) * Cos(data[1]);
        set byVertex w/= 2 <- data[0];
        set byVertex w/= 7 <- data[1];
        return VertexAngles(Layer(2, 0, [1]), byVertex);
    }

    operation PrepareMuTAKernel(data : Double[], output : Qubit[]) : Unit {
        if Length(output) != 2 { fail "The paper kernel needs two qubits."; }
        use input = Qubit[2];
        let _ = Execute(Layer(2, 0, [1]), KernelAngles(data), input, output);
    }

    operation FidelityKernel(
        width : Int, left : Qubit[] => Unit is Adj + Ctl,
        right : Qubit[] => Unit is Adj + Ctl, shots : Int
    ) : Double {
        CheckSampling(1, shots);
        if width <= 0 { fail "Kernel width must be positive."; }
        mutable accepted = 0;
        for _ in 1..shots {
            use register = Qubit[width];
            left(register);
            Adjoint right(register);
            if AllZeroAndReset(register) { set accepted += 1; }
        }
        return IntAsDouble(accepted) / IntAsDouble(shots);
    }

    /// SWAP estimator 2 Pr(ancilla=0)-1. A finite-shot estimate may be negative;
    /// retain it rather than biasing the estimate by clipping every matrix entry.
    operation SwapKernel(width : Int, left : Qubit[] => Unit, right : Qubit[] => Unit, shots : Int) : Double {
        CheckSampling(1, shots);
        if width <= 0 { fail "Kernel width must be positive."; }
        mutable accepted = 0;
        for _ in 1..shots {
            use a = Qubit[width];
            use b = Qubit[width];
            use ancilla = Qubit();
            left(a);
            right(b);
            H(ancilla);
            for q in 0..width - 1 { Controlled SWAP([ancilla], (a[q], b[q])); }
            H(ancilla);
            if MResetZ(ancilla) == Zero { set accepted += 1; }
            ResetAll(a);
            ResetAll(b);
        }
        return 2.0 * IntAsDouble(accepted) / IntAsDouble(shots) - 1.0;
    }

    operation MuTAKernel(left : Double[], right : Double[], shots : Int) : Double {
        return SwapKernel(2, PrepareMuTAKernel(left, _), PrepareMuTAKernel(right, _), shots);
    }

    /// Independent closed-form check of Eq. (5), useful for noiseless small-data
    /// studies. KernelMatrix below measures the quantum implementation instead.
    function ExactKernel(left : Double[], right : Double[]) : Double {
        CheckKernelPoint(left);
        CheckKernelPoint(right);
        let theta = Cos(left[0]) * Cos(left[1]) / 2.0;
        let thetaPrime = Cos(right[0]) * Cos(right[1]) / 2.0;
        let a = Cos(theta) * Cos(thetaPrime);
        let b = Sin(theta) * Sin(thetaPrime);
        return a * a + b * b + 2.0 * a * b * Cos(left[0] + left[1] - right[0] - right[1]);
    }

    operation KernelMatrix(data : Double[][], shots : Int) : Double[][] {
        let count = Length(data);
        CheckSampling(count, shots);
        for point in data { CheckKernelPoint(point); }
        mutable matrix = [[0.0, size = count], size = count];
        for i in 0..count - 1 {
            set matrix w/= i <- (matrix[i] w/ i <- 1.0);
            for j in i + 1..count - 1 {
                let value = MuTAKernel(data[i], data[j], shots);
                set matrix w/= i <- (matrix[i] w/ j <- value);
                set matrix w/= j <- (matrix[j] w/ i <- value);
            }
        }
        return matrix;
    }

    function SvmScore(model : SvmModel, similarities : Double[]) : Double {
        if Length(similarities) != Length(model.Alpha) or Length(model.Labels) != Length(model.Alpha) {
            fail "SVM kernel vector or label array has wrong length.";
        }
        CheckFiniteArray(similarities, "SVM kernel value");
        CheckFiniteArray(model.Alpha, "SVM dual weight");
        CheckFinite(model.Bias, "SVM bias");
        mutable score = model.Bias;
        for i in 0..Length(similarities) - 1 {
            set score += model.Alpha[i] * IntAsDouble(model.Labels[i]) * similarities[i];
        }
        return score;
    }

    function PredictSvm(model : SvmModel, similarities : Double[]) : Int {
        return SvmScore(model, similarities) >= 0.0 ? 1 | -1;
    }

    function Clamp(value : Double, lower : Double, upper : Double) : Double {
        return value < lower ? lower | value > upper ? upper | value;
    }

    /// Classical soft-margin kernel SVM using simplified SMO, wholly in Q#.
    /// The paper specifies the SVM decision rule but not an optimizer; SMO is
    /// this implementation's choice. Input labels are -1/+1. A finite-shot Gram
    /// matrix need not be PSD, and maxIterations is always enforced.
    function TrainSvm(
        kernel : Double[][], labels : Int[], penalty : Double,
        tolerance : Double, maxStablePasses : Int, maxIterations : Int
    ) : SvmModel {
        let count = Length(labels);
        CheckFinite(penalty, "SVM penalty");
        CheckFinite(tolerance, "SVM tolerance");
        if count < 2 or Length(kernel) != count or penalty <= 0.0 or tolerance <= 0.0
            or maxStablePasses <= 0 or maxIterations <= 0 { fail "Invalid SVM training configuration."; }
        for i in 0..count - 1 {
            if Length(kernel[i]) != count or (labels[i] != -1 and labels[i] != 1) {
                fail "SVM needs a square Gram matrix and -1/+1 labels.";
            }
            CheckFiniteArray(kernel[i], "SVM Gram matrix entry");
        }
        if not Any(label -> label == -1, labels) or not Any(label -> label == 1, labels) {
            fail "Binary SVM training requires examples from both classes.";
        }
        for i in 0..count - 1 {
            for j in 0..i - 1 {
                if AbsD(kernel[i][j] - kernel[j][i]) > 1e-8 { fail "SVM Gram matrix must be symmetric."; }
            }
        }
        mutable alpha = [0.0, size = count];
        mutable bias = 0.0;
        mutable stable = 0;
        mutable iterations = 0;
        while stable < maxStablePasses and iterations < maxIterations {
            mutable changes = 0;
            for i in 0..count - 1 {
                mutable errorI = bias - IntAsDouble(labels[i]);
                for k in 0..count - 1 {
                    set errorI += alpha[k] * IntAsDouble(labels[k]) * kernel[k][i];
                }
                let yi = IntAsDouble(labels[i]);
                if (yi * errorI < -tolerance and alpha[i] < penalty)
                    or (yi * errorI > tolerance and alpha[i] > 0.0) {
                    // Try all partners: merely taking maximal |Ei-Ej| can stall
                    // on an infeasible pair even when a useful update exists.
                    mutable updated = false;
                    for offset in 1..count - 1 {
                        let j = (i + offset) % count;
                        if not updated {
                            let yj = IntAsDouble(labels[j]);
                            mutable errorJ = bias - yj;
                            for k in 0..count - 1 {
                                set errorJ += alpha[k] * IntAsDouble(labels[k]) * kernel[k][j];
                            }
                            let oldI = alpha[i];
                            let oldJ = alpha[j];
                            let lower = labels[i] != labels[j] ? Clamp(oldJ - oldI, 0.0, penalty)
                                | Clamp(oldI + oldJ - penalty, 0.0, penalty);
                            let upper = labels[i] != labels[j] ? Clamp(penalty + oldJ - oldI, 0.0, penalty)
                                | Clamp(oldI + oldJ, 0.0, penalty);
                            let eta = 2.0 * kernel[i][j] - kernel[i][i] - kernel[j][j];
                            if upper > lower {
                                // Identical feature vectors have eta=0. Optimize
                                // at the interval endpoints instead of skipping
                                // them: opposite labels can still saturate C.
                                let lowerDelta = lower - oldJ;
                                let upperDelta = upper - oldJ;
                                let lowerGain = yj * (errorI - errorJ) * lowerDelta + eta * lowerDelta * lowerDelta / 2.0;
                                let upperGain = yj * (errorI - errorJ) * upperDelta + eta * upperDelta * upperDelta / 2.0;
                                let endpoint = upperGain > lowerGain and upperGain > 0.0 ? upper
                                    | lowerGain > 0.0 ? lower | oldJ;
                                let nextJ = eta < -1e-12
                                    ? Clamp(oldJ - yj * (errorI - errorJ) / eta, lower, upper) | endpoint;
                                if AbsD(nextJ - oldJ) > tolerance * 0.01 {
                                    let nextI = oldI + yi * yj * (oldJ - nextJ);
                                    let b1 = bias - errorI - yi * (nextI - oldI) * kernel[i][i]
                                        - yj * (nextJ - oldJ) * kernel[i][j];
                                    let b2 = bias - errorJ - yi * (nextI - oldI) * kernel[i][j]
                                        - yj * (nextJ - oldJ) * kernel[j][j];
                                    set bias = nextI > 0.0 and nextI < penalty ? b1
                                        | nextJ > 0.0 and nextJ < penalty ? b2 | (b1 + b2) / 2.0;
                                    set alpha w/= i <- nextI;
                                    set alpha w/= j <- nextJ;
                                    set changes += 1;
                                    set updated = true;
                                }
                            }
                        }
                    }
                }
            }
            set iterations += 1;
            set stable = changes == 0 ? stable + 1 | 0;
        }
        return new SvmModel { Alpha = alpha, Labels = labels, Bias = bias,
            Iterations = iterations, StablePasses = stable };
    }
}
