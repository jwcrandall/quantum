namespace MuTA.CoreTests {
    import MuTA.Core.*;
    import MuTA.Expressivity.*;
    import Std.Arrays.*;
    import Std.Diagnostics.*;
    import Std.Convert.*;

    operation AssertAllZero(qs : Qubit[]) : Unit {
        if not CheckAllZero(qs) { fail "Channel comparison left nonzero amplitudes."; }
    }

    /// Independent Eq. (B2), not implemented via Compile or graph traversal.
    operation ReferenceLayer(width : Int, tip : Int, targets : Int[], vertexAngles : Double[], qs : Qubit[]) : Unit is Adj + Ctl {
        for row in 0..width-1 { Rz(-vertexAngles[5*row], qs[row]); }
        for row in 0..width-1 {
            if Has(targets, row) {
                CNOT(qs[row], qs[tip]);
                Rx(-vertexAngles[5*row+1], qs[row]);
                CNOT(qs[row], qs[tip]);
            } else { Rx(-vertexAngles[5*row+1], qs[row]); }
        }
        for row in 0..width-1 {
            Rz(-vertexAngles[5*row+2], qs[row]);
            Rx(-vertexAngles[5*row+3], qs[row]);
        }
    }

    operation EntangleReference(reference : Qubit[], data : Qubit[]) : Unit is Adj + Ctl {
        for i in 0..Length(data)-1 { H(reference[i]); CNOT(reference[i], data[i]); }
    }

    /// Choi-state comparison checks the entire channel, including relative
    /// phases and every input basis, on sampled MBQC measurement branches.
    operation LayerEquivalence(width : Int, tip : Int, targets : Int[], phase : Double) : Unit {
        let pattern = Layer(width, tip, targets);
        mutable vertexAngles = [];
        for i in 0..5*width-1 { set vertexAngles += [0.13 * IntAsDouble(i+1) + phase]; }
        let angles = AnglesFromVertices(pattern, vertexAngles);
        use (reference, input, output) = (Qubit[width], Qubit[width], Qubit[width]);
        EntangleReference(reference, input);
        let outcomes = Execute(pattern, angles, input, output);
        if Length(outcomes) != 4*width { fail "Wrong measurement count."; }
        Adjoint ReferenceLayer(width, tip, targets, vertexAngles, output);
        Adjoint EntangleReference(reference, output);
        AssertAllZero(reference + input + output);
        EntangleReference(reference, input);
        ApplyCompiled(pattern, angles, input);
        Adjoint ReferenceLayer(width, tip, targets, vertexAngles, input);
        Adjoint EntangleReference(reference, input);
        AssertAllZero(reference + input + output);
    }

    operation GeneralGraphEquivalence(pattern : Pattern, angles : Double[]) : Unit {
        let width = Length(pattern.Inputs);
        use (reference, input, output) = (Qubit[width], Qubit[width], Qubit[width]);
        EntangleReference(reference, input);
        let outcomes = Execute(pattern, angles, input, output);
        Adjoint ApplyCompiled(pattern, angles, output);
        Adjoint EntangleReference(reference, output);
        AssertAllZero(reference + input + output);
    }

    operation TestLayers() : Unit {
        for phase in [0.0, 0.47, -1.2] {
            LayerEquivalence(1, -1, [], phase);
            LayerEquivalence(2, 0, [1], phase);
            LayerEquivalence(2, 1, [0], phase);
        }
        LayerEquivalence(3, 1, [0], 0.2);
        LayerEquivalence(3, 1, [0,2], -0.31);
    }

    operation TestConcatenation() : Unit {
        let p = Concatenate(Layer(1,-1,[]), Layer(1,-1,[]), [(4,0)]);
        if p.VertexCount != 9 or Length(p.Order) != 8 { fail "Layer concatenation counts are incorrect."; }
        GeneralGraphEquivalence(p, [0.1,0.7,-0.2,0.4,-0.8,0.3,1.2,-0.1]);
        // Width-changing concatenation retains an unjoined input and reorders outputs.
        let mixed = Concatenate(Layer(1,-1,[]), Layer(2,1,[0]), [(4,5)]);
        if Length(mixed.Inputs) != 2 or Length(mixed.Outputs) != 2 { fail "Partial-join interface is incorrect."; }
        GeneralGraphEquivalence(mixed, Repeated(0.23, Length(mixed.Order)));
        let left = Layer(2,0,[1]);
        let right = Layer(2,1,[0]);
        let reordered = Concatenate(left,right,[(4,5),(9,0)]);
        // Choi comparison verifies that output permutation is included.
        GeneralGraphEquivalence(reordered, Repeated(0.12, Length(reordered.Order)));
    }

    operation TestBoundaryGraphs() : Unit {
        // No measured vertices, overlapping I and O, final output CZ, permutation.
        let outputsOnly = CreatePattern(2,[(0,1)],[0,1],[1,0],[-1,-1]);
        GeneralGraphEquivalence(outputsOnly,[]);
        // One isolated passthrough qubit plus a measured wire.
        let passthrough = CreatePattern(3,[(0,1)],[0,2],[2,1],[1,-1,-1]);
        GeneralGraphEquivalence(passthrough,[0.731]);
        // Unequal input/output interface is a valid isometry for Execute.
        let isometry = CreatePattern(2,[(0,1)],[0],[0,1],[-1,-1]);
        use (input, output) = (Qubit[1],Qubit[2]);
        H(input[0]);
        let outcomes = Execute(isometry,[],input,output);
        CZ(output[0],output[1]); H(output[0]); H(output[1]);
        AssertAllZero(input+output);
    }

    operation TestCliffordControl() : Unit {
        let p = Layer(1,-1,[]);
        let angles = [0.1,0.2,0.3,0.4];
        use (control, qs) = (Qubit(),Qubit[1]);
        H(control);
        Controlled ApplyCompiled([control],(p,angles,qs));
        Controlled Adjoint ApplyCompiled([control],(p,angles,qs));
        H(control);
        AssertAllZero([control]+qs);
    }

    function TestExpressivity() : Unit {
        let local = PauliLieClosure([[PauliX,PauliI],[PauliZ,PauliI],[PauliI,PauliX],[PauliI,PauliZ]],16);
        if Length(local) != 6 { fail "Disconnected two-qubit algebra should have dimension 6."; }
        let full = PauliLieClosure([[PauliX,PauliI],[PauliZ,PauliI],[PauliI,PauliX],[PauliI,PauliZ],[PauliX,PauliX]],16);
        if Length(full) != 15 { fail "Universal two-qubit algebra should be su(4)."; }
        if Anticommutes([PauliX,PauliX],[PauliZ,PauliZ]) { fail "Two local anticommutations should commute globally."; }
    }

    operation RunAll() : Unit {
        TestLayers(); TestConcatenation(); TestBoundaryGraphs(); TestCliffordControl(); TestExpressivity();
    }
}
