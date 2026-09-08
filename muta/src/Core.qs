namespace MuTA.Core {
    import Std.Arrays.*;
    import Std.Measurement.*;

    /// An open graph with causal flow. Angles/results are indexed by Order,
    /// not by vertex id. Flow has one entry per vertex (-1 on outputs).
    struct Pattern {
        VertexCount : Int,
        Edges : (Int, Int)[],
        Inputs : Int[],
        Outputs : Int[],
        Order : Int[],
        Flow : Int[]
    }

    /// Kind: 0 = Rz, 1 = H, 2 = CZ, 3 = SWAP. Angles use Q# conventions.
    struct CompiledGate { Kind : Int, First : Int, Second : Int, Angle : Double }

    function Has(values : Int[], value : Int) : Bool {
        mutable found = false;
        for v in values { if v == value { set found = true; } }
        return found;
    }

    function IndexOf(values : Int[], value : Int) : Int {
        mutable index = -1;
        for i in 0..Length(values)-1 { if values[i] == value { set index = i; } }
        return index;
    }

    function Neighbors(edges : (Int, Int)[], vertex : Int) : Int[] {
        mutable result = [];
        for (a, b) in edges {
            if a == vertex { set result += [b]; }
            if b == vertex { set result += [a]; }
        }
        return result;
    }

    function ValidateVertices(count : Int, vertices : Int[]) : Unit {
        mutable seen = [];
        for v in vertices {
            if v < 0 or v >= count { fail "Vertex outside graph."; }
            if Has(seen, v) { fail "Duplicate vertex."; }
            set seen += [v];
        }
    }

    /// Returns maximal parallel slices of measured V\\O vertices satisfying
    /// i < f(i) and i < N(f(i))\\{i}; outputs are intentionally omitted.
    function FlowSlices(count : Int, edges : (Int, Int)[], outputs : Int[], flow : Int[]) : Int[][] {
        mutable remaining = [];
        for v in 0..count-1 { if not Has(outputs, v) { set remaining += [v]; } }
        mutable slices = [];
        while Length(remaining) > 0 {
            mutable ready = [];
            for v in remaining {
                mutable blocked = false;
                for u in remaining {
                    if u != v and (flow[u] == v or Has(Neighbors(edges, flow[u]), v)) {
                        set blocked = true;
                    }
                }
                if not blocked { set ready += [v]; }
            }
            if Length(ready) == 0 { fail "Flow has cyclic causal dependencies."; }
            set slices += [ready];
            mutable next = [];
            for v in remaining { if not Has(ready, v) { set next += [v]; } }
            set remaining = next;
        }
        return slices;
    }

    function CreatePattern(count : Int, edges : (Int, Int)[], inputs : Int[], outputs : Int[], flow : Int[]) : Pattern {
        if count <= 0 { fail "A graph must contain a vertex."; }
        ValidateVertices(count, inputs);
        ValidateVertices(count, outputs);
        if Length(flow) != count { fail "Flow length must equal the vertex count."; }
        mutable seenEdges = [];
        for (a, b) in edges {
            if a == b { fail "Self edges are not allowed."; }
            ValidateVertices(count, [a, b]);
            if Has(Neighbors(seenEdges, a), b) { fail "Duplicate undirected edge."; }
            set seenEdges += [(a, b)];
        }
        mutable successors = [];
        for v in 0..count-1 {
            if Has(outputs, v) {
                if flow[v] != -1 { fail "Output vertices must have flow -1."; }
            } else {
                let f = flow[v];
                if f < 0 or f >= count { fail "Invalid flow successor."; }
                if Has(inputs, f) { fail "A flow successor cannot be an input."; }
                if not Has(Neighbors(edges, v), f) { fail "Flow must follow a graph edge."; }
                if Has(successors, f) { fail "Flow must be injective."; }
                set successors += [f];
            }
        }
        mutable order = [];
        for slice in FlowSlices(count, edges, outputs, flow) { set order += slice; }
        return new Pattern { VertexCount=count, Edges=edges, Inputs=inputs, Outputs=outputs, Order=order, Flow=flow };
    }

    /// Sec. III, Fig. 1: row-major vertex 5*row+column. tip=-1 requests
    /// disconnected wires. A triangle adds two edges, not a tip-center edge.
    function Layer(width : Int, tip : Int, targets : Int[]) : Pattern {
        if width <= 0 { fail "Layer width must be positive."; }
        if tip < -1 or tip >= width { fail "Invalid tip wire."; }
        ValidateVertices(width, targets);
        if Has(targets, tip) or (tip == -1 and Length(targets) > 0) { fail "Invalid triangle targets."; }
        mutable edges = [];
        mutable inputs = [];
        mutable outputs = [];
        mutable flow = Repeated(-1, 5*width);
        for row in 0..width-1 {
            set inputs += [5*row];
            set outputs += [5*row+4];
            for col in 0..3 {
                let v = 5*row+col;
                set edges += [(v, v+1)];
                set flow w/= v <- v+1;
            }
        }
        for row in targets { set edges += [(5*tip+1, 5*row), (5*tip+1, 5*row+2)]; }
        return CreatePattern(5*width, edges, inputs, outputs, flow);
    }

    /// Join selected output vertex ids of first to input vertex ids of second.
    /// Unjoined inputs/outputs remain external; second's unjoined vertices get
    /// fresh ids in ascending order. Inputs/outputs preserve each graph's order.
    function Concatenate(first : Pattern, second : Pattern, joins : (Int, Int)[]) : Pattern {
        ValidatePattern(first);
        ValidatePattern(second);
        mutable mapping = Repeated(-1, second.VertexCount);
        mutable usedLeft = [];
        mutable usedRight = [];
        for (left, right) in joins {
            if not Has(first.Outputs, left) or not Has(second.Inputs, right) { fail "Joins must match output and input vertices."; }
            if Has(usedLeft, left) or Has(usedRight, right) { fail "Joins must be one-to-one."; }
            set usedLeft += [left]; set usedRight += [right];
            set mapping w/= right <- left;
        }
        mutable count = first.VertexCount;
        for v in 0..second.VertexCount-1 {
            if mapping[v] == -1 { set mapping w/= v <- count; set count += 1; }
        }
        mutable edges = first.Edges;
        for (a, b) in second.Edges { set edges += [(mapping[a], mapping[b])]; }
        mutable inputs = first.Inputs;
        for v in second.Inputs { if not Has(usedRight, v) { set inputs += [mapping[v]]; } }
        mutable outputs = [];
        for v in first.Outputs { if not Has(usedLeft, v) { set outputs += [v]; } }
        for v in second.Outputs { set outputs += [mapping[v]]; }
        mutable flow = first.Flow + Repeated(-1, count-first.VertexCount);
        for v in second.Order { set flow w/= mapping[v] <- mapping[second.Flow[v]]; }
        return CreatePattern(count, edges, inputs, outputs, flow);
    }

    /// Validates public struct values, including custom causal orders.
    function ValidatePattern(pattern : Pattern) : Unit {
        let canonical = CreatePattern(pattern.VertexCount, pattern.Edges, pattern.Inputs, pattern.Outputs, pattern.Flow);
        ValidateVertices(pattern.VertexCount, pattern.Order);
        if Length(pattern.Order) != Length(canonical.Order) { fail "Order must contain every non-output exactly once."; }
        for v in pattern.Order { if Has(pattern.Outputs, v) { fail "Outputs cannot be measured."; } }
        for i in 0..Length(pattern.Order)-1 {
            let v = pattern.Order[i];
            let future = pattern.Order[i+1...] + pattern.Outputs;
            if not Has(future, pattern.Flow[v]) { fail "Order violates successor dependency."; }
            for w in Neighbors(pattern.Edges, pattern.Flow[v]) {
                if w != v and not Has(future, w) { fail "Order violates correction dependency."; }
            }
        }
    }

    function ValidateAngles(pattern : Pattern, angles : Double[]) : Unit {
        ValidatePattern(pattern);
        if Length(angles) != Length(pattern.Order) { fail "One angle is required per measured vertex."; }
        for angle in angles {
            if angle != angle or angle-angle != 0.0 { fail "Angles must be finite."; }
        }
    }

    /// Convert a vertex-indexed array (including ignored output entries) to the
    /// canonical parameter vector consumed by every executor/trainer.
    function AnglesFromVertices(pattern : Pattern, vertexAngles : Double[]) : Double[] {
        if Length(vertexAngles) != pattern.VertexCount { fail "Incorrect vertex angle count."; }
        return Mapped(v -> vertexAngles[v], pattern.Order);
    }

    function ParameterSlices(pattern : Pattern) : Int[][] {
        ValidatePattern(pattern);
        mutable slices = [];
        for slice in FlowSlices(pattern.VertexCount, pattern.Edges, pattern.Outputs, pattern.Flow) {
            set slices += [Mapped(v -> IndexOf(pattern.Order, v), slice)];
        }
        return slices;
    }

    operation NoNoise(register : Qubit[]) : Unit { }

    /// Measurement-based execution of Eq. (1) and Sec. II flow corrections.
    /// Input and output arrays must be disjoint; output starts in |0...0>.
    /// Input is consumed and reset. Output is corrected, in Outputs order.
    operation Execute(pattern : Pattern, angles : Double[], input : Qubit[], output : Qubit[]) : Result[] {
        return ExecuteWithNoise(pattern, angles, input, output, NoNoise);
    }

    /// The noise callback acts once on the COMPLETE entangled resource state,
    /// including input vertices, before measurements (Appendix D).
    operation ExecuteWithNoise(pattern : Pattern, angles : Double[], input : Qubit[], output : Qubit[], noise : (Qubit[] => Unit)) : Result[] {
        ValidateAngles(pattern, angles);
        if Length(input) != Length(pattern.Inputs) or Length(output) != Length(pattern.Outputs) { fail "Register size does not match graph interface."; }
        for i in 0..Length(input)-1 {
            for j in i+1..Length(input)-1 { if input[i] == input[j] { fail "Input qubits must be distinct."; } }
            for q in output { if input[i] == q { fail "Input and output must be disjoint."; } }
        }
        for i in 0..Length(output)-1 {
            for j in i+1..Length(output)-1 { if output[i] == output[j] { fail "Output qubits must be distinct."; } }
        }
        use resource = Qubit[pattern.VertexCount];
        for i in 0..Length(input)-1 { SWAP(input[i], resource[pattern.Inputs[i]]); }
        for v in 0..pattern.VertexCount-1 { if not Has(pattern.Inputs, v) { H(resource[v]); } }
        for (a, b) in pattern.Edges { CZ(resource[a], resource[b]); }
        noise(resource);
        mutable outcomes = [];
        for k in 0..Length(pattern.Order)-1 {
            let v = pattern.Order[k];
            Rz(-angles[k], resource[v]); H(resource[v]);
            let result = MResetZ(resource[v]);
            set outcomes += [result];
            // Explicit physical Pauli corrections are equivalent to adapting
            // future angles and tracking output Pauli byproducts.
            if result == One {
                X(resource[pattern.Flow[v]]);
                for w in Neighbors(pattern.Edges, pattern.Flow[v]) {
                    if w != v { Z(resource[w]); }
                }
            }
        }
        for i in 0..Length(output)-1 { SWAP(resource[pattern.Outputs[i]], output[i]); }
        ResetAll(resource);
        return outcomes;
    }

    /// Appendix B's general open-graph-to-circuit translation. Requires equal
    /// input/output counts. Live vertex positions follow flow paths. A final
    /// permutation makes output register order identical to Execute.
    function Compile(pattern : Pattern, angles : Double[]) : CompiledGate[] {
        ValidateAngles(pattern, angles);
        if Length(pattern.Inputs) != Length(pattern.Outputs) { fail "Unitary compilation requires equal input/output counts."; }
        mutable live = pattern.Inputs;
        mutable used = Repeated(false, Length(pattern.Edges));
        mutable plan = [];
        for k in 0..Length(pattern.Order)-1 {
            let v = pattern.Order[k];
            let wire = IndexOf(live, v);
            if wire < 0 { fail "Measured vertex is not on a live input flow path."; }
            for e in 0..Length(pattern.Edges)-1 {
                let (a, b) = pattern.Edges[e];
                if not used[e] and (a == v or b == v) {
                    let other = a == v ? b | a;
                    if other != pattern.Flow[v] {
                        let otherWire = IndexOf(live, other);
                        if otherWire < 0 { fail "Junction neighbor must be a live input."; }
                        set plan += [new CompiledGate { Kind=2, First=wire, Second=otherWire, Angle=0.0 }];
                    }
                    set used w/= e <- true;
                }
            }
            set plan += [new CompiledGate { Kind=0, First=wire, Second=-1, Angle=-angles[k] }, new CompiledGate { Kind=1, First=wire, Second=-1, Angle=0.0 }];
            set live w/= wire <- pattern.Flow[v];
        }
        for e in 0..Length(pattern.Edges)-1 {
            if not used[e] {
                let (a, b) = pattern.Edges[e];
                let (i, j) = (IndexOf(live, a), IndexOf(live, b));
                if i < 0 or j < 0 { fail "Remaining edge is not an output edge."; }
                set plan += [new CompiledGate { Kind=2, First=i, Second=j, Angle=0.0 }];
            }
        }
        for i in 0..Length(live)-1 {
            let j = IndexOf(live, pattern.Outputs[i]);
            if j < 0 { fail "Output is not on a flow path."; }
            if i != j {
                set plan += [new CompiledGate { Kind=3, First=i, Second=j, Angle=0.0 }];
                let saved = live[i];
                set live w/= i <- live[j]; set live w/= j <- saved;
            }
        }
        return plan;
    }

    operation ApplyCompiled(pattern : Pattern, angles : Double[], register : Qubit[]) : Unit is Adj + Ctl {
        if Length(register) != Length(pattern.Inputs) { fail "Incorrect compiled register width."; }
        let plan = Compile(pattern, angles);
        for gate in plan {
            if gate.Kind == 0 { Rz(gate.Angle, register[gate.First]); }
            elif gate.Kind == 1 { H(register[gate.First]); }
            elif gate.Kind == 2 { CZ(register[gate.First], register[gate.Second]); }
            else { SWAP(register[gate.First], register[gate.Second]); }
        }
    }
}
