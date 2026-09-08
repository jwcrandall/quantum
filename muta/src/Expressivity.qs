namespace MuTA.Expressivity {
    import Std.Math.*;
    import Std.Convert.*;

    function SameWord(a : Pauli[], b : Pauli[]) : Bool {
        if Length(a) != Length(b) { return false; }
        mutable same = true;
        for i in 0..Length(a)-1 { if a[i] != b[i] { set same = false; } }
        return same;
    }

    function HasWord(words : Pauli[][], word : Pauli[]) : Bool {
        mutable found = false;
        for existing in words { if SameWord(existing, word) { set found = true; } }
        return found;
    }

    function IsIdentity(word : Pauli[]) : Bool {
        mutable identity = true;
        for p in word { if p != PauliI { set identity = false; } }
        return identity;
    }

    function Anticommutes(a : Pauli[], b : Pauli[]) : Bool {
        if Length(a) != Length(b) { fail "Pauli words must have equal width."; }
        mutable odd = false;
        for i in 0..Length(a)-1 {
            if a[i] != PauliI and b[i] != PauliI and a[i] != b[i] { set odd = not odd; }
        }
        return odd;
    }

    /// Product modulo its irrelevant nonzero scalar phase.
    function ProductWord(a : Pauli[], b : Pauli[]) : Pauli[] {
        if Length(a) != Length(b) { fail "Pauli words must have equal width."; }
        mutable product = [];
        for i in 0..Length(a)-1 {
            let (p, q) = (a[i], b[i]);
            if p == q { set product += [PauliI]; }
            elif p == PauliI { set product += [q]; }
            elif q == PauliI { set product += [p]; }
            elif p != PauliX and q != PauliX { set product += [PauliX]; }
            elif p != PauliY and q != PauliY { set product += [PauliY]; }
            else { set product += [PauliZ]; }
        }
        return product;
    }

    /// Appendix C: exact Lie closure for individually available nonidentity
    /// Pauli generators iP. Distinct Pauli words are linearly independent.
    /// This is an exponential small-system analysis, not a training algorithm.
    function PauliLieClosure(generators : Pauli[][], maxTerms : Int) : Pauli[][] {
        if Length(generators) == 0 or Length(generators[0]) == 0 { fail "Provide nonempty Pauli generators."; }
        if maxTerms <= 0 { fail "maxTerms must be positive."; }
        let width = Length(generators[0]);
        mutable basis = [];
        for word in generators {
            if Length(word) != width { fail "Inconsistent generator width."; }
            if not IsIdentity(word) and not HasWord(basis, word) {
                if Length(basis) >= maxTerms { fail "Lie closure exceeded maxTerms."; }
                set basis += [word];
            }
        }
        mutable cursor = 0;
        while cursor < Length(basis) {
            // Commutators with all currently known words suffice; newly added
            // words are themselves visited by subsequent outer iterations.
            let currentCount = Length(basis);
            for j in 0..currentCount-1 {
                if Anticommutes(basis[cursor], basis[j]) {
                    let product = ProductWord(basis[cursor], basis[j]);
                    if not HasWord(basis, product) {
                        if Length(basis) >= maxTerms { fail "Lie closure exceeded maxTerms."; }
                        set basis += [product];
                    }
                }
            }
            set cursor += 1;
        }
        return basis;
    }

    /// Eq. (C1), applicable only when the associated Lie algebra is simple and
    /// the paper's ensemble assumptions hold. It does not predict finite-depth
    /// optimizer performance. Caller supplies the independently found dimension.
    function SimpleAlgebraVarianceBound(width : Int, dimension : Int) : Double {
        if width < 0 or dimension <= 0 { fail "Invalid width or algebra dimension."; }
        return 2.0 ^ IntAsDouble(width) / IntAsDouble(dimension);
    }
}
