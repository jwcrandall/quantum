namespace MuTA.QuantumEntry {
    import MuTA.Core.*;
    import Std.Math.*;
    import Std.Measurement.*;

    /// Device-independent quantum workload with static parameters. Compile this
    /// together with Core.qs; the classical optimizers and random noise samplers
    /// are host/simulator programs and are not part of a submitted QIR job.
    operation SampleConnectedLayer() : Result[] {
        let p = Layer(2,0,[1]);
        let angles = AnglesFromVertices(p,[0.0,0.0,0.0,0.0,0.0,0.0,-PI()/2.0,0.0,0.0,0.0]);
        use input = Qubit[2];
        use output = Qubit[2];
        let outcomes = Execute(p,angles,input,output);
        return MResetEachZ(output);
    }
}
