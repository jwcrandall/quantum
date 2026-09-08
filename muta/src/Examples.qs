namespace MuTA.Examples {
    import MuTA.Core.*;
    import MuTA.Applications.*;
    import MuTA.Training.*;
    import Std.Math.*;
    import Std.Measurement.*;
    import Std.Convert.*;

    function AdamDefaults(steps : Int, rate : Double, parameterShift : Bool) : AdamConfig {
        return new AdamConfig { Steps=steps, LearningRate=rate, Beta1=0.9, Beta2=0.999,
            Epsilon=1e-8, Threshold=0.005, UseParameterShift=parameterShift,
            DifferenceStep=0.15, MaxEvaluations=1000 };
    }

    function DqnDefaults() : DqnConfig {
        return new DqnConfig { Episodes=30, HiddenSize=12, LearningRate=0.03,
            Gamma=0.95, EpsilonStart=1.0, EpsilonMin=0.1, EpsilonDecay=0.95,
            ReplayCapacity=80, BatchSize=8, UpdatesPerEpisode=4, TargetPeriod=5,
            Threshold=-1.0, MaxEvaluations=40 };
    }

    /// Pure-state preparations spanning all Bloch directions, for held-out checks.
    operation ProbeState(sample : Int, register : Qubit[]) : Unit is Adj + Ctl {
        if sample % 6 == 1 { X(register[0]); }
        elif sample % 6 == 2 { H(register[0]); }
        elif sample % 6 == 3 { X(register[0]); H(register[0]); }
        elif sample % 6 == 4 { H(register[0]); S(register[0]); }
        elif sample % 6 == 5 { H(register[0]); Adjoint S(register[0]); }
    }

    /// Sec. IV A: train a full four-angle one-wire MuTA on fixed Haar samples.
    /// Return independent held-out losses before/after and training evaluations.
    operation GateLearning() : (Double, Double, Int) {
        let p = Layer(1,-1,[]);
        let train = SampleHaarDataset(1,4);
        let target = ApplySingleQubitTarget([0.47,-0.31,0.83],_);
        let objective = AverageGateInfidelity(p,_,PrepareDatasetState(train,_,_),target,4,32);
        let result = TrainAdam(objective,[0.0,0.0,0.0,0.0],AdamDefaults(25,0.18,true));
        let before = AverageGateInfidelity(p,[0.0,0.0,0.0,0.0],ProbeState,target,6,128);
        let after = AverageGateInfidelity(p,result.Parameters,ProbeState,target,6,128);
        return (before,after,result.Evaluations);
    }

    operation PhaseTarget(register : Qubit[]) : Unit is Adj + Ctl { Rz(-PI()/4.0,register[0]); }

    /// A constrained one-parameter pattern keeps the discrete examples short;
    /// the same callbacks/trainers accept the entire measured-angle vector.
    operation DiscreteQuantumLoss(theta : Double[]) : Double {
        if Length(theta) != 1 { fail "Example expects one free angle."; }
        return AverageGateInfidelity(Layer(1,-1,[]),[theta[0],0.0,0.0,0.0],ProbeState,PhaseTarget,6,32);
    }

    operation GreedyLearning() : (Double[], Double, Int) {
        let result = GreedyOptimize(DiscreteQuantumLoss,[0.0],[[0]],
            [0.0,PI()/4.0,PI()/2.0],0.0,0,1,0.001,20);
        return (result.Parameters,DiscreteQuantumLoss(result.Parameters),result.Evaluations);
    }

    operation DqnLearning() : (Double[], Double, Int) {
        let result = TrainDqn(DiscreteQuantumLoss,[0.0],[[0]],
            [0.0,PI()/4.0,PI()/2.0],DqnDefaults());
        return (result.Parameters,DiscreteQuantumLoss(result.Parameters),result.Evaluations);
    }

    operation ClassificationState(sample : Int, qs : Qubit[]) : Unit is Adj + Ctl {
        let theta = [0.0,PI()/4.0,PI()/2.0,PI()/8.0,0.12,0.68][sample % 6];
        let phi = [0.0,0.2,1.1,0.0,0.7,1.0][sample % 6];
        PrepareStateFamily(0,theta,phi,qs);
    }

    operation JointClassifierObjective(theta : Double[]) : Double {
        if Length(theta) != 10 { fail "Classifier has four tied angles and six polynomial coefficients."; }
        return QuantumClassifierLoss(Layer(2,-1,[]),theta[0..3],theta[4..9],
            ClassificationState,[0,1,0,0,0,1],128,0.5);
    }

    /// Sec. IV C: jointly update tied basis angles and degree-two head. Finite
    /// differences are used because the margin/polynomial objective is nonlinear.
    /// This tiny diagnostic dataset is not a reproduction of Fig. 5 accuracy.
    operation ClassifierLearning() : (Double, Double, Int) {
        let initial = [0.1,-0.1,0.1,0.0, 0.0,1.0,1.0,0.0,0.0,0.0];
        let result = TrainAdam(JointClassifierObjective,initial,AdamDefaults(8,0.08,false));
        return (result.History[0],JointClassifierObjective(result.Parameters),result.Evaluations);
    }

    /// Fig. 6 task: train the four outcome-controlled correction angles while
    /// the known Bell preparation/analysis is held fixed as an inductive bias.
    /// TeleportationInfidelity also accepts all twenty angles for full training.
    operation TeleportCorrectionLoss(angles : Double[]) : Double {
        let exact = ExactTeleportationParameters();
        return TeleportationInfidelity(exact[0..15]+angles,ProbeState,6,16);
    }

    operation TeleportationLearning() : (Double, Double, Int) {
        let result = TrainAdam(TeleportCorrectionLoss,[-2.2,-2.2,0.0,0.0],AdamDefaults(10,0.2,true));
        return (result.History[0],TeleportCorrectionLoss(result.Parameters),result.Evaluations);
    }

    /// Sec. IV E: shot-estimated MuTA SWAP kernels followed by a Q# SVM.
    operation KernelClassification() : (Int[], Double[][]) {
        let data = [[0.0,0.0],[0.1,0.1],[1.4,1.4],[1.5,1.5]];
        let matrix = KernelMatrix(data,512);
        let model = TrainSvm(matrix,[1,1,-1,-1],10.0,1e-4,5,100);
        mutable predictions = [];
        for row in matrix { set predictions += [PredictSvm(model,row)]; }
        return (predictions,matrix);
    }

    /// Sec. IV B/App. D: fixed sampled noisy labels vs per-shot resource noise.
    /// Returns clean, bit-flip-label, Brownian-label, and resource infidelities.
    operation NoiseComparison() : Double[] {
        let p = Layer(1,-1,[]);
        let dataset = SampleHaarDataset(1,4);
        let prepare = PrepareDatasetState(dataset,_,_);
        let target = ApplyCompiled(p,[0.0,0.0,0.0,0.0],_);
        mutable flips = [];
        mutable brownian = [];
        for _ in 1..4 {
            set flips += [SampleBitFlipProgram(1,0.5)];
            set brownian += [SampleBrownianProgram(1,3,0.2,2)];
        }
        return [AverageGateInfidelity(p,[0.0,0.0,0.0,0.0],prepare,target,4,64),
            AverageNoisyLabelInfidelity(p,[0.0,0.0,0.0,0.0],prepare,target,flips,64),
            AverageNoisyLabelInfidelity(p,[0.0,0.0,0.0,0.0],prepare,target,brownian,64),
            AverageNoisyResourceInfidelity(p,[0.0,0.0,0.0,0.0],prepare,target,4,64,0.1)];
    }

    /// Train separate models on each noisy objective, then evaluate all three
    /// on clean held-out probes. Fixed label samples are reused during fitting.
    /// Return initial clean loss, then bit-flip, Brownian, and resource models.
    operation NoisyGateLearning() : Double[] {
        let p = Layer(1,-1,[]);
        let dataset = SampleHaarDataset(1,6);
        let prepare = PrepareDatasetState(dataset,_,_);
        let initial = [0.3,0.1,-0.2,0.1];
        mutable flips = [];
        mutable brownian = [];
        for _ in 1..6 {
            set flips += [SampleBitFlipProgram(1,0.1)];
            set brownian += [SampleBrownianProgram(1,2,0.1,2)];
        }
        let flipLoss = AverageNoisyLabelInfidelity(p,_,prepare,PhaseTarget,flips,32);
        let brownianLoss = AverageNoisyLabelInfidelity(p,_,prepare,PhaseTarget,brownian,32);
        let resourceLoss = AverageNoisyResourceInfidelity(p,_,prepare,PhaseTarget,6,32,0.03);
        let config = AdamDefaults(8,0.15,true);
        let a = TrainAdam(flipLoss,initial,config);
        let b = TrainAdam(brownianLoss,initial,config);
        let c = TrainAdam(resourceLoss,initial,config);
        return [AverageGateInfidelity(p,initial,ProbeState,PhaseTarget,6,128),
            AverageGateInfidelity(p,a.Parameters,ProbeState,PhaseTarget,6,128),
            AverageGateInfidelity(p,b.Parameters,ProbeState,PhaseTarget,6,128),
            AverageGateInfidelity(p,c.Parameters,ProbeState,PhaseTarget,6,128)];
    }

    /// Small quantum-only entry used for Adaptive_RI QIR compilation checks.
    operation SampleConnectedLayer() : Result[] {
        return MuTA.QuantumEntry.SampleConnectedLayer();
    }
}
