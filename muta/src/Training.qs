namespace MuTA.Training {
    import Std.Convert.*;
    import Std.Math.*;
    import Std.Random.*;

    /// All optimizers return the best evaluated pattern, not the final explored
    /// pattern. History contains its loss after each objective evaluation.
    struct TrainingResult {
        Parameters : Double[],
        Loss : Double,
        History : Double[],
        Success : Bool,
        Evaluations : Int
    }

    struct AdamConfig {
        Steps : Int,
        LearningRate : Double,
        Beta1 : Double,
        Beta2 : Double,
        Epsilon : Double,
        Threshold : Double,
        UseParameterShift : Bool,
        DifferenceStep : Double,
        MaxEvaluations : Int
    }

    struct DqnConfig {
        Episodes : Int,
        HiddenSize : Int,
        LearningRate : Double,
        Gamma : Double,
        EpsilonStart : Double,
        EpsilonMin : Double,
        EpsilonDecay : Double,
        ReplayCapacity : Int,
        BatchSize : Int,
        UpdatesPerEpisode : Int,
        TargetPeriod : Int,
        Threshold : Double,
        MaxEvaluations : Int
    }

    /// Dense one-hidden-layer ReLU network; weights are row-major.
    struct QNetwork {
        InputSize : Int,
        HiddenSize : Int,
        OutputSize : Int,
        InputWeights : Double[],
        HiddenBias : Double[],
        OutputWeights : Double[],
        OutputBias : Double[]
    }

    struct ReplayTransition {
        State : Double[],
        Action : Int,
        Reward : Double,
        NextState : Double[],
        Terminal : Bool
    }

    function IsFiniteTrainingValue(value : Double) : Bool {
        return value == value and AbsD(value) <= 1.7976931348623157e308;
    }

    function RequireFiniteTrainingValue(value : Double) : Unit {
        if not IsFiniteTrainingValue(value) {
            fail "Training requires finite parameters, hyperparameters, and losses.";
        }
    }

    function RequireProbability(value : Double) : Unit {
        RequireFiniteTrainingValue(value);
        if value < 0.0 or value > 1.0 { fail "Expected probability in [0, 1]."; }
    }

    function RequireFiniteParameters(theta : Double[]) : Unit {
        for value in theta { RequireFiniteTrainingValue(value); }
    }

    /// Slices partition CANONICAL PARAMETER INDICES, not graph vertex labels.
    /// A wire's unmeasured output therefore has no entry here. Flattening keeps
    /// the supplied causal order; it must not reorder the caller's theta array.
    function MeasurementOrder(parameterCount : Int, slices : Int[][]) : Int[] {
        if parameterCount < 0 { fail "Parameter count cannot be negative."; }
        mutable seen = [false, size = parameterCount];
        mutable order = [];
        for slice in slices {
            if Length(slice) == 0 { fail "Temporal slices cannot be empty."; }
            for index in slice {
                if index < 0 or index >= parameterCount {
                    fail "A temporal slice contains an invalid parameter index.";
                }
                if seen[index] { fail "Temporal slices repeat a parameter index."; }
                set seen w/= index <- true;
                set order += [index];
            }
        }
        if Length(order) != parameterCount {
            fail "Temporal slices must partition all trainable parameters.";
        }
        return order;
    }

    function ValidateDiscreteInputs(theta : Double[], slices : Int[][], angles : Double[]) : Unit {
        RequireFiniteParameters(theta);
        RequireFiniteParameters(angles);
        if Length(angles) == 0 { fail "The allowed angle set cannot be empty."; }
        for i in 0..Length(angles) - 1 {
            for j in 0..i - 1 {
                if angles[i] == angles[j] { fail "Allowed angles must be distinct."; }
            }
        }
        for value in theta {
            mutable allowed = false;
            for angle in angles { if value == angle { set allowed = true; } }
            if not allowed { fail "Initial discrete parameters must belong to the angle set."; }
        }
        let _ = MeasurementOrder(Length(theta), slices);
    }

    operation CheckedLoss(loss : Double[] => Double, theta : Double[]) : Double {
        RequireFiniteParameters(theta);
        let value = loss(theta);
        RequireFiniteTrainingValue(value);
        return value;
    }

    function MakeTrainingResult(theta : Double[], value : Double, history : Double[],
        threshold : Double, evaluations : Int) : TrainingResult {
        return new TrainingResult {
            Parameters = theta, Loss = value, History = history,
            Success = value < threshold, Evaluations = evaluations
        };
    }

    /// Appendix E, Algorithm 1. Every window exhaustively enumerates the allowed
    /// assignments with an odometer (no exponentially large candidate array).
    /// As in the literal pseudocode, EACH candidate replaces the current pattern
    /// if it improves the loss OR an epsilon draw accepts it. This differs from
    /// choosing just one random window assignment with probability epsilon.
    /// The explicit initial pattern is attempt zero; restarts counts additional
    /// random attempts, so restarts = 0 still performs a useful search. Keeping
    /// a separate incumbent prevents exploration/restarts from losing the best
    /// solution. The evaluation budget bounds exponential windows and restarts.
    operation GreedyOptimize(loss : Double[] => Double, initial : Double[], slices : Int[][],
        angles : Double[], epsilon : Double, restarts : Int, maxWindow : Int,
        threshold : Double, maxEvaluations : Int) : TrainingResult {
        ValidateDiscreteInputs(initial, slices, angles);
        RequireProbability(epsilon);
        RequireFiniteTrainingValue(threshold);
        if restarts < 0 or maxWindow < 1 or maxEvaluations < 1 {
            fail "Greedy search requires nonnegative restarts and positive window/budget.";
        }
        mutable best = initial;
        mutable bestLoss = CheckedLoss(loss, initial);
        mutable history = [bestLoss];
        mutable evaluations = 1;
        mutable current = initial;
        mutable currentLoss = bestLoss;
        if bestLoss < threshold or Length(initial) == 0 {
            return MakeTrainingResult(best, bestLoss, history, threshold, evaluations);
        }
        for attempt in 0..restarts {
            if attempt > 0 {
                if evaluations >= maxEvaluations {
                    return MakeTrainingResult(best, bestLoss, history, threshold, evaluations);
                }
                for k in 0..Length(initial) - 1 {
                    set current w/= k <- angles[DrawRandomInt(0, Length(angles) - 1)];
                }
                set currentLoss = CheckedLoss(loss, current);
                set evaluations += 1;
                if currentLoss < bestLoss { set best = current; set bestLoss = currentLoss; }
                set history += [bestLoss];
                if bestLoss < threshold {
                    return MakeTrainingResult(best, bestLoss, history, threshold, evaluations);
                }
            }
            for width in 1..MinI(maxWindow, Length(slices)) {
                for start in 0..Length(slices) - width {
                    mutable indices = [];
                    for offset in 0..width - 1 { set indices += slices[start + offset]; }
                    mutable digits = [0, size = Length(indices)];
                    mutable finished = false;
                    while not finished {
                        if evaluations >= maxEvaluations {
                            return MakeTrainingResult(best, bestLoss, history, threshold, evaluations);
                        }
                        mutable candidate = current;
                        for j in 0..Length(indices) - 1 {
                            set candidate w/= indices[j] <- angles[digits[j]];
                        }
                        let candidateLoss = CheckedLoss(loss, candidate);
                        set evaluations += 1;
                        if candidateLoss < bestLoss { set best = candidate; set bestLoss = candidateLoss; }
                        set history += [bestLoss];
                        if candidateLoss < currentLoss or DrawRandomDouble(0.0, 1.0) < epsilon {
                            set current = candidate;
                            set currentLoss = candidateLoss;
                        }
                        if bestLoss < threshold {
                            return MakeTrainingResult(best, bestLoss, history, threshold, evaluations);
                        }
                        mutable place = 0;
                        mutable carry = true;
                        while carry and place < Length(digits) {
                            if digits[place] + 1 < Length(angles) {
                                set digits w/= place <- digits[place] + 1;
                                set carry = false;
                            } else {
                                set digits w/= place <- 0;
                                set place += 1;
                            }
                        }
                        set finished = carry;
                    }
                }
            }
        }
        return MakeTrainingResult(best, bestLoss, history, threshold, evaluations);
    }

    /// Exact two-shift derivative for a LINEAR expectation/fidelity objective
    /// whose independently parameterized gates have generators P/2, P^2 = I.
    /// It is not the derivative of an arbitrary nonlinear postprocessed loss,
    /// a tied angle used by multiple gates, or a general classical objective.
    /// Use CentralDifferenceGradient (or a supplied chain rule) in those cases.
    operation ParameterShiftGradient(loss : Double[] => Double, theta : Double[]) : Double[] {
        RequireFiniteParameters(theta);
        mutable gradient = [0.0, size = Length(theta)];
        for k in 0..Length(theta) - 1 {
            let plus = theta w/ k <- theta[k] + PI() / 2.0;
            let minus = theta w/ k <- theta[k] - PI() / 2.0;
            if plus[k] == theta[k] or minus[k] == theta[k] {
                fail "Parameter shift is too small for this parameter's magnitude.";
            }
            let value = (CheckedLoss(loss, plus) - CheckedLoss(loss, minus)) / 2.0;
            RequireFiniteTrainingValue(value);
            set gradient w/= k <- value;
        }
        return gradient;
    }

    operation CentralDifferenceGradient(loss : Double[] => Double, theta : Double[],
        step : Double) : Double[] {
        RequireFiniteParameters(theta);
        RequireFiniteTrainingValue(step);
        if step <= 0.0 { fail "Finite-difference step must be positive."; }
        mutable gradient = [0.0, size = Length(theta)];
        for k in 0..Length(theta) - 1 {
            let plus = theta w/ k <- theta[k] + step;
            let minus = theta w/ k <- theta[k] - step;
            if plus[k] == theta[k] or minus[k] == theta[k] {
                fail "Finite-difference step is too small for this parameter's magnitude.";
            }
            let value = (CheckedLoss(loss, plus) - CheckedLoss(loss, minus)) / (2.0 * step);
            RequireFiniteTrainingValue(value);
            set gradient w/= k <- value;
        }
        return gradient;
    }

    /// Adam for Section IV objectives. Gradient probes are included in both the
    /// evaluation count and incumbent history; any probe can improve the output.
    /// Parameter-shift applicability is the caller's responsibility; select the
    /// finite-difference path for nonlinear classical postprocessing.
    operation TrainAdam(loss : Double[] => Double, initial : Double[], config : AdamConfig) : TrainingResult {
        RequireFiniteParameters(initial);
        RequireFiniteTrainingValue(config.LearningRate);
        RequireProbability(config.Beta1);
        RequireProbability(config.Beta2);
        RequireFiniteTrainingValue(config.Epsilon);
        RequireFiniteTrainingValue(config.Threshold);
        RequireFiniteTrainingValue(config.DifferenceStep);
        if config.Steps < 0 or config.LearningRate <= 0.0 or config.Beta1 >= 1.0 or
            config.Beta2 >= 1.0 or config.Epsilon <= 0.0 or config.MaxEvaluations < 1 or
            config.DifferenceStep <= 0.0 {
            fail "Invalid Adam step count, rates, numerical epsilon, difference step, or budget.";
        }
        let n = Length(initial);
        mutable theta = initial;
        mutable best = theta;
        mutable bestLoss = CheckedLoss(loss, theta);
        mutable evaluations = 1;
        mutable history = [bestLoss];
        mutable moment = [0.0, size = n];
        mutable variance = [0.0, size = n];
        mutable beta1Power = 1.0;
        mutable beta2Power = 1.0;
        if n == 0 or bestLoss < config.Threshold {
            return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
        }
        for _ in 1..config.Steps {
            if config.MaxEvaluations - evaluations < 2 * n + 1 {
                return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
            }
            mutable gradient = [0.0, size = n];
            let shift = config.UseParameterShift ? PI() / 2.0 | config.DifferenceStep;
            let divisor = config.UseParameterShift ? 2.0 | 2.0 * shift;
            for k in 0..n - 1 {
                let plus = theta w/ k <- theta[k] + shift;
                let minus = theta w/ k <- theta[k] - shift;
                RequireFiniteParameters(plus);
                RequireFiniteParameters(minus);
                if plus[k] == theta[k] or minus[k] == theta[k] {
                    fail "Gradient shift is too small for this parameter's magnitude.";
                }
                let plusLoss = CheckedLoss(loss, plus);
                set evaluations += 1;
                if plusLoss < bestLoss { set best = plus; set bestLoss = plusLoss; }
                set history += [bestLoss];
                let minusLoss = CheckedLoss(loss, minus);
                set evaluations += 1;
                if minusLoss < bestLoss { set best = minus; set bestLoss = minusLoss; }
                set history += [bestLoss];
                let derivative = (plusLoss - minusLoss) / divisor;
                RequireFiniteTrainingValue(derivative);
                set gradient w/= k <- derivative;
            }
            set beta1Power *= config.Beta1;
            set beta2Power *= config.Beta2;
            for k in 0..n - 1 {
                set moment w/= k <- config.Beta1 * moment[k] + (1.0 - config.Beta1) * gradient[k];
                set variance w/= k <- config.Beta2 * variance[k] + (1.0 - config.Beta2) * gradient[k] * gradient[k];
                let correctedMoment = moment[k] / (1.0 - beta1Power);
                let correctedVariance = variance[k] / (1.0 - beta2Power);
                set theta w/= k <- theta[k] - config.LearningRate * correctedMoment /
                    (Sqrt(correctedVariance) + config.Epsilon);
            }
            RequireFiniteParameters(theta);
            let value = CheckedLoss(loss, theta);
            set evaluations += 1;
            if value < bestLoss { set best = theta; set bestLoss = value; }
            set history += [bestLoss];
            if bestLoss < config.Threshold {
                return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
            }
        }
        return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
    }

    /// Canonical angle values / pi, assignment mask, and next-index one-hot.
    /// The mask distinguishes a chosen zero angle from an unassigned parameter.
    /// nextIndex = -1 denotes terminal state (no next-index one-hot).
    function EncodeDqnState(theta : Double[], assigned : Bool[], nextIndex : Int) : Double[] {
        let n = Length(theta);
        if Length(assigned) != n or nextIndex < -1 or nextIndex >= n {
            fail "Invalid DQN state dimensions or next parameter.";
        }
        RequireFiniteParameters(theta);
        mutable state = [0.0, size = 3 * n];
        for k in 0..n - 1 {
            if assigned[k] {
                set state w/= k <- theta[k] / PI();
                set state w/= n + k <- 1.0;
            }
        }
        if nextIndex >= 0 { set state w/= 2 * n + nextIndex <- 1.0; }
        return state;
    }

    operation InitializeQNetwork(inputSize : Int, hiddenSize : Int, outputSize : Int) : QNetwork {
        if inputSize < 1 or hiddenSize < 1 or outputSize < 1 {
            fail "Neural-network dimensions must be positive.";
        }
        let firstScale = Sqrt(6.0 / IntAsDouble(inputSize + hiddenSize));
        let secondScale = Sqrt(6.0 / IntAsDouble(hiddenSize + outputSize));
        mutable first = [0.0, size = inputSize * hiddenSize];
        mutable second = [0.0, size = hiddenSize * outputSize];
        for j in 0..Length(first) - 1 { set first w/= j <- DrawRandomDouble(-firstScale, firstScale); }
        for j in 0..Length(second) - 1 { set second w/= j <- DrawRandomDouble(-secondScale, secondScale); }
        return new QNetwork {
            InputSize = inputSize, HiddenSize = hiddenSize, OutputSize = outputSize,
            InputWeights = first, HiddenBias = [0.0, size = hiddenSize],
            OutputWeights = second, OutputBias = [0.0, size = outputSize]
        };
    }

    function ValidateQNetwork(network : QNetwork) : Unit {
        if network.InputSize < 1 or network.HiddenSize < 1 or network.OutputSize < 1 or
            Length(network.InputWeights) != network.InputSize * network.HiddenSize or
            Length(network.OutputWeights) != network.HiddenSize * network.OutputSize or
            Length(network.HiddenBias) != network.HiddenSize or
            Length(network.OutputBias) != network.OutputSize {
            fail "Invalid Q-network dimensions.";
        }
        RequireFiniteParameters(network.InputWeights);
        RequireFiniteParameters(network.HiddenBias);
        RequireFiniteParameters(network.OutputWeights);
        RequireFiniteParameters(network.OutputBias);
    }

    function QNetworkForward(network : QNetwork, state : Double[]) : (Double[], Double[]) {
        ValidateQNetwork(network);
        RequireFiniteParameters(state);
        if Length(state) != network.InputSize { fail "Q-network state has incorrect dimension."; }
        mutable hidden = [0.0, size = network.HiddenSize];
        for h in 0..network.HiddenSize - 1 {
            mutable value = network.HiddenBias[h];
            for j in 0..network.InputSize - 1 {
                set value += network.InputWeights[h * network.InputSize + j] * state[j];
            }
            RequireFiniteTrainingValue(value);
            set hidden w/= h <- MaxD(0.0, value);
        }
        mutable output = network.OutputBias;
        for a in 0..network.OutputSize - 1 {
            for h in 0..network.HiddenSize - 1 {
                set output w/= a <- output[a] + network.OutputWeights[a * network.HiddenSize + h] * hidden[h];
            }
        }
        RequireFiniteParameters(output);
        return (hidden, output);
    }

    function ArgMaxQ(values : Double[]) : Int {
        if Length(values) == 0 { fail "Cannot select from empty Q values."; }
        RequireFiniteParameters(values);
        mutable best = 0;
        for a in 1..Length(values) - 1 { if values[a] > values[best] { set best = a; } }
        return best;
    }

    function PushReplay(buffer : ReplayTransition[], transition : ReplayTransition,
        capacity : Int) : ReplayTransition[] {
        if capacity < 1 or Length(buffer) > capacity { fail "Invalid replay capacity."; }
        if Length(buffer) == capacity { return buffer[1...] + [transition]; }
        return buffer + [transition];
    }

    function BellmanTarget(transition : ReplayTransition, target : QNetwork, gamma : Double) : Double {
        RequireProbability(gamma);
        RequireFiniteTrainingValue(transition.Reward);
        // Do not bootstrap terminal states, including when their placeholder
        // NextState has no features; this is essential for Appendix E's reward.
        if transition.Terminal { return transition.Reward; }
        let (_, values) = QNetworkForward(target, transition.NextState);
        let result = transition.Reward + gamma * values[ArgMaxQ(values)];
        RequireFiniteTrainingValue(result);
        return result;
    }

    /// One actual minibatch SGD step on mean-squared Bellman error. Every sample
    /// gradient uses the SAME pre-update online network. The target network is
    /// immutable during the step. ReLU derivative at zero is defined as zero.
    function DqnMinibatchStep(online : QNetwork, target : QNetwork,
        batch : ReplayTransition[], gamma : Double, learningRate : Double) : (QNetwork, Double) {
        ValidateQNetwork(online);
        ValidateQNetwork(target);
        RequireProbability(gamma);
        RequireFiniteTrainingValue(learningRate);
        if Length(batch) == 0 or learningRate <= 0.0 or online.InputSize != target.InputSize or
            online.HiddenSize != target.HiddenSize or online.OutputSize != target.OutputSize {
            fail "Invalid DQN minibatch, step size, or target network dimensions.";
        }
        mutable firstGradient = [0.0, size = Length(online.InputWeights)];
        mutable hiddenGradient = [0.0, size = online.HiddenSize];
        mutable secondGradient = [0.0, size = Length(online.OutputWeights)];
        mutable outputGradient = [0.0, size = online.OutputSize];
        mutable meanLoss = 0.0;
        let count = IntAsDouble(Length(batch));
        for sample in batch {
            if sample.Action < 0 or sample.Action >= online.OutputSize { fail "Invalid replay action."; }
            let (hidden, values) = QNetworkForward(online, sample.State);
            let expected = BellmanTarget(sample, target, gamma);
            let error = values[sample.Action] - expected;
            set meanLoss += error * error / count;
            let outputDelta = 2.0 * error / count;
            set outputGradient w/= sample.Action <- outputGradient[sample.Action] + outputDelta;
            for h in 0..online.HiddenSize - 1 {
                let secondIndex = sample.Action * online.HiddenSize + h;
                set secondGradient w/= secondIndex <- secondGradient[secondIndex] + outputDelta * hidden[h];
                if hidden[h] > 0.0 {
                    let hiddenDelta = outputDelta * online.OutputWeights[secondIndex];
                    set hiddenGradient w/= h <- hiddenGradient[h] + hiddenDelta;
                    for j in 0..online.InputSize - 1 {
                        let firstIndex = h * online.InputSize + j;
                        set firstGradient w/= firstIndex <- firstGradient[firstIndex] + hiddenDelta * sample.State[j];
                    }
                }
            }
        }
        mutable first = online.InputWeights;
        mutable hiddenBias = online.HiddenBias;
        mutable second = online.OutputWeights;
        mutable outputBias = online.OutputBias;
        for i in 0..Length(first) - 1 { set first w/= i <- first[i] - learningRate * firstGradient[i]; }
        for i in 0..Length(hiddenBias) - 1 { set hiddenBias w/= i <- hiddenBias[i] - learningRate * hiddenGradient[i]; }
        for i in 0..Length(second) - 1 { set second w/= i <- second[i] - learningRate * secondGradient[i]; }
        for i in 0..Length(outputBias) - 1 { set outputBias w/= i <- outputBias[i] - learningRate * outputGradient[i]; }
        let updated = new QNetwork {
            InputSize = online.InputSize, HiddenSize = online.HiddenSize, OutputSize = online.OutputSize,
            InputWeights = first, HiddenBias = hiddenBias, OutputWeights = second, OutputBias = outputBias
        };
        ValidateQNetwork(updated);
        RequireFiniteTrainingValue(meanLoss);
        return (updated, meanLoss);
    }

    function DqnEpsilon(config : DqnConfig, episode : Int) : Double {
        if episode < 0 { fail "Episode index cannot be negative."; }
        RequireProbability(config.EpsilonStart);
        RequireProbability(config.EpsilonMin);
        RequireProbability(config.EpsilonDecay);
        if config.EpsilonMin > config.EpsilonStart { fail "Minimum epsilon cannot exceed initial epsilon."; }
        if episode == 0 { return MaxD(config.EpsilonMin, config.EpsilonStart); }
        if config.EpsilonDecay == 0.0 { return config.EpsilonMin; }
        return MaxD(config.EpsilonMin, config.EpsilonStart * config.EpsilonDecay ^ IntAsDouble(episode));
    }

    /// Hard-copy schedule uses one-based COMPLETED optimizer steps. Thus period
    /// m copies after steps m, 2m, ...; initial target weights equal online.
    function UpdatedTarget(online : QNetwork, target : QNetwork,
        completedUpdates : Int, period : Int) : QNetwork {
        if completedUpdates < 1 or period < 1 { fail "Invalid target-copy schedule."; }
        if completedUpdates % period == 0 { return online; }
        return target;
    }

    /// Algorithm 2 SAMPLEEPISODE, exposed separately for policy evaluation and
    /// deterministic testing. Exactly one terminal objective evaluation occurs.
    operation SampleDqnEpisode(loss : Double[] => Double, parameterCount : Int,
        slices : Int[][], angles : Double[], online : QNetwork, epsilon : Double,
        buffer : ReplayTransition[], capacity : Int) : (Double[], Double, ReplayTransition[]) {
        if parameterCount < 0 or Length(angles) == 0 or capacity < 1 or Length(buffer) > capacity {
            fail "Invalid episode parameter count, angle set, or replay capacity.";
        }
        ValidateDiscreteInputs([angles[0], size = parameterCount], slices, angles);
        RequireProbability(epsilon);
        let order = MeasurementOrder(parameterCount, slices);
        if parameterCount == 0 { return ([], CheckedLoss(loss, []), buffer); }
        ValidateQNetwork(online);
        if online.InputSize != 3 * parameterCount or online.OutputSize != Length(angles) {
            fail "Episode network dimensions do not match the parameters and actions.";
        }
        mutable replay = buffer;
        mutable theta = [0.0, size = parameterCount];
        mutable assigned = [false, size = parameterCount];
        mutable value = 0.0;
        for position in 0..parameterCount - 1 {
            let index = order[position];
            let state = EncodeDqnState(theta, assigned, index);
            mutable action = 0;
            if DrawRandomDouble(0.0, 1.0) < epsilon {
                set action = DrawRandomInt(0, Length(angles) - 1);
            } else {
                let (_, values) = QNetworkForward(online, state);
                set action = ArgMaxQ(values);
            }
            // Append-order action selection must be mapped back to canonical
            // theta[index]. A nontrivial order such as [1, 0] is valid.
            set theta w/= index <- angles[action];
            set assigned w/= index <- true;
            let terminal = position == parameterCount - 1;
            mutable reward = 0.0;
            mutable nextIndex = -1;
            if terminal {
                set value = CheckedLoss(loss, theta);
                set reward = -value;
            } else { set nextIndex = order[position + 1]; }
            let transition = new ReplayTransition {
                State = state, Action = action, Reward = reward,
                NextState = EncodeDqnState(theta, assigned, nextIndex), Terminal = terminal
            };
            set replay = PushReplay(replay, transition, capacity);
        }
        return (theta, value, replay);
    }

    /// Appendix E, Algorithm 2: epsilon-greedy episodes, terminal reward -loss,
    /// FIFO replay, random minibatches with replacement, SGD, and a lagged target
    /// network. Epsilon decays geometrically per episode. Training occurs after
    /// each complete episode, once a full minibatch is available. This explicit
    /// finite driver fills in scheduling/network choices left open by the paper.
    operation TrainDqn(loss : Double[] => Double, initial : Double[], slices : Int[][],
        angles : Double[], config : DqnConfig) : TrainingResult {
        ValidateDiscreteInputs(initial, slices, angles);
        RequireFiniteTrainingValue(config.LearningRate);
        RequireFiniteTrainingValue(config.Threshold);
        RequireProbability(config.Gamma);
        RequireProbability(config.EpsilonStart);
        RequireProbability(config.EpsilonMin);
        RequireProbability(config.EpsilonDecay);
        if config.Episodes < 0 or config.HiddenSize < 1 or config.LearningRate <= 0.0 or
            config.EpsilonMin > config.EpsilonStart or config.ReplayCapacity < 1 or config.BatchSize < 1 or
            config.BatchSize > config.ReplayCapacity or config.UpdatesPerEpisode < 1 or
            config.TargetPeriod < 1 or config.MaxEvaluations < 1 {
            fail "Invalid DQN episode, network, replay, exploration, or evaluation configuration.";
        }
        let n = Length(initial);
        mutable best = initial;
        mutable bestLoss = CheckedLoss(loss, best);
        mutable history = [bestLoss];
        mutable evaluations = 1;
        if n == 0 or bestLoss < config.Threshold {
            return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
        }
        mutable online = InitializeQNetwork(3 * n, config.HiddenSize, Length(angles));
        mutable target = online;
        mutable replay = [];
        mutable updates = 0;
        for episode in 0..config.Episodes - 1 {
            if evaluations >= config.MaxEvaluations {
                return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
            }
            let epsilon = DqnEpsilon(config, episode);
            let (theta, value, newReplay) = SampleDqnEpisode(loss, n, slices, angles, online,
                epsilon, replay, config.ReplayCapacity);
            set replay = newReplay;
            set evaluations += 1;
            if value < bestLoss { set best = theta; set bestLoss = value; }
            set history += [bestLoss];
            if bestLoss < config.Threshold {
                return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
            }
            if Length(replay) >= config.BatchSize {
                for _ in 1..config.UpdatesPerEpisode {
                    mutable batch = [];
                    for _ in 1..config.BatchSize {
                        set batch += [replay[DrawRandomInt(0, Length(replay) - 1)]];
                    }
                    let (updated, _) = DqnMinibatchStep(online, target, batch, config.Gamma, config.LearningRate);
                    set online = updated;
                    set updates += 1;
                    set target = UpdatedTarget(online, target, updates, config.TargetPeriod);
                }
            }
        }
        return MakeTrainingResult(best, bestLoss, history, config.Threshold, evaluations);
    }
}
