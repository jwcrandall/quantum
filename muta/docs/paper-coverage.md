# Paper coverage and conventions

This project implements the computational methods in Luis Mantilla Calderón,
Robert Raussendorf, Polina Feldmann, and Dmytro Bondarenko,
[*Measurement-based quantum machine learning*](https://doi.org/10.1103/2snk-m8c6),
Physical Review A **113**, 042421 (2026). Page numbers below refer to the
14-page published article.

The implementation provides reusable algorithms and small correctness checks.
It does not reproduce the paper's datasets, optimization hyperparameters,
repeated-run averages, figures, or reported classification accuracies. Quantum
primitives and classical optimization are implemented in Q#; a Python runner
loads the project and invokes its examples and checks.
The runnable gate example uses one wire, the discrete examples expose one free
angle, and the teleportation example trains correction angles while holding the
two Bell stages fixed. Their general objective APIs also accept entangling
targets, complete discrete patterns, and all twenty staged instrument angles.

## Scope by paper section

| Paper location | Computational method | Implementation scope |
| --- | --- | --- |
| Sec. II, p. 2, Eq. (1) | Open graph-state preparation and equatorial measurements | Graph construction, explicit input/output sets, a supplied flow with a compatible schedule, and outcome-dependent Pauli corrections equivalent to adaptive measurement angles and an output frame. |
| Sec. III, pp. 3–4 | MuTA layers, partial triangle connectivity, and layer concatenation | Five-vertex wire segments, four-vertex triangle motifs, selectable tips and connected rows, and composition through designated input/output identifications. |
| Sec. III, Table I, p. 3 | Single-qubit rotations and tunable Ising-XX interaction | Measurement patterns and a unitary reference provide the universal MuTA building blocks. |
| Sec. II, Eq. (2); Sec. IV A, p. 5 | Supervised gate learning | Mean infidelity evaluated with caller-supplied state preparation and target operations; continuous optimization with Adam and numerical or parameter-shift derivatives as appropriate. |
| Sec. IV B, p. 5 | Training with noisy labels | Bit-flip and sampled Brownian-unitary label perturbations, separate from noise on the resource state. |
| Sec. IV C, pp. 5–6, Eq. (3) | Metrological quantum-state classifier | The two state families, tied wire angles, output probabilities, a trainable degree-two polynomial head, and the soft-margin classification objective. |
| Sec. IV D, pp. 6–7, Fig. 6 | Learning a teleportation instrument | A staged MuTA resource preparation, Bell analysis, and measurement-conditioned correction protocol with an infidelity objective. It implements the task with an explicit staged layout, rather than copying the 23-vertex drawing. |
| Sec. IV E, p. 7, Eqs. (4)–(5) | MuTA feature map and kernel SVM | The specified two-feature embedding, MuTA and unitary-reference execution, fidelity and SWAP-test kernel estimates, kernel matrices, and a classical SVM solver and predictor. |
| Sec. V, pp. 7–8 | Optimization with restricted measurement angles | Caller-selected angle alphabets; the paper's nominal alphabet is `{0, π/4, π/2}`. This is an algorithmic restriction, not an implementation of a GKP encoding or a magic-state factory. |
| App. B, pp. 10–11 | General MBQC-to-circuit translation | For equal input/output counts, follow the live input positions along flow paths, insert retained junction CZ gates, apply `H Rz(-α)`, and finish with CZ gates between retained outputs. Output ordering is accounted for. |
| App. C, p. 11 | Dynamical Lie-algebra analysis | A bounded Pauli-word commutator closure utility supports small generator sets. It does not prove finite-depth expressivity, estimate trainability, or handle arbitrary dense Hamiltonians. |
| App. D, p. 11 | Local depolarization of the resource state | Sampled Pauli-error trajectories implement the stated channel before the adaptive measurement sequence. Averaging includes the random measurement outcomes rather than choosing a favorable branch. |
| App. E, pp. 11–12, Algorithm 1 | Temporal slicewise epsilon-greedy search | Exhaustive candidate assignments within consecutive flow slices, exploratory acceptance, restarts, and an early-stopping loss threshold. |
| App. E, p. 12, Algorithm 2 | Deep-Q optimization of measurement patterns | Sequential partial-pattern episodes, terminal loss reward, a neural online Q function, replay minibatches, Bellman targets, and a periodically synchronized target network. |

Appendix A's theorem about classical ReLU-like networks and the proofs of MuTA's
architectural properties are mathematical results, not additional training
algorithms. The recurrent, attention-based, and probabilistically adapted
extensions in Sec. VI are research proposals and are not presented as implemented
methods. References to the paper's seven properties describe its ideal model;
they are not guarantees of noisy optimization or a claim that every fixed-depth
restricted-angle model is universal.

## Angle, graph, and operator conventions

The observable measured at a graph vertex is

\[
M(\alpha)=\cos(\alpha)X+\sin(\alpha)Y.
\]

Q# uses `Rz(θ) = exp(-i θ Z / 2)`, so one corrected cluster-wire measurement
teleports the input through `H Rz(-α)`. A basis measurement can be performed by
applying `Rz(-α)` followed by `H` and measuring in the computational basis.
The executor applies the flow's conditional X and Z corrections directly to
unmeasured resource qubits. This is equivalent to sign and π-shift adjustments
of subsequent measurement angles and a tracked output frame. The nominal angle
array therefore remains unchanged while the quantum control flow responds to
measurement outcomes.

Angles are stored in **measurement-order slots**, not vertex-number order.
For slot `s`, its angle belongs to `pattern.Order[s]`. Outputs have no measurement
angle. Translate paper labels through that ordering when constructing a pattern.
Training slices group these slots by their temporal dependencies. Appendix E's
example list includes output vertices `{4,9}`; this implementation excludes
outputs, consistently with the paper's definition of the measured set `V \\ O`.
The constructor chooses maximal available flow slices, which can be more
parallel than the illustrative ordering in Appendix E. No fixed numerical
vertex order should be assumed. A supplied alternate order must obey the same
causal dependencies.

A triangle between tip row `i` and base row `j` consists of
`Q[i,1], Q[j,0], Q[j,1], Q[j,2]`. In addition to wire edges, its two cross-wire
edges connect `Q[i,1]` with `Q[j,0]` and `Q[j,2]`. It is a four-vertex motif on a
bipartite graph. The vertices describe qubits, without prescribing a physical
layout or any Majorana encoding.

For a MuTA layer, let `a[k,c]` be the angle at row `k`, column `c`, for
`c = 0,1,2,3`. Equations (B1)–(B2) give the following chronological circuit:

1. Apply `Rz(-a[k,0])` on every wire `k`.
2. Apply `Rx(-a[i,1])` on the tip. For each other wire `j`, apply
   `Rxx(-a[j,1])` between `i` and `j` when that triangle is present; otherwise
   apply `Rx(-a[j,1])` on `j`.
3. Apply `Rz(-a[k,2])` on every wire.
4. Apply `Rx(-a[k,3])` on every wire.

Here `Rxx(θ) = exp(-i θ X⊗X / 2)`. Operators within each listed stage commute.
With no triangles, this reduces to independent wire rotations. With all angles
zero, it is the identity. In the paper's `(2,0)` layer, making only vertex 6's
angle nonzero gives `Rxx(-α6)`, so its output on `|00⟩` is
`cos(α6/2)|00⟩ + i sin(α6/2)|11⟩`. This sign matters when comparing with Table I.

## Classifier and noise conventions

The metrological classifier uses the normalized single-qubit generator
`h = Z/2` and the collective generator `H = (Z⊗I + I⊗Z)/2`. Consequently,

\[
F_Q=4\left[p_+ + p_- -(p_+ - p_-)^2\right],
\]

where `p+` and `p-` are the probabilities of `00` and `11` in the generator's
eigenbasis. This gives SQL `2` and Heisenberg limit `4`, matching the normalization
defined in Sec. IV C. That section later writes `h = Z`; taken literally this
would rescale both thresholds by four. The code uses the earlier normalized
definition. The trainable polynomial head is independent of this analytic QFI
formula; the formula supplies labels and an evaluation reference.

The resource-noise parameter follows Appendix D exactly:

\[
\mathcal N_p(\rho)=(1-p)\rho+
\frac p3(X\rho X+Y\rho Y+Z\rho Z).
\]

Thus `p` is the total probability of a nonidentity Pauli error. It is not the
alternative convention in which `p` is the weight of the maximally mixed state.
Independent samples reproduce this channel in the ensemble; individual runs
remain stochastic trajectories. No claim is made that a physical device's
noise follows this model.

Brownian perturbations are generated from explicitly sampled Hamiltonian
increments. Their discretization and normalization are implementation choices;
they are not an exact reproduction of the Fig. 4 ensemble from the paper's
separate Brownian-circuit reference. Draw noisy labels once and hold them fixed
when comparing training objectives if the experiment is intended to model a
fixed corrupted dataset.

## Optimization semantics and limits

Algorithm 1's exploration rule is interpreted literally from its pseudocode:
for each candidate, accept it if it improves the current loss **or** with
probability epsilon. This can accept a worse candidate. The exhaustive work for
a window with `r` measured slots is proportional to `|alphabet|^r`; larger
windows are not assumed computationally inexpensive.
The supplied initial pattern is attempt zero, and `restarts` counts additional
random attempts. This makes `restarts = 0` a useful single run. A separate
incumbent preserves the best evaluated pattern, including across exploratory
steps and restarts. This is a practical extension of the pseudocode's final
return value.

Algorithm 2 follows Appendix E's partial-pattern formulation: intermediate
rewards are zero and the terminal reward is negative loss. This differs from
the more informal complete-pattern description in Sec. V. The target network,
replay buffer, and neural function approximator are real training components,
not a table or a random-search substitute. Random seeds, exploration schedules,
network shape, and stopping limits remain explicit choices for an experiment.
The implementation encodes assigned-angle values, an assignment mask, and the
next-slot index; the mask distinguishes an unassigned slot from an angle set to
zero. Exploration decays geometrically per episode. Target synchronization uses
one-based completed training updates, so a period `m` copies the online network
after updates `m, 2m, …`. Replay minibatches are sampled with replacement.

All trainers return the best evaluated parameters, their loss, a best-so-far
history, success status, and the actual objective-evaluation count. Derivative
probes count as evaluations and can improve the returned incumbent. An explicit
evaluation budget bounds the work even when window enumeration or gradient
calculation would otherwise be expensive.

The familiar two-evaluation parameter-shift rule applies to one independently
parameterized Pauli-generated rotation and a linear expectation-value objective.
For tied parameters, contributions must be summed over their occurrences.
For nonlinear objectives such as the polynomial/hinge classifier, differentiate
the measured probabilities and apply the classical chain rule, or use numerical
derivatives. Applying the same two-point shift directly to an arbitrary loss
callback is not generally valid.

## Portability and validation boundary

The Q# project contains no cloud-provider configuration, credentials, device
topology, or tetron-specific operations. Its quantum contracts use qubits,
standard gates, measurements, reset, and caller-supplied state-preparation
operations. Its learning contracts use classical objective callbacks. This
separates algorithm specification from a provider integration; it does not mean
every QPU can execute the entire program unchanged.

Adaptive MBQC requires mid-circuit measurement and feedforward. Q# simulation can
execute the full hybrid control flow. Hardware execution depends on the target's
supported profile, reset and feedback capabilities, compilation, and available
resources. A practical deployment can keep classical optimization on the host
and submit supported quantum objective evaluations separately. Restricted-angle
experiments still use ideal qubit rotations in this reference implementation;
they do not synthesize or distill non-Clifford resources.

The executable compilation boundary is demonstrated by `export_qir.py`, which
loads only `Core.qs` and `QuantumEntry.qs` and emits `Adaptive_RI` QIR. The full
project uses `Std.Random` in its classical optimizers and noise samplers, which
is not supported by that target profile. QIR export is therefore a check of a
quantum workload, not a claim that the entire hybrid trainer runs on a QPU.

Correctness should be judged through independent unitary references, phase-
sensitive or entangled inputs, proper outcome correction, estimator identities,
and optimizer invariants. Small demonstrations are checks of implementation
behavior, not evidence of quantum advantage, scalable trainability, or the
paper's reported numerical performance.
