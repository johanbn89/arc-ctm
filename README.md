# ARC CTM

This repository tests a focused hypothesis:

> Can recurrent latent inference identify a task-invariant operator from a small demonstration set?

The first experiment is deliberately synthetic. Each episode samples one grid operator and provides a few `(input, output)` demonstrations plus unseen query inputs. A single task state is shared across every demonstration. It is revised over recurrent CTM ticks using the residual between the current support predictions and the known support outputs. The final state is then applied to unseen query grids.

Because the synthetic generator knows the ground-truth operator, outer training also uses a configurable auxiliary operator-classification loss on the inferred state. This label is never supplied to the inner loop or at inference. It makes operator identification explicit and prevents a degenerate decoder from minimizing loss by averaging all transformations. Set `--operator-loss-weight 0` for the fully emergent variant.

The implementation follows the defining CTM mechanics—internal recurrence, temporal neuron-level models, and synchronization-derived representations—but adds weight sharing across neurons. The default `typed` model uses `K` shared temporal processors, soft learned neuron-to-type assignments, and small private gain/offset parameters.

## Architecture

At tick `t`:

1. Decode the current task state with the same conditional grid decoder `G(h_t, x)` for every support input.
2. Compare those predictions with the support outputs and permutation-invariantly aggregate the residual evidence.
3. Update the CTM pre-activation history with a shared synapse network.
4. Apply the typed neuron-level temporal model.
5. Form the next task representation from recurrent neuron synchronization.
6. Apply that representation to both support and unseen query inputs.

The model parameters remain fixed inside an episode. Only recurrent activations change, so the ticks implement an activation-space inner learning loop.

## Run

Create an editable installation:

```powershell
python -m pip install -e .
```

Run a short typed-CTM experiment:

```powershell
arc-ctm --steps 500 --mode typed --types 8
```

Compare the neuron-level sharing regimes:

```powershell
arc-ctm --steps 500 --mode private
arc-ctm --steps 500 --mode shared
arc-ctm --steps 500 --mode typed --types 4
arc-ctm --steps 500 --mode typed --types 8
arc-ctm --steps 500 --mode typed --types 16
```

By default, the command writes a checkpoint and JSON metrics under `artifacts/`. Use `--no-save` for a smoke run.

## Current synthetic task

Episodes use square categorical grids and the eight dihedral symmetries: identity, quarter/half/three-quarter rotations, left-right and up-down reflection, main-diagonal reflection, and anti-diagonal reflection. Support and query grids are independently sampled. Query loss—not reconstruction alone—is the primary objective.

Reported metrics include:

- query cell and exact-grid accuracy at every tick;
- support accuracy at every tick;
- operator-identification accuracy at every tick;
- the first tick at which all support demonstrations are solved;
- functional agreement when the same operator is inferred from different support sets;
- behavioral and task-state separation between same-operator and different-operator episodes on identical queries;
- parameter counts and learned neuron-type usage.

## Research sequence

1. Establish unseen-input operator inference.
2. Measure invariance across independently sampled demonstration subsets.
3. Train and validate adaptive stopping against actual operator convergence.
4. Compare typed CTM with private-NLM CTM, GRU/LSTM, and recurrent Transformer baselines.
5. Replace the direct grid decoder with a restricted ARC DSL executor.

The current code covers the first experiment and already exposes the invariance measurements needed to begin the second. It intentionally does not claim ARC generalization: random synthetic grids isolate operator identification before object-centric ARC structure and program decoding are introduced.

## Genuine ARC-AGI-1 single-task experiment

The repository also contains a stricter first contact with a real ARC task. Download the canonical ARC-AGI-1 repository into `data/arc_agi_1`, then run:

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/fchollet/ARC-AGI.git data/arc_agi_1
git -C data/arc_agi_1 sparse-checkout set data/training data/evaluation
arc-ctm-overfit --task-id 0d3d703e
```

Task `0d3d703e` is a genuine 3x3 color-substitution problem. Training uses only its four demonstration pairs:

- each step holds out one demonstration as the query;
- the other three demonstrations form the support set;
- color identities are randomly permuted per episode, preventing fixed-color memorization;
- the official test output is never used in the loss;
- after training, support outputs are deliberately corrupted to measure whether the prediction actually depends on the demonstrations.

The default real-task decoder is `cellwise`: it applies one shared, task-conditioned color rule to every location. This is the appropriate inductive bias for `0d3d703e` and prevents the global flattened-grid decoder from memorizing entire demonstration patterns. Use `--decoder-mode global` to reproduce that control.

The command saves a checkpoint, JSON metrics, and a dependency-free visual report under `artifacts/arc_overfit/`. This is an overfit/capacity experiment, not an ARC benchmark score.

### Explicit binding-state experiment

For color-substitution tasks, the structured alternative represents the inferred
operator directly as a recurrent `10 x 10` color-binding matrix. Every binding
edge shares the same temporal residual update, so simultaneous relabeling of
input and output colors does not change the algorithm:

```powershell
arc-ctm-overfit --architecture binding --task-id 0d3d703e --steps 20
```

The task's mapping is an involution, so reverse evidence is enabled by default
(`5 -> 1` also supports `1 -> 5`). Ablate that assumption with:

```powershell
arc-ctm-overfit --architecture binding --task-id 0d3d703e --steps 20 --no-involution-prior
```

This model is an explicit structured baseline, not a claim that an unconstrained
CTM discovered the representation. Its purpose is to determine whether the
operator state and residual-learning loop are sufficient before making the
update rule less hand-designed.

To keep the binding state and residual structure but learn the shared recurrent
update from a random neural initialization, run:

```powershell
arc-ctm-overfit --architecture binding --binding-update learned --task-id 0d3d703e --steps 300
```

Final evaluation includes 128 fresh randomly relabeled leave-one-out episodes by
default, separate from the relabelings sampled during optimization.

## Genuine ARC task-level transfer

The multi-task experiment removes the largest weakness of the single-task
overfit. It selects four official ARC-AGI-1 training tasks using demonstrations
and public test inputs only, then runs complete leave-one-task-out evaluation:

```powershell
arc-ctm-multitask --steps 60 --randomized-eval-episodes 256
```

For each fold, the shared recurrent updater is optimized on three task IDs using
random color relabelings. The fourth task ID is never sampled during
optimization. At inference, its demonstrations are reduced to color-association
counts, allowing examples with different rectangular grid dimensions, and the
learned updater builds a fresh `10 x 10` binding state. Its official test target
is used only for final scoring.

The four tasks (`0d3d703e`, `b1948b0a`, `c8f0f002`, and `d511f180`) are all
deterministic cellwise color substitutions. This is therefore transfer within a
narrow, explicitly selected rule family—not general ARC solving. The binding
state and residual evidence remain hand-structured; only the small shared
recurrent update network is learned. The command writes per-fold checkpoints,
JSON results, and a standalone visual report under `artifacts/arc_multitask/`.

## Genuine ARC spatial-operator transfer

A second task family tests whether the same recurrent-inference idea extends
beyond color bindings. Seven official ARC tasks were selected solely because
their training demonstrations uniquely identify one non-identity transform from
the eight dihedral grid operations:

```powershell
arc-ctm-spatial --steps 60 --randomized-eval-episodes 256
```

The experiment again holds out complete task IDs. Demonstrations are scored
against the fixed spatial operator bank, a 68-parameter shared updater revises
the operator state for six ticks, and the most probable discrete operator is
executed on the query. Random color relabeling prevents dependence on the
original palette. Deliberately flipped support outputs and an asymmetric probe
test whether the inferred program changes with the evidence.

This is a structured program-selection experiment: the rotations/reflections
and mismatch evidence are supplied by code rather than learned from pixels. It
validates task-conditioned recurrent selection and execution, not discovery of
the operator library itself.

## Learned raw-pair encoder and unseen compositions

The composition experiment replaces hand-built demonstration evidence with a
learned bilinear pair encoder. Each task applies one spatial transform followed
by one cyclic recoloring. The model sees raw `(input, output)` demonstrations,
recurrently revises a low-dimensional operator state from learned prediction
errors, and executes factorized spatial and color heads on unseen queries:

```powershell
arc-ctm-compose --steps 500 --learning-rate 0.001
```

Two identically initialized models receive the same episodes. The I/O-only
baseline is trained from support/query grid losses. The trace-supervised model
also receives the spatial operator token, recoloring token, and intermediate
post-spatial grid. Four compositions are reserved for validation and four
different compositions form an untouched test split. Every primitive occurs in
training, but the excluded pairings do not.

The executor's spatial/color factorization and candidate primitive library are
still hand-designed. The learned advance is the raw-pair evidence encoder and
recurrent operator inference—not unrestricted operator invention. Results,
checkpoints, replication summary, and a standalone visual report are written to
`artifacts/arc_composition/`.
