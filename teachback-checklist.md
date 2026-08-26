# ARC solver teachback checklist

## Problem and interface

- [ ] Explain the episode interface `(x1, y1, ..., X) -> Y_hat`.
- [ ] Distinguish the support demonstrations from the unseen query target.

## Current structured solution

- [ ] Trace a color task from raw grids to association counts, recurrent state, and output.
- [ ] Trace a spatial task from raw grids to hypothesis scores, recurrent state, and output.
- [ ] Identify exactly which computations are hand-designed and which parameters are learned.

## Memory and learning

- [ ] Distinguish long-term memory in weights from temporary task memory in activations.
- [ ] Explain how recurrent ticks revise temporary memory without changing model weights.
- [ ] Explain what is—and is not—remembered between ARC tasks.

## Proposed learned encoder

- [ ] Explain how a learned pair encoder could replace hand-built evidence extraction.
- [ ] Explain why experience, inductive bias, retrieval memory, and discovery are complementary.
- [ ] Describe a staged path from structured inference to raw-pair rule discovery.

## Evidence and remaining gaps

- [ ] Interpret the task-level holdout, color-relabeling, and corrupted-support controls.
- [ ] State why the current result is promising but not general ARC solving.

## Current understanding supplied by Johan

- The desired encoder likely needs prior experience and/or suitable inductive biases.
- A useful design may combine persistent memory with discovery on the current task.
