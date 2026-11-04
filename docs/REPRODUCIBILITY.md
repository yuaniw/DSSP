# Reproduction and compatibility notes

This document records validation of this RoboTwin release. It is separate from the aggregate manuscript results presented in the [main README](../README.md). Detailed local job logs and data are not distributed with the code.

## Validation setup

- Task: Stack Bowls Two (`stack_bowls_two`).
- Setting: `demo_clean`, 50 demonstrations, training seed 2.
- Method: the selected DSSP trajectory configuration, XYZ point clouds, Mamba1, EMA, and dynamics loss weight 0.05.
- Training: 1,000 epochs, 49,000 optimizer updates.
- Hardware used for the rename checks: one NVIDIA RTX 4090.
- RoboTwin base revision: `ea25643210c3036e57eb44b4cd4b59b8dc1e98d7`.

## Existing reproduction evidence

The pre-rename fresh-training run completed on November 4, 2026: 1,000 epochs and
49,000 optimizer updates. On Stack Bowls Two, the retrained checkpoint scored
90/100, the original checkpoint rerun scored 92/100, and the historical paper
result was 93%. Expert filtering selected different scene seeds in the two
reruns (95 common seeds); this is not an exact paired 100-scene comparison.

The public summary here records the measured results. Detailed migration reports, machine-specific filesystem paths, and cluster logs are maintained locally and are not included in the public code release.

## DSSP rename validation

On November 4, 2026, GPU validation verified strict loading of the existing
checkpoint, including the model, EMA and optimizer. All 227 state-dictionary
entries remain compatible. Direct actions, six cached action chunks and the
loss on a 303-step training trajectory matched the pre-rename implementation
exactly. The largest gradient difference was 4.44e-12, below the 5.29e-12
difference observed when repeating the unchanged model's backward pass.

The integration check completed one fresh training epoch (49 optimizer updates), then
evaluated the existing 1,000-epoch checkpoint through the public DSSP launcher.
The single evaluation episode succeeded and produced a valid H.264 video:
308 frames, 320 by 240 pixels, 30.8 seconds. The video also passed a full decode
check. The existing checkpoint's SHA-256 hash is unchanged. This integration
check does not replace the historical 100-episode benchmark reported above.

## Dataset and checkpoint provenance

Validation reused the existing demonstration dataset and simulator assets through local links. It did not generate new demonstrations. These filesystem links are specific to the validation machine and are ignored by Git. New users must prepare the data and assets described in the [usage guide](../policy/DSSP/README.md).

The original research checkout and checkpoint weights remain intact. The locally retrained checkpoint has SHA-256:

```text
915262a66fbf2037d03a7e2bf220b4c470728d47d0611a97f6977e5dfe690dc3
```

The DSSP checkpoint directory is a relative alias to the existing pre-rename directory on the validation machine. This alias is not a downloadable pretrained model. An external checkpoint can instead be selected with `--checkpoint_path`.

## Scope and remaining validation

Only the RoboTwin DSSP implementation is included. The paper's MetaWorld, Adroit, and real-robot experiments are not part of this checkout. The local single-task validation does not establish a new 50-task aggregate, and the one-episode video check is not a statistical success-rate estimate.

Dependency versions are recorded in `policy/DSSP/environment.yaml`; installing the full environment from scratch on a clean machine remains untested.
