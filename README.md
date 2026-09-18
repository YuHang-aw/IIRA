# IIRA2 Macro Policy

This repository contains the public, cache-based Macro controller used for
auditable medical-image evidence experiments. The reinforcement-learning stage
trains a small MLP over precomputed probabilities and localization features;
it does not update the foundation model or the visual verifier.


## Macro contract

The state has nine features in this fixed order:

```text
[qp, q_ent, q_conf, kp, k_conf, abs(qp-kp), qroi_p,
 abs(qp-qroi_p), loc_score]
```

Actions are `DIRECT_COMMIT`, `QUERY_AND_REVISE`, and `ABSTAIN`. All four
evidence values are already present in the state, so this implementation is a
cache-based evidence adoption policy. It does not claim online evidence
acquisition.

The default policy is a `9 -> 64 -> 64` Tanh MLP with action, alpha, and gamma
heads. The loss uses a group baseline, symmetric PPO-style ratio clipping,
uniform-policy KL, and entropy regularization. Ranking reward is intentionally
not part of this minimal public implementation until its source formula is
fully specified.

Two identical QUERY-masked differentiable reward terms train the fusion heads;
their sum has coefficient two. The detached group advantage trains action logits.

Status: partial reconstruction. The core numerical functions have synthetic CPU
tests. A full trainer, cache-schema adapter, checkpoint compatibility and Ascend
execution are not yet verified. This is not an experiment reproduction claim.
Historical plans under `docs/superpowers` describe earlier designs; the current
core contract is `docs/PUBLIC_RECONSTRUCTION_SPEC_CN.md`.

## Quick start

```bash
python -m pip install -e .
python -m pytest -q
```

The tests use only synthetic probabilities and labels; they do not require
restricted data or model downloads.

## Hardware execution

Use the official PyTorch and `torch_npu` versions supplied for the target
Ascend 910B/910C installation. The controller has only a few thousand
parameters, so the NPU profile is useful for integration and end-to-end cache
generation rather than for fitting this MLP itself. Keep model and dataset
identifiers in external manifests and pass their locations through the three
environment variables above.
