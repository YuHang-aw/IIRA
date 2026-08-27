# IIRA 2.0 KBCSv2 Medical Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, calibrate, audit, freeze, and cache a RAD-DINO-based medical image sensor that emits separate disease probability, localization, and reliability evidence.

**Architecture:** A frozen-or-partially-tuned RAD-DINO backbone emits CLS and patch features. Separate classifier and pathology-conditioned localizer heads consume those features; independent post-hoc calibration and cross-fitted reliability estimation complete the immutable evidence record. Model selection uses only MIMIC development partitions, and the selected checkpoint is frozen before response-cache or controller work.

**Tech Stack:** Python 3.11, PyTorch, `torch_npu` on target only, Transformers, NumPy, pandas, scikit-learn, scikit-image, PyArrow, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-iira2-design.md`

## Global Constraints

- Execute the foundation/assets and data/Qwen plans first.
- Backbone is `microsoft/rad-dino` revision `110cbc18d5133582e320b43d53bf5c44e410c936` and loads from local files only.
- `disease probability != localization score != reliability` in type names, storage columns, and metrics.
- Main localizer development uses MIMIC train plus authorized MIMIC-CXR-Ext-ILS; VinDr test is evaluation-only.
- MIMIC validation is partitioned into model-selection and calibration subsets; test is locked.
- Reliability is estimated from out-of-fold held-out correctness and never reads test outcomes.
- Missing ROI is represented by `None`, never a whole-image box.
- Once evidence caching begins, KBCSv2 checkpoint and calibrators are frozen and content-addressed.
- Every implementation task follows red-green-refactor and ends with an isolated commit.

---

### Task 1: External evidence schema and serialization

**Files:**
- Create: `src/iira2/evidence/schema.py`
- Create: `src/iira2/evidence/base.py`
- Create: `src/iira2/evidence/__init__.py`
- Test: `tests/evidence/test_schema.py`

**Interfaces:**
- Consumes: pathology-specific classifier, localization, and reliability outputs.
- Produces: `ExternalEvidence`, `EvidenceModel`, `EvidenceIdentity`, and `ExternalEvidence.validate()`.

- [ ] **Step 1: Write failing separation and ROI tests**

```python
def test_evidence_keeps_three_scores_separate(valid_evidence) -> None:
    assert valid_evidence.probability_calibrated != valid_evidence.localization_score
    assert valid_evidence.reliability != valid_evidence.localization_score


def test_missing_roi_requires_both_coordinate_forms_none(valid_evidence) -> None:
    invalid = replace(valid_evidence, roi_xyxy_normalized=None, roi_xyxy_original=(0, 0, 10, 10))
    with pytest.raises(EvidenceValidationError, match="ROI coordinate forms"):
        invalid.validate()
```

- [ ] **Step 2: Run schema tests and confirm missing module**

Run: `python -m pytest tests/evidence/test_schema.py -v`

Expected: FAIL because evidence schema does not exist.

- [ ] **Step 3: Implement the immutable evidence object**

```python
@dataclass(frozen=True)
class ExternalEvidence:
    evidence_id: str
    sample_id: str
    pathology: Pathology
    probability_raw: float
    probability_calibrated: float
    roi_xyxy_normalized: tuple[float, float, float, float] | None
    roi_xyxy_original: tuple[int, int, int, int] | None
    localization_score: float | None
    reliability: float | None
    source_model: str
    source_revision: str
    checkpoint_sha256: str
    calibrator_id: str
    preprocess_version: str
    metadata: Mapping[str, object]
```

Validate finite probabilities/reliability in `[0,1]`, normalized boxes within `[0,1]` with positive area, original boxes with positive integer area, paired ROI absence, non-empty provenance, and an `evidence_id` derived from canonical payload identity rather than predictions.

- [ ] **Step 4: Add safe internal and exchange serializers**

Internal serialization includes salted sample ID and provenance; exchange serialization contains aggregate source identity only and rejects raw image/subject/study keys.

- [ ] **Step 5: Run schema tests and commit**

Run: `python -m pytest tests/evidence/test_schema.py -v`

Expected: PASS.

```bash
git add src/iira2/evidence tests/evidence/test_schema.py
git commit -m "feat: define external evidence contract"
```

### Task 2: Local-only RAD-DINO backbone adapter

**Files:**
- Create: `src/iira2/evidence/rad_dino.py`
- Create: `src/iira2/evidence/features.py`
- Create: `configs/evidence/rad_dino.yaml`
- Test: `tests/evidence/test_rad_dino.py`
- Test: `tests/evidence/test_features.py`

**Interfaces:**
- Consumes: local snapshot and preprocessed image batch.
- Produces: `RadDinoBackbone.from_local(snapshot, mode)`, `BackboneFeatures(cls, patches, grid_shape)`, and `BackboneFreezeAudit`.

- [ ] **Step 1: Write failing shape and local-only tests using a fake model**

```python
def test_backbone_splits_cls_and_patch_tokens(fake_rad_dino) -> None:
    features = fake_rad_dino.encode(torch.zeros(2, 3, 518, 518))
    assert features.cls.shape == (2, 768)
    assert features.patches.shape[0] == 2
    assert features.patches.shape[-1] == 768


def test_rad_dino_loader_is_local_only(fake_transformers, snapshot) -> None:
    RadDinoBackbone.from_local(snapshot, mode="frozen")
    assert fake_transformers.kwargs["local_files_only"] is True
```

- [ ] **Step 2: Run backbone tests and confirm missing adapter**

Run: `python -m pytest tests/evidence/test_rad_dino.py tests/evidence/test_features.py -v`

Expected: FAIL because the RAD-DINO adapter is absent.

- [ ] **Step 3: Implement official processor/model loading and feature extraction**

Load config/processor/model from the pinned local snapshot, assert `Dinov2Model`, obtain CLS at token zero, derive the patch grid from processor output and token count, and reject shapes that cannot form the declared grid. Read hidden/patch/image sizes from config, with observed values tested but not hard-coded in business logic.

- [ ] **Step 4: Implement backbone modes and audits**

`frozen` disables all gradients; `last_block` enables only the final encoder block plus layer norm; `full` enables the whole backbone and is non-P0. The audit lists trainable names/counts and fails if mode does not match. Use BF16 only after the target device probe says it is supported.

- [ ] **Step 5: Run backbone tests and commit**

Run: `python -m pytest tests/evidence/test_rad_dino.py tests/evidence/test_features.py -v`

Expected: PASS with fakes.

```bash
git add src/iira2/evidence/rad_dino.py src/iira2/evidence/features.py configs/evidence/rad_dino.yaml tests/evidence
git commit -m "feat: extract RAD-DINO medical image features"
```

### Task 3: Multi-label disease classifier and training objective

**Files:**
- Create: `src/iira2/evidence/classifier.py`
- Create: `src/iira2/evidence/losses.py`
- Create: `src/iira2/evidence/batches.py`
- Test: `tests/evidence/test_classifier.py`
- Test: `tests/evidence/test_losses.py`

**Interfaces:**
- Consumes: `BackboneFeatures.cls`, eight-pathology labels, and observed-label mask.
- Produces: `DiseaseClassifier`, `ClassifierOutput(raw_logits)`, and `masked_bce_with_logits(output, labels, observed_mask)`.

- [ ] **Step 1: Write failing classifier and uncertain-mask tests**

```python
def test_classifier_emits_one_logit_per_pathology() -> None:
    model = DiseaseClassifier(hidden_size=16, pathology_count=8, head="linear")
    assert model(torch.zeros(4, 16)).raw_logits.shape == (4, 8)


def test_unobserved_label_has_zero_gradient() -> None:
    logits = torch.tensor([[0.1, 0.2]], requires_grad=True)
    loss = masked_bce_with_logits(logits, torch.tensor([[1.0, 0.0]]), torch.tensor([[True, False]]))
    loss.backward()
    assert logits.grad[0, 1].item() == 0.0
```

- [ ] **Step 2: Run classifier tests and confirm missing modules**

Run: `python -m pytest tests/evidence/test_classifier.py tests/evidence/test_losses.py -v`

Expected: FAIL because classifier/loss code does not exist.

- [ ] **Step 3: Implement linear and one-hidden-layer MLP heads**

The linear head is P0. The MLP uses `LayerNorm -> Linear -> GELU -> Dropout -> Linear`; dimensions and dropout come from config. Return raw logits only. Probability calibration happens later and no localization tensor enters the classifier probability.

- [ ] **Step 4: Implement masked loss and class weighting**

Normalize over observed entries only, reject an all-unobserved batch, and compute optional positive weights from training data only. Persist counts and weight hash in the checkpoint metadata.

- [ ] **Step 5: Run classifier tests and commit**

Run: `python -m pytest tests/evidence/test_classifier.py tests/evidence/test_losses.py -v`

Expected: PASS.

```bash
git add src/iira2/evidence/classifier.py src/iira2/evidence/losses.py src/iira2/evidence/batches.py tests/evidence
git commit -m "feat: train KBCSv2 disease classifier"
```

### Task 4: Pathology-conditioned localizer and ROI projection

**Files:**
- Create: `src/iira2/evidence/localizer.py`
- Create: `src/iira2/evidence/roi.py`
- Create: `src/iira2/evidence/localization_loss.py`
- Test: `tests/evidence/test_localizer.py`
- Test: `tests/evidence/test_roi.py`

**Interfaces:**
- Consumes: patch features, pathology indices, optional Ext-ILS masks/boxes, and `CoordinateTransform`.
- Produces: `LocalizationOutput(heatmap, score, box_normalized)`, `PathologyConditionedLocalizer`, and `project_roi_to_original(...)`.

- [ ] **Step 1: Write failing conditioned-output and no-ROI tests**

```python
def test_localizer_changes_with_pathology_embedding(localizer, patch_features) -> None:
    edema = localizer(patch_features, torch.tensor([3])).heatmap
    effusion = localizer(patch_features, torch.tensor([4])).heatmap
    assert not torch.equal(edema, effusion)


def test_low_confidence_localization_returns_no_box() -> None:
    result = heatmap_to_roi(torch.zeros(37, 37), threshold=0.5)
    assert result.box_normalized is None
```

- [ ] **Step 2: Run localizer tests and confirm missing modules**

Run: `python -m pytest tests/evidence/test_localizer.py tests/evidence/test_roi.py -v`

Expected: FAIL because localizer/ROI code does not exist.

- [ ] **Step 3: Implement patch decoder and box-head option**

The P0 patch decoder projects patch tokens and a learned pathology embedding into a grid logit map. The box-head option predicts normalized center/size plus presence. Keep map confidence/localization score separate from disease logits. Loss combines masked BCE/Dice for masks and smooth-L1/GIoU for boxes only when the corresponding annotation exists.

- [ ] **Step 4: Implement deterministic ROI extraction and projection**

Threshold the sigmoid heatmap, select the largest connected component, reject regions below configured area/confidence, form an exclusive-max normalized box, and project through the recorded coordinate transform. Return both ROI fields as `None` when invalid.

- [ ] **Step 5: Add random-baseline metric fixture and run tests**

Assert a centered synthetic lesion beats same-area random pointing accuracy under a fixed seed, and exact box projection is within one original pixel.

Run: `python -m pytest tests/evidence/test_localizer.py tests/evidence/test_roi.py -v`

Expected: PASS.

- [ ] **Step 6: Commit localization**

```bash
git add src/iira2/evidence/localizer.py src/iira2/evidence/roi.py src/iira2/evidence/localization_loss.py tests/evidence
git commit -m "feat: localize pathology evidence"
```

### Task 5: KBCSv2 calibrator binding and leakage guards

**Files:**
- Create: `src/iira2/evidence/calibration.py`
- Test: `tests/evidence/test_kbcs_calibration.py`

**Interfaces:**
- Consumes: raw logits/probabilities, labels, split identity, pathology.
- Produces: `fit_kbcs_calibrators(...) -> Mapping[Pathology, CalibratorArtifact]` and calibrated classifier outputs bound to a KBCSv2 checkpoint.

- [ ] **Step 1: Write failing leakage and range tests**

```python
def test_kbcs_calibrator_rejects_locked_test_split() -> None:
    with pytest.raises(CalibrationLeakageError):
        fit_kbcs_calibrators(test_predictions(), method="platt", split_ref=locked_test_ref())


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic", "beta"])
def test_kbcs_artifact_is_bound_to_checkpoint(method, kbcs_calibration_fixture) -> None:
    artifact = fit_kbcs_calibrators(kbcs_calibration_fixture, method, calibration_ref())[Pathology.EDEMA]
    p = apply_calibrator(artifact, kbcs_calibration_fixture.raw_logits)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert artifact.source == "kbcs_v2"
    assert artifact.model_hash == kbcs_calibration_fixture.checkpoint_hash
```

- [ ] **Step 2: Run calibration tests and confirm missing package**

Run: `python -m pytest tests/evidence/test_kbcs_calibration.py -v`

Expected: FAIL because the KBCSv2 calibration binding does not exist.

- [ ] **Step 3: Implement raw, temperature, Platt, isotonic, and beta methods**

Use the generic serializable calibration package from the data/Qwen plan. Fit per pathology on the KBCSv2 calibration subset and bind every artifact to `source=kbcs_v2`, checkpoint hash, ontology hash, split hash, and raw-logit input representation.

- [ ] **Step 4: Add double-calibration and degeneracy guards**

Reject already-calibrated inputs, single-class fit sets, test splits, ontology mismatch, checkpoint mismatch, and Qwen-sourced calibration artifacts. Candidate selection uses calibration-subset cross-validation and never the MIMIC/VinDr test sets.

- [ ] **Step 5: Run calibration tests and commit**

Run: `python -m pytest tests/calibration tests/evidence/test_kbcs_calibration.py -v`

Expected: PASS.

```bash
git add src/iira2/evidence/calibration.py tests/evidence/test_kbcs_calibration.py
git commit -m "feat: calibrate KBCSv2 probabilities"
```

### Task 6: Cross-fitted evidence reliability estimator

**Files:**
- Create: `src/iira2/evidence/reliability.py`
- Create: `src/iira2/evidence/crossfit.py`
- Test: `tests/evidence/test_reliability.py`
- Test: `tests/evidence/test_crossfit.py`

**Interfaces:**
- Consumes: held-out calibrated predictions, labels, entropy, localization score, missing-ROI flag, pathology.
- Produces: `ReliabilityEstimator`, `ReliabilityFeatures`, and `crossfit_reliability(records, folds, seed) -> ReliabilityArtifact`.

- [ ] **Step 1: Write failing no-self-fit and missing-ROI tests**

```python
def test_crossfit_prediction_never_uses_its_own_fold(crossfit_result) -> None:
    assert all(row.sample_fold not in row.fit_folds for row in crossfit_result.rows)


def test_missing_roi_is_an_explicit_feature() -> None:
    features = ReliabilityFeatures.from_values("Edema", 0.8, 0.5, None)
    assert features.missing_roi == 1.0
```

- [ ] **Step 2: Run reliability tests and confirm missing implementation**

Run: `python -m pytest tests/evidence/test_reliability.py tests/evidence/test_crossfit.py -v`

Expected: FAIL because reliability modules do not exist.

- [ ] **Step 3: Implement correctness target and cross-fitting**

Define correctness using the development-locked decision threshold for the pathology, not test outcomes. Features are pathology one-hot, calibrated margin, predictive entropy, localization score with missing indicator, and no ground-truth-derived field. Use subject-grouped folds and regularized logistic regression; each emitted training prediction comes from a model not fit on that subject/fold.

- [ ] **Step 4: Fit final artifact after out-of-fold evaluation**

Report OOF Brier/ECE/AUROC, then fit one final estimator on the whole calibration subset for future samples. Store fold assignments/hash, feature schema/version, thresholds, ontology, classifier checkpoint, calibrator ID, and artifact hash.

- [ ] **Step 5: Run reliability tests and commit**

Run: `python -m pytest tests/evidence/test_reliability.py tests/evidence/test_crossfit.py -v`

Expected: PASS.

```bash
git add src/iira2/evidence/reliability.py src/iira2/evidence/crossfit.py tests/evidence
git commit -m "feat: estimate cross-fitted evidence reliability"
```

### Task 7: KBCSv2 training, model selection, and freeze gate

**Files:**
- Create: `src/iira2/evidence/kbcs_v2.py`
- Create: `src/iira2/evidence/training.py`
- Create: `src/iira2/evidence/checkpoint.py`
- Create: `src/iira2/cli/train_kbcs.py`
- Create: `configs/experiment/kbcs_frozen_heads.yaml`
- Create: `configs/experiment/kbcs_last_block.yaml`
- Test: `tests/evidence/test_kbcs_training.py`
- Test: `tests/evidence/test_checkpoint_freeze.py`

**Interfaces:**
- Consumes: backbone, classifier, localizer, training/model-selection loaders.
- Produces: `KBCSv2`, `KBCSCheckpoint`, `train_kbcs(config)`, and `select_kbcs_checkpoint(candidates, metrics) -> KBCSCheckpoint`.

- [ ] **Step 1: Write failing gradient-boundary and selection tests**

```python
def test_frozen_mode_updates_heads_not_backbone(kbcs_model, training_batch) -> None:
    audit = one_training_update(kbcs_model, training_batch)
    assert audit.classifier_gradient_norm > 0
    assert audit.backbone_gradient_norm == 0


def test_selection_never_accepts_test_metrics(candidate_metrics) -> None:
    with pytest.raises(LockedSplitError):
        select_kbcs_checkpoint(candidate_metrics, metric_split="test")
```

- [ ] **Step 2: Run training tests and confirm missing orchestration**

Run: `python -m pytest tests/evidence/test_kbcs_training.py tests/evidence/test_checkpoint_freeze.py -v`

Expected: FAIL because KBCSv2 orchestration does not exist.

- [ ] **Step 3: Implement multi-task training and finite-step audit**

Combine observed-label classifier loss and annotation-available localization loss with configured weights. Use NPU-safe PyTorch operations, BF16 autocast only when probed, FP32 loss reduction, gradient clipping, deterministic seed capture, checkpoint/resume of model/optimizer/scheduler/RNG, and compact epoch metrics.

- [ ] **Step 4: Implement two-candidate model selection**

Train frozen-backbone heads first, then optionally final-block tuning. Compare only on MIMIC model-selection validation using declared Brier/AUROC plus localization against same-area random. Select exactly one checkpoint under a predeclared lexicographic rule, then fit calibration/reliability on the disjoint calibration subset.

- [ ] **Step 5: Implement immutable freeze seal**

After selection, write model/calibrator/reliability hashes and set all parameters to evaluation/frozen mode. `FreezeSeal.verify()` rehashes files and asserts no trainable parameter. Any subsequent mutation invalidates evidence generation.

- [ ] **Step 6: Run training tests and commit**

Run: `python -m pytest tests/evidence/test_kbcs_training.py tests/evidence/test_checkpoint_freeze.py -v`

Expected: PASS on synthetic CPU batches.

```bash
git add src/iira2/evidence/kbcs_v2.py src/iira2/evidence/training.py src/iira2/evidence/checkpoint.py src/iira2/cli/train_kbcs.py configs/experiment/kbcs_frozen_heads.yaml configs/experiment/kbcs_last_block.yaml tests/evidence
git commit -m "feat: train and freeze KBCSv2"
```

### Task 8: Evidence cache, Gate 2, and target-device smoke

**Files:**
- Create: `src/iira2/evidence/cache.py`
- Create: `src/iira2/evidence/evaluate.py`
- Create: `src/iira2/cli/cache_evidence.py`
- Create: `src/iira2/cli/evaluate_kbcs.py`
- Create: `configs/experiment/kbcs_smoke.yaml`
- Test: `tests/evidence/test_cache.py`
- Test: `tests/evidence/test_gate2.py`
- Test: `tests/npu/test_kbcs_npu.py`

**Interfaces:**
- Consumes: frozen `KBCSv2`, `FreezeSeal`, calibrated/reliability artifacts, sample records.
- Produces: `EvidenceCache`, `KBCSGateResult`, `evaluate_gate2(...) -> KBCSGateResult`, and immutable `ExternalEvidence` rows.

```python
@dataclass(frozen=True)
class KBCSGateResult:
    status: Literal["PASS", "FAIL", "BLOCKED"]
    metrics: Mapping[str, float | None]
    reasons: tuple[str, ...]
    artifact_refs: tuple[str, ...]
```

- [ ] **Step 1: Write failing cache identity and mutation tests**

```python
def test_evidence_cache_key_includes_all_artifact_hashes(cache_key) -> None:
    assert cache_key.digest != replace(cache_key, calibrator_id="other").digest


def test_cache_generation_rejects_unsealed_model(unsealed_kbcs, records) -> None:
    with pytest.raises(FreezeSealError):
        build_evidence_cache(unsealed_kbcs, records)
```

- [ ] **Step 2: Run cache/gate tests and confirm missing code**

Run: `python -m pytest tests/evidence/test_cache.py tests/evidence/test_gate2.py -v`

Expected: FAIL because evidence cache/evaluation code does not exist.

- [ ] **Step 3: Implement content-addressed Parquet evidence cache**

Key every row by dataset/version/split/sample/pathology/image hash, preprocessing version, checkpoint, calibrator, reliability artifact, and ontology hash. Write sharded temporary files and atomically finalize with row count, schema hash, and file hashes. Resume skips only hash-identical complete shards.

- [ ] **Step 4: Implement Gate 2 evaluation**

On locked internal MIMIC test, compare classification against prevalence using paired bootstrap Brier or AUROC CI, verify calibration improves Brier or NLL without significant degradation of the other, compare localization to same-area random, and validate all schema/provenance/coordinate fields. Return `PASS`, `FAIL`, or `BLOCKED`; negative results are retained.

- [ ] **Step 5: Run host suite and one target NPU inference/backward smoke**

Run: `python -m pytest tests/evidence tests/calibration -v`

Run on target: `python -m pytest tests/npu/test_kbcs_npu.py -v`

Expected: host suite PASS. Target smoke verifies one image inference and one head-training update; it is the only evidence for NPU readiness.

- [ ] **Step 6: Commit KBCSv2 cache and gate**

```bash
git add src/iira2/evidence src/iira2/cli/cache_evidence.py src/iira2/cli/evaluate_kbcs.py configs/experiment/kbcs_smoke.yaml tests/evidence tests/npu/test_kbcs_npu.py
git commit -m "feat: cache and gate frozen KBCSv2 evidence"
```

### Task 9: Evidence-conditioned frozen Qwen revision cache

**Files:**
- Create: `src/iira2/evidence/revision_cache.py`
- Modify: `src/iira2/cli/cache_agent.py`
- Create: `configs/experiment/qwen_kbcs_revision_cache.yaml`
- Test: `tests/evidence/test_revision_cache.py`
- Test: `tests/evidence/test_revision_prompt_boundary.py`

**Interfaces:**
- Consumes: frozen evidence cache, frozen Qwen adapter, Qwen calibrators, image records, and evidence-conditioned prompt.
- Produces: `build_revision_cache(...) -> AgentResponseCache` with one content-addressed revised response per exact evidence row.

- [ ] **Step 1: Write failing identity and no-label tests**

```python
def test_revision_cache_key_changes_with_evidence_identity(sample, evidence) -> None:
    first = revision_key(sample, evidence, model_revision="q", prompt_hash="p", calibrator_id="c")
    second = revision_key(sample, replace(evidence, evidence_id="other"),
                          model_revision="q", prompt_hash="p", calibrator_id="c")
    assert first.digest != second.digest


def test_revision_prompt_has_no_ground_truth(revision_prompt_payload) -> None:
    assert "label" not in revision_prompt_payload
    assert "correct" not in revision_prompt_payload
    assert "test_outcome" not in revision_prompt_payload
```

- [ ] **Step 2: Run revision-cache tests and confirm missing implementation**

Run: `python -m pytest tests/evidence/test_revision_cache.py tests/evidence/test_revision_prompt_boundary.py -v`

Expected: FAIL because revision-cache orchestration does not exist.

- [ ] **Step 3: Implement exact evidence-conditioned revisions**

First generate raw evidence-conditioned Qwen scores on the MIMIC calibration subset and fit a separate `source=qwen38_revision` calibrator without using MIMIC test or VinDr. Freeze that artifact. Then, for each sample/pathology, load the original image and exact frozen `ExternalEvidence`; supply calibrated disease probability, valid ROI crop when present, localization score, reliability, and provenance to `Qwen38Adapter.revise_with_evidence`. Do not supply labels, test correctness, unsupported whole-image pseudo-ROI, or hidden reasoning. Apply the frozen Qwen revision calibrator and store raw/calibrated structured response.

- [ ] **Step 4: Enforce train/evaluation evidence-source parity**

The revision cache identity includes dataset/split, image/preprocessing hash, evidence cache ID, KBCSv2/checkpoint/calibrator/reliability hashes, Qwen revision, prompt/candidate hashes, and Qwen calibrator ID. Macro training and evaluation configs must reference the same source kind `kbcs_v2`; a silent source switch raises `CacheIdentityError`.

- [ ] **Step 5: Add resumability and target NPU smoke**

Resume only complete hash-identical shards, keep failures explicit, and print only progress/throughput/peak memory/failure count. Run one real image/evidence revision on the target 2-way profile before scheduling the full 4-way cache build.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/evidence/test_revision_cache.py tests/evidence/test_revision_prompt_boundary.py tests/agents/test_response_cache.py -v`

Expected: PASS with fake Qwen/evidence objects.

```bash
git add src/iira2/evidence/revision_cache.py src/iira2/cli/cache_agent.py configs/experiment/qwen_kbcs_revision_cache.yaml tests/evidence
git commit -m "feat: cache evidence-conditioned Qwen revisions"
```
