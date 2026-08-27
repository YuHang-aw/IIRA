# IIRA 2.0 Data and Frozen Qwen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement compliant dataset intake, canonical pathology records, leakage audits, deterministic image preprocessing, the official frozen Qwen3.8 adapter, constrained probability scoring, and replayable response caches.

**Architecture:** Dataset adapters normalize metadata into immutable internal records while preserving source-specific identifiers only inside the container. The Qwen adapter depends on a small multimodal-agent protocol and injects model/processor objects for host tests; production loads the pinned local snapshot only. Direct and evidence-conditioned responses are cached separately and content-addressed.

**Tech Stack:** Python 3.11, pandas, PyArrow, pydicom, Pillow, NumPy, scikit-image, Hugging Face Transformers, safetensors, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-iira2-design.md`

## Global Constraints

- Execute `2026-08-27-iira2-foundation-assets-plan.md` first.
- Restricted PhysioNet/Stanford data is downloaded only by the user through official access and stays under `D:\trustModel\datasets\restricted`.
- VinDr test is locked against training, calibration, threshold selection, and model selection.
- MIMIC uses official splits; train/validation/test subject and study overlap must be asserted.
- The core ontology is exactly eight pathologies from the approved design.
- Qwen parameters remain frozen in the main experiment and only local files are loaded.
- No free-text chain-of-thought or free-text percentage is stored as the main probability.
- RAD-DINO pretraining exposure is recorded; cross-dataset claims are task-specific, not foundation-data unseen.
- Every implementation task follows red-green-refactor and ends with an isolated commit.

---

### Task 1: Restricted dataset access manifests and audit CLI

**Files:**
- Create: `manifests/datasets.yaml`
- Create: `src/iira2/data/access.py`
- Create: `src/iira2/data/audit_status.py`
- Create: `src/iira2/data/__init__.py`
- Create: `src/iira2/cli/data.py`
- Create: `scripts/print_restricted_download_instructions.ps1`
- Create: `docs/DATASETS.md`
- Create: `docs/AUTH_REQUIRED.md`
- Test: `tests/data/test_access_manifest.py`

**Interfaces:**
- Consumes: declared official dataset root and expected file patterns.
- Produces: `DatasetSpec`, `DatasetAudit`, and `audit_dataset(spec, root) -> DatasetAudit`.

- [ ] **Step 1: Write failing tests for blocked and partial datasets**

```python
def test_missing_credentialed_dataset_is_blocked(dataset_spec, tmp_path) -> None:
    audit = audit_dataset(dataset_spec, tmp_path)
    assert audit.state.value == "BLOCKED_BY_TERMS"
    assert audit.required_user_action


def test_present_subset_is_partial(dataset_spec, tmp_path) -> None:
    (tmp_path / dataset_spec.required_files[0]).parent.mkdir(parents=True)
    (tmp_path / dataset_spec.required_files[0]).touch()
    audit = audit_dataset(dataset_spec, tmp_path)
    assert audit.state.value == "PARTIAL"
```

- [ ] **Step 2: Run the test and confirm missing data-access code**

Run: `python -m pytest tests/data/test_access_manifest.py -v`

Expected: FAIL because the data access module does not exist.

- [ ] **Step 3: Implement official-access specifications**

Declare MIMIC-CXR-JPG 2.1.0, MIMIC-CXR reports 2.1.0, VinDr-CXR 1.0.0, and MIMIC-CXR-Ext-ILS 1.0.0 with official landing URL, access class, version, expected root, required metadata files, optional bulk image groups, and license/DUA note. Do not include credentials, cookies, signed URLs, or third-party mirrors.

`audit_dataset` returns exact missing relative paths and one of `MISSING`, `PARTIAL`, `VERIFIED`, or `BLOCKED_BY_TERMS`; it hashes metadata files but does not hash terabytes of images until explicitly requested.

- [ ] **Step 4: Implement instruction-only download helper**

The PowerShell script prints official URLs, target directories, a resumable official-client command template, and the post-download audit command. It reads no password, starts no browser session, and never reports success itself.

- [ ] **Step 5: Run the access tests and audit empty roots**

Run: `python -m pytest tests/data/test_access_manifest.py -v`

Run: `python -m iira2.cli.data audit --all --root D:\trustModel`

Expected: tests PASS; unavailable credentialed datasets report `BLOCKED_BY_TERMS` with required user action.

- [ ] **Step 6: Commit dataset access boundaries**

```bash
git add manifests/datasets.yaml src/iira2/data src/iira2/cli/data.py scripts/print_restricted_download_instructions.ps1 docs/DATASETS.md docs/AUTH_REQUIRED.md tests/data/test_access_manifest.py
git commit -m "feat: add compliant dataset intake audits"
```

### Task 2: Canonical ontology and immutable sample records

**Files:**
- Create: `configs/data/pathology_ontology.yaml`
- Create: `src/iira2/data/ontology.py`
- Create: `src/iira2/data/records.py`
- Test: `tests/data/test_ontology.py`
- Test: `tests/data/test_records.py`

**Interfaces:**
- Consumes: source labels and uncertain policy.
- Produces: `Pathology`, `LabelValue`, `SampleRecord`, `load_ontology(path)`, and `map_label(source, raw_label, raw_value, policy)`.

- [ ] **Step 1: Write failing ontology tests**

```python
def test_core_ontology_has_exactly_eight_labels(ontology) -> None:
    assert [p.value for p in ontology.pathologies] == [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Pleural Effusion", "Pneumonia", "Lung Opacity", "Pneumothorax",
    ]


def test_uncertain_ignore_is_not_negative(ontology) -> None:
    assert ontology.map_mimic("Edema", -1, "ignore") is None
```

- [ ] **Step 2: Run ontology tests and confirm missing types**

Run: `python -m pytest tests/data/test_ontology.py tests/data/test_records.py -v`

Expected: FAIL because ontology and record modules do not exist.

- [ ] **Step 3: Implement canonical labels and records**

```python
class Pathology(str, Enum):
    ATELECTASIS = "Atelectasis"
    CARDIOMEGALY = "Cardiomegaly"
    CONSOLIDATION = "Consolidation"
    EDEMA = "Edema"
    PLEURAL_EFFUSION = "Pleural Effusion"
    PNEUMONIA = "Pneumonia"
    LUNG_OPACITY = "Lung Opacity"
    PNEUMOTHORAX = "Pneumothorax"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset: str
    split: str
    image_path: Path
    pathology: Pathology
    label: int
    subject_key: str | None
    study_key: str | None
    source_image_key: str
    metadata: Mapping[str, object]
```

Create `sample_id` as a salted SHA256-derived stable ID from dataset/version/source key. Preserve raw identifiers only in internal fields; exchange serializers must not accept a `SampleRecord` directly. Implement `ignore`, `u_zero`, and `u_one` as explicit policies and hash the resolved ontology plus mapping.

- [ ] **Step 4: Add mapping completeness tests**

Assert each source label maps to one canonical pathology or an explicit excluded reason, conflicting aliases fail, and changing uncertain policy changes the ontology hash.

- [ ] **Step 5: Run ontology and record tests**

Run: `python -m pytest tests/data/test_ontology.py tests/data/test_records.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the canonical data contract**

```bash
git add configs/data/pathology_ontology.yaml src/iira2/data/ontology.py src/iira2/data/records.py tests/data/test_ontology.py tests/data/test_records.py
git commit -m "feat: define canonical pathology records"
```

### Task 3: MIMIC metadata loader and split-leakage assertions

**Files:**
- Create: `src/iira2/data/mimic.py`
- Create: `src/iira2/data/splits.py`
- Test: `tests/data/test_mimic.py`
- Test: `tests/data/test_mimic_split.py`
- Test fixture: `tests/fixtures/mimic/metadata.csv`
- Test fixture: `tests/fixtures/mimic/split.csv`
- Test fixture: `tests/fixtures/mimic/labels.csv`

**Interfaces:**
- Consumes: official MIMIC metadata, split, and label CSV files.
- Produces: `iter_mimic_records(root, ontology, policy) -> Iterator[SampleRecord]` and `audit_mimic_splits(records) -> SplitAudit`.

- [ ] **Step 1: Write failing join and leakage tests**

```python
def test_mimic_loader_joins_on_subject_and_study(fixture_root, ontology) -> None:
    rows = list(iter_mimic_records(fixture_root, ontology, "ignore"))
    assert {(r.subject_key, r.study_key, r.split) for r in rows} == {
        ("10", "100", "train"), ("20", "200", "validate")
    }


def test_subject_overlap_fails_closed(records_with_overlap) -> None:
    with pytest.raises(SplitLeakageError, match="subject"):
        audit_mimic_splits(records_with_overlap)
```

- [ ] **Step 2: Run MIMIC tests and confirm missing loader failure**

Run: `python -m pytest tests/data/test_mimic.py tests/data/test_mimic_split.py -v`

Expected: FAIL because the loader is absent.

- [ ] **Step 3: Implement typed CSV loading and official split preservation**

Read IDs as strings, validate required columns, join with cardinality checks, resolve the official JPG path, emit one record per non-ignored pathology label, and keep train/validate/test names unchanged. Fail on duplicate metadata keys, missing joins, non-binary resolved labels, and absent image files in strict mode.

- [ ] **Step 4: Implement split and validation partition audits**

Assert no subject or study crosses official splits. Derive model-selection and calibration subsets only from official validation using a seeded subject-level hash partition. Persist partition seed, membership hash, and counts; test remains untouched.

- [ ] **Step 5: Run MIMIC tests**

Run: `python -m pytest tests/data/test_mimic.py tests/data/test_mimic_split.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the MIMIC loader**

```bash
git add src/iira2/data/mimic.py src/iira2/data/splits.py tests/data/test_mimic.py tests/data/test_mimic_split.py tests/fixtures/mimic
git commit -m "feat: load and audit MIMIC splits"
```

### Task 4: VinDr aggregation and deterministic DICOM conversion

**Files:**
- Create: `src/iira2/data/vindr.py`
- Create: `src/iira2/data/dicom.py`
- Create: `src/iira2/data/coordinates.py`
- Create: `src/iira2/cli/prepare_vindr.py`
- Test: `tests/data/test_vindr.py`
- Test: `tests/data/test_dicom.py`
- Test: `tests/data/test_coordinates.py`
- Test fixture: `tests/fixtures/vindr/annotations.csv`

**Interfaces:**
- Consumes: official VinDr annotation/image-label CSVs and DICOM files.
- Produces: `VinDrImageRecord`, `aggregate_vindr_annotations(frame)`, `convert_dicom(source, destination, config) -> DerivedImage`, and `CoordinateTransform`.

- [ ] **Step 1: Write failing radiologist aggregation and coordinate tests**

```python
def test_vindr_rows_are_aggregated_before_split(annotation_frame) -> None:
    records = aggregate_vindr_annotations(annotation_frame)
    assert len(records) == annotation_frame["image_id"].nunique()


def test_coordinate_round_trip_is_subpixel() -> None:
    transform = CoordinateTransform.letterbox((1024, 2048), (518, 518))
    box = (100.0, 200.0, 700.0, 900.0)
    restored = transform.to_original(transform.to_derived(box))
    assert np.max(np.abs(np.asarray(restored) - np.asarray(box))) < 1.0
```

- [ ] **Step 2: Run VinDr/DICOM tests and confirm missing implementation**

Run: `python -m pytest tests/data/test_vindr.py tests/data/test_dicom.py tests/data/test_coordinates.py -v`

Expected: FAIL because VinDr and DICOM modules do not exist.

- [ ] **Step 3: Implement image-level aggregation and locked split rules**

Aggregate all radiologist rows by `image_id`, preserve each box with class and reader ID internally, combine image-level labels under an explicit consensus policy, and only then partition official train if a development calibration split is needed. The official test flag makes all fitting APIs raise `LockedSplitError`.

- [ ] **Step 4: Implement deterministic DICOM preprocessing**

Handle MONOCHROME1 inversion, rescale slope/intercept, configured windowing, finite-value checks, aspect-preserving resize/padding, 16-bit to normalized float conversion, and output PNG without overwriting source. Store source/derived hashes, shapes, photometric interpretation, window, and transform version in an internal sidecar.

- [ ] **Step 5: Add checkerboard, corner, and immutable-source tests**

Create synthetic DICOM fixtures at test time. Assert corner orientation, checkerboard preservation, coordinate round trip, identical output hashes across reruns, and unchanged source hash.

- [ ] **Step 6: Run VinDr/DICOM tests**

Run: `python -m pytest tests/data/test_vindr.py tests/data/test_dicom.py tests/data/test_coordinates.py -v`

Expected: PASS.

- [ ] **Step 7: Commit VinDr preprocessing**

```bash
git add src/iira2/data/vindr.py src/iira2/data/dicom.py src/iira2/data/coordinates.py src/iira2/cli/prepare_vindr.py tests/data tests/fixtures/vindr
git commit -m "feat: prepare and audit VinDr images"
```

### Task 5: Cross-dataset duplicate and provenance audit

**Files:**
- Create: `src/iira2/data/duplicates.py`
- Create: `src/iira2/data/provenance.py`
- Test: `tests/data/test_duplicates.py`
- Test: `tests/data/test_provenance.py`

**Interfaces:**
- Consumes: derived-image records from MIMIC and VinDr.
- Produces: `DuplicateAudit`, `ProvenanceRecord`, and `audit_duplicates(records) -> DuplicateAudit`.

- [ ] **Step 1: Write failing exact/perceptual duplicate tests**

```python
def test_exact_duplicate_across_splits_is_critical(records_with_same_sha) -> None:
    audit = audit_duplicates(records_with_same_sha)
    assert audit.critical_count == 1


def test_perceptual_match_is_reported_not_silently_removed(near_duplicate_records) -> None:
    audit = audit_duplicates(near_duplicate_records)
    assert audit.review_count == 1
```

- [ ] **Step 2: Run duplicate tests and confirm missing module**

Run: `python -m pytest tests/data/test_duplicates.py tests/data/test_provenance.py -v`

Expected: FAIL because duplicate/provenance code does not exist.

- [ ] **Step 3: Implement exact SHA256, perceptual hash, and source-key audits**

Report within-dataset and cross-dataset groups separately. Exact cross-split duplicates fail the gate. Perceptual matches carry distance, source datasets, splits, and a required review state; they are not automatically dropped. Persist only salted IDs in compact summaries.

- [ ] **Step 4: Record RAD-DINO exposure boundary**

Provenance metadata must state that RAD-DINO foundation pretraining included MIMIC-CXR, CheXpert, and NIH-CXR, while task-specific KBCSv2 heads/calibration are developed only on declared splits.

- [ ] **Step 5: Run duplicate/provenance tests and commit**

Run: `python -m pytest tests/data/test_duplicates.py tests/data/test_provenance.py -v`

Expected: PASS.

```bash
git add src/iira2/data/duplicates.py src/iira2/data/provenance.py tests/data/test_duplicates.py tests/data/test_provenance.py
git commit -m "feat: audit image duplication and provenance"
```

### Task 6: Multimodal agent protocol and official local Qwen loader

**Files:**
- Create: `src/iira2/agents/base.py`
- Create: `src/iira2/agents/qwen38.py`
- Create: `src/iira2/agents/freeze.py`
- Create: `src/iira2/agents/__init__.py`
- Create: `docs/QWEN38_MIGRATION.md`
- Test: `tests/agents/test_qwen38_adapter.py`
- Test: `tests/agents/test_frozen_qwen.py`

**Interfaces:**
- Consumes: local Qwen snapshot, image, pathology, optional evidence `Mapping[str, object]`.
- Produces: `MultimodalAgentAdapter`, `PreparedAgentInput`, `AgentResponse`, and `Qwen38Adapter.from_local(snapshot, device_plan)`.

- [ ] **Step 1: Write failing local-only and freeze tests with fakes**

```python
def test_qwen_loader_pins_local_files_and_revision(fake_transformers, snapshot) -> None:
    Qwen38Adapter.from_local(snapshot, device_plan="cpu")
    assert fake_transformers.kwargs["local_files_only"] is True


def test_main_qwen_has_no_trainable_parameters(fake_qwen_adapter) -> None:
    assert audit_trainable_parameters(fake_qwen_adapter.model).trainable == 0
```

- [ ] **Step 2: Run adapter tests and confirm missing protocol**

Run: `python -m pytest tests/agents/test_qwen38_adapter.py tests/agents/test_frozen_qwen.py -v`

Expected: FAIL because agent modules do not exist.

- [ ] **Step 3: Define adapter protocol and immutable response**

```python
class MultimodalAgentAdapter(Protocol):
    def prepare_inputs(self, image: Image.Image, pathology: Pathology) -> PreparedAgentInput: ...
    def initial_probability(self, prepared: PreparedAgentInput) -> "AgentProbability": ...
    def direct_decision(self, prepared: PreparedAgentInput) -> "AgentResponse": ...
    def revise_with_evidence(self, prepared: PreparedAgentInput, evidence: Mapping[str, object]) -> "AgentResponse": ...
    def detached_state_features(self, response: "AgentResponse") -> np.ndarray: ...
```

`AgentResponse` stores final structured fields, normalized probability, candidate scores, prompt/template hash, model revision, and detached state features. It has no reasoning-text field.

- [ ] **Step 4: Implement official architecture discovery**

Load `AutoConfig` and assert `model_type == "qwen3_5"` and that `Qwen3_5ForConditionalGeneration` appears in `architectures`. Load the official processor and model class supported by the pinned Transformers build; never use the uncertain `AutoModelForMultimodalLM`. Read hidden sizes, vision config, image tokens, and module names from the loaded config/model rather than business-code constants. Set every parameter `requires_grad_(False)` and `eval()`.

- [ ] **Step 5: Add local config smoke and unsupported-version failure tests**

Test exact revision metadata, processor/chat-template availability, non-thinking template options, image input construction, BF16 selection on NPU, and an actionable error when installed Transformers cannot resolve `qwen3_5`.

- [ ] **Step 6: Run adapter tests and commit**

Run: `python -m pytest tests/agents/test_qwen38_adapter.py tests/agents/test_frozen_qwen.py -v`

Expected: PASS with fake model/processor; mark real-weight test separately.

```bash
git add src/iira2/agents docs/QWEN38_MIGRATION.md tests/agents
git commit -m "feat: add frozen official Qwen3.8 adapter"
```

### Task 7: Constrained binary probability scoring

**Files:**
- Create: `src/iira2/agents/probability.py`
- Create: `src/iira2/agents/prompts.py`
- Create: `configs/agent/qwen38.yaml`
- Test: `tests/agents/test_probability.py`
- Test: `tests/agents/test_prompts.py`

**Interfaces:**
- Consumes: prepared multimodal prefix and configured positive/negative candidate token sequences.
- Produces: `AgentProbability` and `score_binary_candidates(model, prepared, candidates) -> AgentProbability`.

- [ ] **Step 1: Write failing numeric scoring tests**

```python
def test_binary_probability_normalizes_sequence_log_probs() -> None:
    result = normalize_candidate_scores(positive=-0.2, negative=-1.2)
    assert result.positive == pytest.approx(0.7310586, rel=1e-6)
    assert result.negative == pytest.approx(0.2689414, rel=1e-6)


def test_non_finite_candidate_score_fails() -> None:
    with pytest.raises(ProbabilityError):
        normalize_candidate_scores(float("nan"), -1.0)
```

- [ ] **Step 2: Run probability tests and confirm missing code**

Run: `python -m pytest tests/agents/test_probability.py tests/agents/test_prompts.py -v`

Expected: FAIL because probability scoring is absent.

- [ ] **Step 3: Implement teacher-forced sequence scoring**

For each candidate, append its token sequence to the identical prepared prefix, run a forward pass, sum token log-probabilities only at candidate positions, and normalize the positive/negative sequence sums with a two-way log-softmax. Assert finite output in `[0,1]`, record both token ID sequences, and return no generated reasoning.

```python
@dataclass(frozen=True)
class AgentProbability:
    probability_raw: float
    probability_calibrated: float | None
    positive_sequence_logp: float
    negative_sequence_logp: float
    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]
    calibrator_id: str | None
```

- [ ] **Step 4: Freeze prompt and template hashes**

Provide separate direct and evidence-conditioned templates. The evidence template accepts probability, valid ROI crop presence, localization score, reliability, source model/revision, and preprocessing version. It must not accept ground truth or test correctness. Store SHA256 of rendered template and candidate configuration.

- [ ] **Step 5: Run scoring tests and commit**

Run: `python -m pytest tests/agents/test_probability.py tests/agents/test_prompts.py -v`

Expected: PASS.

```bash
git add src/iira2/agents/probability.py src/iira2/agents/prompts.py configs/agent/qwen38.yaml tests/agents
git commit -m "feat: score frozen Qwen binary probabilities"
```

### Task 8: Generic calibration and frozen Qwen calibration

**Files:**
- Create: `src/iira2/calibration/base.py`
- Create: `src/iira2/calibration/methods.py`
- Create: `src/iira2/calibration/artifact.py`
- Create: `src/iira2/calibration/__init__.py`
- Create: `src/iira2/agents/calibration.py`
- Test: `tests/calibration/test_methods.py`
- Test: `tests/calibration/test_split_guard.py`
- Test: `tests/agents/test_agent_calibration.py`

**Interfaces:**
- Consumes: raw Qwen candidate probabilities/logits, labels, pathology, and MIMIC calibration split identity.
- Produces: `Calibrator`, `CalibratorArtifact`, `fit_calibrator(...)`, `apply_calibrator(...)`, and `fit_agent_calibrators(...)`.

- [ ] **Step 1: Write failing leakage, range, and separate-source tests**

```python
def test_agent_calibrator_rejects_locked_test_split() -> None:
    with pytest.raises(CalibrationLeakageError):
        fit_agent_calibrators(test_predictions(), method="temperature", split_ref=locked_test_ref())


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic", "beta"])
def test_agent_calibration_is_finite_and_bounded(method, agent_calibration_fixture) -> None:
    artifact = fit_calibrator(method, *agent_calibration_fixture, split_ref=calibration_ref(), source="qwen38")
    p = apply_calibrator(artifact, agent_calibration_fixture[0])
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert artifact.source == "qwen38"
```

- [ ] **Step 2: Run calibration tests and confirm missing package**

Run: `python -m pytest tests/calibration tests/agents/test_agent_calibration.py -v`

Expected: FAIL because calibration modules do not exist.

- [ ] **Step 3: Implement serializable calibration methods**

Implement raw identity, temperature, Platt, isotonic, and beta calibration. Use logits where required, documented numeric epsilon, per-pathology fitting by default, and plain JSON/NumPy parameters rather than executable pickle objects. Each artifact stores source, method, fit split hash, ontology hash, model/checkpoint hash, input representation, library versions, and artifact SHA256.

- [ ] **Step 4: Add degeneracy and double-calibration guards**

Reject locked test splits, single-class fit data, source/model mismatch, already-calibrated inputs, ontology mismatch, and non-finite values. Isotonic clips out-of-bounds explicitly; beta records zero/one epsilon handling.

- [ ] **Step 5: Fit and select the Qwen calibrator on development data only**

Generate raw Qwen probabilities for MIMIC calibration rows, fit each configured candidate, select by development Brier with NLL non-degradation tie-break, and freeze one artifact per pathology. Record all candidate results and the selected artifact IDs; never consult MIMIC test or VinDr.

- [ ] **Step 6: Run calibration tests and commit**

Run: `python -m pytest tests/calibration tests/agents/test_agent_calibration.py -v`

Expected: PASS.

```bash
git add src/iira2/calibration src/iira2/agents/calibration.py tests/calibration tests/agents/test_agent_calibration.py
git commit -m "feat: calibrate frozen Qwen probabilities"
```

### Task 9: Content-addressed Qwen response cache and baseline CLI

**Files:**
- Create: `src/iira2/agents/cache.py`
- Create: `src/iira2/agents/baseline.py`
- Create: `src/iira2/cli/cache_agent.py`
- Create: `src/iira2/cli/evaluate_agent.py`
- Create: `configs/experiment/qwen_direct_smoke.yaml`
- Test: `tests/agents/test_response_cache.py`
- Test: `tests/agents/test_baseline.py`

**Interfaces:**
- Consumes: `SampleRecord`, adapter, prompt hash, model revision, optional evidence ID.
- Produces: `AgentResponseCache`, `AgentCacheKey`, and `evaluate_direct(records, adapter) -> list[AgentResponse]`.

- [ ] **Step 1: Write failing cache-separation tests**

```python
def test_direct_and_evidence_responses_have_distinct_keys(sample, evidence) -> None:
    direct = AgentCacheKey.direct(sample, model_revision="r", prompt_hash="p", calibrator_id="c1")
    revised = AgentCacheKey.revised(sample, evidence.id, model_revision="r", prompt_hash="q", calibrator_id="c2")
    assert direct.digest != revised.digest


def test_cache_rejects_model_revision_mismatch(cache, key, response) -> None:
    cache.put(key, response)
    with pytest.raises(CacheIdentityError):
        cache.get(replace(key, model_revision="other"))
```

- [ ] **Step 2: Run cache tests and confirm missing implementation**

Run: `python -m pytest tests/agents/test_response_cache.py tests/agents/test_baseline.py -v`

Expected: FAIL because response cache code does not exist.

- [ ] **Step 3: Implement immutable Arrow/Parquet cache records**

Key on dataset version, split, salted sample ID, pathology, image hash, model revision, processor/template hash, candidate hash, calibrator ID, and evidence ID for revised responses. Store raw and calibrated probability separately. Use a temporary shard plus atomic rename; reject duplicate keys with different payload hashes. Keep internal source identifiers out of exchange summaries.

- [ ] **Step 4: Implement direct baseline and compact progress output**

The baseline CLI supports `--resume`, deterministic shard order, failure records, and `local_files_only`. Console output contains completed/total, throughput, peak NPU memory, and failure count only. It writes predictions to `internal/` and key metrics to the compact status protocol.

- [ ] **Step 5: Run host tests and real-weight smoke when assets are verified**

Run: `python -m pytest tests/data tests/agents -v`

Run: `python -m iira2.cli.evaluate_agent --config configs/experiment/qwen_direct_smoke.yaml data.limit=1 runtime.profile=qwen_2npu`

Expected: host tests PASS. The real-weight smoke is `COMPLETE` only on the target NPU; otherwise it records `BLOCKED` without changing the host-test result.

- [ ] **Step 6: Commit data/Qwen phase**

```bash
git add src/iira2/agents src/iira2/cli/cache_agent.py src/iira2/cli/evaluate_agent.py tests/agents configs/experiment/qwen_direct_smoke.yaml
git commit -m "feat: cache and evaluate frozen Qwen responses"
```

### Task 10: Qwen SelfCheck baseline without external-evidence claims

**Files:**
- Create: `src/iira2/agents/self_check.py`
- Create: `configs/experiment/qwen_self_check.yaml`
- Test: `tests/agents/test_self_check.py`

**Interfaces:**
- Consumes: original image/pathology, direct response, frozen Qwen adapter, dedicated self-check prompt/calibrator.
- Produces: `SelfCheckEvidence`, `run_self_check(...) -> AgentResponse`, and a response-cache entry tagged `source_kind=self_check`.

- [ ] **Step 1: Write failing provenance and prompt-isolation tests**

```python
def test_self_check_is_not_external_evidence(self_check_result) -> None:
    assert self_check_result.source_kind == "self_check"
    assert self_check_result.is_external is False


def test_self_check_cache_key_differs_from_direct(sample) -> None:
    direct = AgentCacheKey.direct(sample, model_revision="r", prompt_hash="direct", calibrator_id="c1")
    checked = AgentCacheKey.self_check(sample, model_revision="r", prompt_hash="self", calibrator_id="c2")
    assert direct.digest != checked.digest
```

- [ ] **Step 2: Run the test and confirm missing SelfCheck implementation**

Run: `python -m pytest tests/agents/test_self_check.py -v`

Expected: FAIL because `iira2.agents.self_check` does not exist.

- [ ] **Step 3: Implement constrained second-opinion scoring**

Render a dedicated prompt that includes the original image/pathology and the model's structured direct conclusion but no KBCSv2 field, ground truth, or hidden reasoning. Score the same positive/negative candidate sequences, apply a separately fitted development calibrator, and store only structured probabilities/provenance.

- [ ] **Step 4: Enforce experiment and reporting labels**

The experiment arm is `self_check`; its evidence source is `same_qwen`, `is_external=false`, and it cannot satisfy Gate 3 external complementarity. Compact/internal reports must never label it KBCS, verifier, or external grounding.

- [ ] **Step 5: Run agent tests and commit**

Run: `python -m pytest tests/agents -v`

Expected: PASS.

```bash
git add src/iira2/agents/self_check.py configs/experiment/qwen_self_check.yaml tests/agents/test_self_check.py
git commit -m "feat: add Qwen self-check baseline"
```
