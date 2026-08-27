# IIRA 2.0 Evaluation and Ascend Offline Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver common evaluation/statistics/intervention machinery, Go/No-Go experiment orchestration, compact exchange reports, Ascend 910C runtime profiles, and verified network-disabled execution.

**Architecture:** Evaluation consumes immutable prediction/transition tables and produces deterministic internal artifacts plus a small versioned summary. A gate engine controls expensive phases without deleting negative results. Device/runtime adapters isolate NPU and distributed concerns from algorithms; final readiness requires real 910C and network-disabled evidence.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, scikit-learn, scikit-image, PyArrow, Pillow, PyTorch, `torch_npu`, HCCL, pytest, static HTML/JSON.

**Spec:** `docs/superpowers/specs/2026-08-27-iira2-design.md`

## Global Constraints

- Execute all four prior implementation plans first.
- MIMIC test and VinDr test are locked; VinDr runs only after development decisions and gate configuration are frozen.
- Statistical comparisons are paired and include 95% confidence intervals; MIMIC uses subject-level resampling when subject IDs exist, VinDr uses image-level resampling with the limitation stated.
- Abstention is excluded from covered probability loss but reported through coverage/selective risk; it is never encoded as probability `0.5`.
- Clean, stress, and intervention outputs are distinct experiment identities and never overwrite each other.
- The compact exchange report contains only aggregate key metrics and redacted artifact references.
- Real NPU and offline readiness can only be claimed from the target Ascend 910C container.
- Use 2-way feasibility then 4-way Qwen sharding; 7-way is opt-in only after a passing HCCL/device-plan probe.
- Every implementation task follows red-green-refactor and ends with an isolated commit.

---

### Task 1: Probability, policy, and complementarity metrics

**Files:**
- Create: `src/iira2/evaluation/records.py`
- Create: `src/iira2/evaluation/probability.py`
- Create: `src/iira2/evaluation/policy.py`
- Create: `src/iira2/evaluation/complementarity.py`
- Create: `src/iira2/evaluation/__init__.py`
- Test: `tests/evaluation/test_probability.py`
- Test: `tests/evaluation/test_policy.py`
- Test: `tests/evaluation/test_complementarity.py`

**Interfaces:**
- Consumes: immutable covered predictions, labels, actions, query/abstain flags, locked thresholds.
- Produces: `PredictionRecord`, `ProbabilityMetrics`, `PolicyMetrics`, `ComplementarityTable`, and `evaluate_predictions(records, threshold_artifact)`.

- [ ] **Step 1: Write failing hand-computed metric tests**

```python
def test_brier_matches_hand_calculation() -> None:
    metrics = probability_metrics(np.array([0.0, 1.0, 0.5]), np.array([0, 1, 1]), bins=10)
    assert metrics.brier == pytest.approx((0.0 + 0.0 + 0.25) / 3)


def test_abstained_rows_are_not_probability_half() -> None:
    rows = [prediction(probability=0.9, label=1, covered=True),
            prediction(probability=0.1, label=0, covered=False)]
    metrics = evaluate_policy(rows)
    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.covered_count == 1
```

- [ ] **Step 2: Run metric tests and confirm missing evaluation package**

Run: `python -m pytest tests/evaluation/test_probability.py tests/evaluation/test_policy.py tests/evaluation/test_complementarity.py -v`

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Implement probability metrics with explicit undefined states**

Compute Brier, NLL with documented epsilon, fixed-bin ECE, AUROC, AUPRC, sensitivity, and specificity. Thresholds must carry a development split hash. Return `None` plus reason for single-class AUROC/AUPRC or empty covered subsets; never serialize undefined as zero.

- [ ] **Step 4: Implement policy and complementarity metrics**

Compute query rate, abstain rate, coverage, mean belief change, correction rate, harm rate, and evidence adoption under an explicit adoption threshold. Build the four Qwen/KBCSv2 correctness cells per pathology plus disagreement rate, error correlation, and disagreement-conditional accuracy.

- [ ] **Step 5: Run metrics tests and commit**

Run: `python -m pytest tests/evaluation/test_probability.py tests/evaluation/test_policy.py tests/evaluation/test_complementarity.py -v`

Expected: PASS.

```bash
git add src/iira2/evaluation tests/evaluation
git commit -m "feat: evaluate probabilities and macro policy"
```

### Task 2: Selective risk and paired statistical inference

**Files:**
- Create: `src/iira2/evaluation/selective.py`
- Create: `src/iira2/evaluation/bootstrap.py`
- Create: `src/iira2/evaluation/permutation.py`
- Create: `src/iira2/evaluation/multiseed.py`
- Test: `tests/evaluation/test_selective.py`
- Test: `tests/evaluation/test_bootstrap.py`
- Test: `tests/evaluation/test_permutation.py`

**Interfaces:**
- Consumes: paired baseline/method records and resampling unit keys.
- Produces: `RiskCoverageCurve`, `PairedEstimate`, `MultiSeedSummary`, `paired_bootstrap(...)`, and `paired_permutation_test(...)`.

- [ ] **Step 1: Write failing paired-resampling tests**

```python
def test_subject_bootstrap_keeps_subject_rows_together(subject_records) -> None:
    sample = draw_cluster_bootstrap(subject_records, cluster_key="subject_key", seed=7)
    assert every_subject_is_whole(sample)


def test_paired_difference_uses_identical_row_keys(baseline, method) -> None:
    with pytest.raises(PairingError):
        paired_bootstrap(baseline, method[:-1], metric="brier", unit="image", seed=1)
```

- [ ] **Step 2: Run statistics tests and confirm missing modules**

Run: `python -m pytest tests/evaluation/test_selective.py tests/evaluation/test_bootstrap.py tests/evaluation/test_permutation.py -v`

Expected: FAIL because selective/statistical modules do not exist.

- [ ] **Step 3: Implement deterministic risk-coverage curves**

Sort by the declared confidence/selective score with stable salted-sample-ID tie breaking. Report coverage, selective Brier risk, and count at every unique threshold plus fixed coverage grid. Include abstained rows in denominator but not covered loss numerator.

- [ ] **Step 4: Implement paired bootstrap and permutation tests**

Resample subjects for MIMIC and images for VinDr, recompute paired metric differences, return percentile 95% CI, point estimate, resample count, seed, and unit. Use paired sign-flip/permutation for mean per-row loss differences. Validate exact row-key equality and no duplicates before resampling.

- [ ] **Step 5: Implement multi-seed summaries**

Return mean, sample standard deviation, and every seed's value; do not replace paired data uncertainty with seed variance. Store both when applicable.

- [ ] **Step 6: Run statistics tests and commit**

Run: `python -m pytest tests/evaluation/test_selective.py tests/evaluation/test_bootstrap.py tests/evaluation/test_permutation.py -v`

Expected: PASS.

```bash
git add src/iira2/evaluation/selective.py src/iira2/evaluation/bootstrap.py src/iira2/evaluation/permutation.py src/iira2/evaluation/multiseed.py tests/evaluation
git commit -m "feat: add paired selective statistics"
```

### Task 3: Localization metrics and matched controls

**Files:**
- Create: `src/iira2/evaluation/localization.py`
- Create: `src/iira2/evaluation/controls.py`
- Test: `tests/evaluation/test_localization.py`
- Test: `tests/evaluation/test_controls.py`

**Interfaces:**
- Consumes: valid predicted/original boxes, radiologist boxes, image geometry, fixed seed.
- Produces: `LocalizationMetrics`, `ControlRegion`, `evaluate_localization(...)`, and matched/random/contralateral control generators.

- [ ] **Step 1: Write failing IoU, pointing, and control-area tests**

```python
def test_identical_boxes_have_iou_one() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_same_area_random_control_preserves_area() -> None:
    target = (10, 10, 30, 40)
    control = same_area_random(target, image_shape=(100, 100), seed=9)
    assert box_area(control.box) == box_area(target)
```

- [ ] **Step 2: Run localization tests and confirm missing modules**

Run: `python -m pytest tests/evaluation/test_localization.py tests/evaluation/test_controls.py -v`

Expected: FAIL because localization/control modules do not exist.

- [ ] **Step 3: Implement localization evaluation**

Compute IoU, pointing accuracy, box recall at declared IoU thresholds, and mAP only when enough scored predictions/ground truth exist. Aggregate multiple radiologist boxes by matching against any valid same-pathology box and report reader/consensus policy. Exclude missing ROI from IoU denominator but report missing-ROI rate.

- [ ] **Step 4: Implement deterministic controls**

Generate same-area random, matched non-overlapping, and horizontal contralateral regions within image bounds. Carry failure reason when a valid region cannot be placed; never shrink controls silently.

- [ ] **Step 5: Run localization tests and commit**

Run: `python -m pytest tests/evaluation/test_localization.py tests/evaluation/test_controls.py -v`

Expected: PASS.

```bash
git add src/iira2/evaluation/localization.py src/iira2/evaluation/controls.py tests/evaluation
git commit -m "feat: evaluate localized evidence"
```

### Task 4: Evidence stress and ROI interventions

**Files:**
- Create: `src/iira2/interventions/evidence_stress.py`
- Create: `src/iira2/interventions/image_masks.py`
- Create: `src/iira2/interventions/runner.py`
- Create: `src/iira2/interventions/__init__.py`
- Create: `configs/experiment/evidence_stress.yaml`
- Create: `configs/experiment/localization_intervention.yaml`
- Test: `tests/interventions/test_evidence_stress.py`
- Test: `tests/interventions/test_image_masks.py`
- Test: `tests/interventions/test_runner.py`

**Interfaces:**
- Consumes: clean immutable evidence/cache, clean-run ROI, source image, intervention config.
- Produces: derived stressed evidence, masked images, and `InterventionResult` paired to the clean run.

- [ ] **Step 1: Write failing clean-cache immutability and fixed-ROI tests**

```python
def test_stress_does_not_mutate_clean_evidence(clean_evidence) -> None:
    before = hash_evidence(clean_evidence)
    _ = apply_evidence_stress(clean_evidence, noise=0.1, flip_rate=0.0, missing_rate=0.0, seed=3)
    assert hash_evidence(clean_evidence) == before


def test_intervention_uses_clean_run_roi(clean_run, localizer_spy) -> None:
    run_intervention(clean_run, mask="blackout", localizer=localizer_spy)
    assert localizer_spy.call_count == 0
```

- [ ] **Step 2: Run intervention tests and confirm missing package**

Run: `python -m pytest tests/interventions -v`

Expected: FAIL because intervention modules do not exist.

- [ ] **Step 3: Implement evidence stress as derived artifacts**

Apply configured probability noise, flips, missing evidence, calibration shift, and ROI corruption using deterministic sample-addressed RNG. Every row retains parent evidence ID, corruption type/parameters/seed, and a new hash. Stress outputs cannot be labeled clean or enter main tables.

- [ ] **Step 4: Implement ROI blackout/gray/local-mean and controls**

Operate on clean-run target boxes and same-area/matched/contralateral controls. Preserve image size/mode, clip boxes safely, and hash the output plus mask geometry. Re-querying the localizer is a separate explicitly named experiment.

- [ ] **Step 5: Implement paired intervention summaries**

Report N, delta Brier, delta probability, action flip rate, paired CI, and paired p-value for all QUERY rows, all valid-ROI rows, and evidence-adopted subset separately.

- [ ] **Step 6: Run intervention tests and commit**

Run: `python -m pytest tests/interventions -v`

Expected: PASS.

```bash
git add src/iira2/interventions configs/experiment/evidence_stress.yaml configs/experiment/localization_intervention.yaml tests/interventions
git commit -m "feat: stress and intervene on external evidence"
```

### Task 5: Experiment matrix and Go/No-Go engine

**Files:**
- Create: `src/iira2/experiments/registry.py`
- Create: `src/iira2/experiments/gates.py`
- Create: `src/iira2/experiments/lock.py`
- Create: `src/iira2/experiments/__init__.py`
- Create: `configs/experiment/matrix.yaml`
- Create: `docs/EXPERIMENT_MATRIX.md`
- Create: `docs/STATISTICAL_PLAN.md`
- Test: `tests/experiments/test_registry.py`
- Test: `tests/experiments/test_gates.py`
- Test: `tests/experiments/test_vindr_lock.py`

**Interfaces:**
- Consumes: compact/internal metric artifacts and immutable config/model/data/cache identities.
- Produces: `ExperimentArm`, `GateDecision`, `ExperimentLock`, and `evaluate_gate(gate_id, artifacts) -> GateDecision`.

- [ ] **Step 1: Write failing matrix-completeness and VinDr-lock tests**

```python
def test_matrix_contains_all_frozen_arms(registry) -> None:
    assert set(registry.ids) == {
        "A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2", "C3", "C4",
        "D1", "D2", "E1", "E2"
    }


def test_vindr_requires_frozen_experiment_lock(lock_builder) -> None:
    with pytest.raises(ExperimentNotFrozen):
        lock_builder.authorize_external_test("E1")
```

- [ ] **Step 2: Run experiment tests and confirm missing modules**

Run: `python -m pytest tests/experiments -v`

Expected: FAIL because experiment registry/gates do not exist.

- [ ] **Step 3: Encode the exact approved matrix and dependencies**

Each arm declares config path, dataset/split, prerequisites, baseline, query budget policy, output schema, and whether it is clean/stress/intervention. Qwen fine-tuning, GRPO, DAPO-inspired Options, and Appendix-CISPO remain outside P0 and cannot block matrix completion.

- [ ] **Step 4: Implement Gates 0 through 5**

Use the exact design thresholds and evidence requirements. `GateDecision` contains `PASS`, `FAIL`, or `BLOCKED`, reason, measured values, threshold, artifact hashes, and permitted next phases. A failed complementarity gate stops large Macro-CISPO but preserves negative KBCSv2/baseline results.

- [ ] **Step 5: Implement the external-test lock**

Hash resolved configs, code commit, Qwen/KBCSv2/calibrator/reliability/cache identities, ontology, thresholds, query/abstain costs, seeds, and statistical plan. VinDr evaluation requires this hash-sealed local lock and records one authorized run identity; no fit-capable object is passed to the external-test runner.

- [ ] **Step 6: Run experiment tests and commit**

Run: `python -m pytest tests/experiments -v`

Expected: PASS.

```bash
git add src/iira2/experiments configs/experiment/matrix.yaml docs/EXPERIMENT_MATRIX.md docs/STATISTICAL_PLAN.md tests/experiments
git commit -m "feat: lock experiment matrix and gates"
```

### Task 6: Compact HTML and PNG reports

**Files:**
- Create: `src/iira2/reporting/compact.py`
- Create: `src/iira2/reporting/html.py`
- Create: `src/iira2/reporting/png.py`
- Create: `src/iira2/reporting/templates/report.html`
- Create: `src/iira2/cli/report.py`
- Test: `tests/reporting/test_compact_report.py`
- Test: `tests/reporting/test_report_redaction.py`
- Test: `tests/reporting/test_png_layout.py`

**Interfaces:**
- Consumes: `RunSummary`, gate decision, approved key metrics.
- Produces: `REPORT.html`, `REPORT.png`, and no additional exchange data.

- [ ] **Step 1: Write failing redaction and fixed-layout tests**

```python
def test_compact_report_contains_only_allowlisted_metric_keys(run_summary) -> None:
    report = build_compact_report(run_summary)
    assert set(report.metrics).issubset(ALLOWED_KEY_METRICS)


def test_png_elements_fit_canvas(compact_report) -> None:
    layout = layout_report(compact_report, width=1600, height=900)
    assert all(0 <= box.left < box.right <= 1600 for box in layout.boxes)
    assert all(0 <= box.top < box.bottom <= 900 for box in layout.boxes)
```

- [ ] **Step 2: Run compact-report tests and confirm missing renderers**

Run: `python -m pytest tests/reporting/test_compact_report.py tests/reporting/test_report_redaction.py tests/reporting/test_png_layout.py -v`

Expected: FAIL because compact renderers do not exist.

- [ ] **Step 3: Implement one-screen report model**

Display run/status/phase/gate, algorithm/dataset/split/seed, config/code/model/cache short hashes, elapsed/peak memory, and only phase-applicable key metrics. Show `COMPLETE`, `PARTIAL`, `MISSING`, `BLOCKED`, or `FAILED` visibly. Limit warnings/blockers to sanitized concise strings and show omitted-detail counts.

- [ ] **Step 4: Render self-contained HTML and Pillow PNG**

Use a static escaped HTML template with no external JS/CSS/font/network reference. Draw the 1600x900 PNG with Pillow using a bundled redistributable font or Pillow default, precomputed row heights, ellipsis for sanitized long labels, and a footer pointing to redacted relative evidence refs. Do not screenshot via a browser.

- [ ] **Step 5: Add content parity and PHI-path tests**

Assert JSON/HTML/PNG layout model share the same status/key metric values; HTML contains no absolute drive path, subject/study/image ID, traceback, raw prompt, trajectory, or prediction rows. Verify PNG is nonblank, exact dimensions, and all text boxes fit.

- [ ] **Step 6: Run report tests and commit**

Run: `python -m pytest tests/reporting -v`

Expected: PASS.

```bash
git add src/iira2/reporting src/iira2/cli/report.py tests/reporting
git commit -m "feat: render compact offline reports"
```

### Task 7: Unified launch, dry-run, resume, seed sweep, and grid

**Files:**
- Create: `src/iira2/cli/launch.py`
- Create: `src/iira2/experiments/launcher.py`
- Create: `src/iira2/experiments/sweep.py`
- Create: `src/iira2/experiments/lifecycle.py`
- Test: `tests/experiments/test_launcher.py`
- Test: `tests/experiments/test_sweep.py`
- Test: `tests/experiments/test_lifecycle.py`

**Interfaces:**
- Consumes: config path, dot-list overrides, optional seed/grid/resume arguments.
- Produces: deterministic `RunRequest` objects and one run directory/status per request.

- [ ] **Step 1: Write failing validation-before-allocation and deterministic-grid tests**

```python
def test_invalid_config_fails_before_device_allocation(device_spy, invalid_config) -> None:
    with pytest.raises(ConfigError):
        launch(invalid_config)
    assert device_spy.allocations == 0


def test_grid_order_is_deterministic() -> None:
    first = expand_grid({"seed": [2, 1], "reward.query_cost": [0.0, 0.1]})
    second = expand_grid({"reward.query_cost": [0.0, 0.1], "seed": [2, 1]})
    assert [r.run_id for r in first] == [r.run_id for r in second]
```

- [ ] **Step 2: Run launcher tests and confirm missing orchestration**

Run: `python -m pytest tests/experiments/test_launcher.py tests/experiments/test_sweep.py tests/experiments/test_lifecycle.py -v`

Expected: FAIL because launcher modules do not exist.

- [ ] **Step 3: Implement validate-resolve-hash-launch lifecycle**

Resolve YAML/overrides, validate illegal combinations, hash config, create run directory/status, probe prerequisites, dispatch the experiment arm, update internal artifacts, evaluate gates, render compact reports, and close status atomically. Support either `--config <path>` for one run or `--matrix <path> --arm <id>` for a registered arm. Catch known blocked states separately from failures and never swallow exceptions from internal logs.

- [ ] **Step 4: Implement dry-run/resume/sweep semantics**

Dry-run performs schema, path, asset, lock, split, cache, storage, and device-plan checks without model allocation. Resume requires exact run identity/checkpoint hashes. Grid keys sort lexicographically and values preserve declared order; run IDs derive from canonical config plus seed. Each seed is a separate run.

- [ ] **Step 5: Run launcher tests and CLI dry-run**

Run: `python -m pytest tests/experiments/test_launcher.py tests/experiments/test_sweep.py tests/experiments/test_lifecycle.py -v`

Run: `python -m iira2.cli.launch --config configs/experiment/macro_cispo.yaml runtime.dry_run=true`

Expected: tests PASS; dry-run emits compact prerequisites/status without allocating a model/NPU.

- [ ] **Step 6: Commit unified launch flow**

```bash
git add src/iira2/cli/launch.py src/iira2/experiments tests/experiments
git commit -m "feat: launch reproducible experiment runs"
```

### Task 8: Ascend device and distributed runtime profiles

**Files:**
- Create: `src/iira2/runtime/device.py`
- Create: `src/iira2/runtime/distributed.py`
- Create: `src/iira2/runtime/device_map.py`
- Create: `configs/runtime/probe_1npu.yaml`
- Create: `configs/runtime/qwen_2npu.yaml`
- Create: `configs/runtime/qwen_4npu.yaml`
- Create: `configs/runtime/train_4npu.yaml`
- Create: `configs/runtime/max_7npu.yaml`
- Create: `scripts/launch_npu.ps1`
- Test: `tests/runtime/test_device.py`
- Test: `tests/runtime/test_device_map.py`
- Test: `tests/runtime/test_distributed.py`
- Test: `tests/npu/test_qwen_npu.py`
- Test: `tests/npu/test_hccl.py`

**Interfaces:**
- Consumes: target environment probe and runtime profile.
- Produces: `DeviceContext`, `DistributedContext`, `QwenDevicePlan`, and validated process launch arguments.

- [ ] **Step 1: Write failing device-count and unprobed-seven-device tests**

```python
def test_profile_never_uses_eight_devices() -> None:
    assert max(profile.device_count for profile in load_runtime_profiles()) == 7


def test_max7_requires_passing_hccl_probe(environment_probe) -> None:
    with pytest.raises(RuntimeProfileError):
        build_device_context("max_7npu", replace(environment_probe, hccl_verified=False))
```

- [ ] **Step 2: Run runtime tests and confirm missing device modules**

Run: `python -m pytest tests/runtime/test_device.py tests/runtime/test_device_map.py tests/runtime/test_distributed.py -v`

Expected: FAIL because device/distributed runtime code does not exist.

- [ ] **Step 3: Implement lazy torch_npu device handling**

Import `torch_npu` only when an NPU profile is requested. Set device by local rank, expose BF16 autocast only when supported, synchronize for timing, report allocated/reserved peak memory, and normalize NPU errors into internal diagnostics plus compact blocker codes. CPU host tests remain importable without `torch_npu`.

- [ ] **Step 4: Implement HCCL training and Qwen sharding plans separately**

Use HCCL distributed data parallel for KBCSv2/controller training. For frozen Qwen inference, build explicit 2-way then 4-way module placement from the loaded model structure and measured per-device capacity; validate no missing/duplicate parameters and run an image smoke before accepting. Do not describe DDP as tensor parallel and do not infer a 7-way Qwen map from device count.

- [ ] **Step 5: Implement profile launch and safety checks**

Reject requested count above 7, count above observed availability, `max_7npu` without HCCL evidence, and any CUDA device/backend. Record visible NPU IDs, ranks, master address/port, HCCL settings, profile hash, and actual peak memory internally.

- [ ] **Step 6: Run host tests then target-only smokes**

Run: `python -m pytest tests/runtime -v`

Run on target: `python -m pytest tests/npu/test_qwen_npu.py tests/npu/test_hccl.py -v`

Expected: host tests PASS. Target tests verify one-image Qwen inference on 2-way, then 4-way, plus a two-rank HCCL all-reduce; failures remain `BLOCKED` and prevent higher profiles.

- [ ] **Step 7: Commit Ascend runtime profiles**

```bash
git add src/iira2/runtime configs/runtime scripts/launch_npu.ps1 tests/runtime tests/npu
git commit -m "feat: add Ascend NPU runtime profiles"
```

### Task 9: Network-disabled verification and offline acceptance

**Files:**
- Create: `src/iira2/offline/network_guard.py`
- Create: `src/iira2/offline/acceptance.py`
- Create: `src/iira2/cli/verify_offline.py`
- Create: `configs/experiment/offline_smoke.yaml`
- Create: `docs/OFFLINE_RUNBOOK.md`
- Create: `docs/KNOWN_LIMITATIONS.md`
- Test: `tests/offline/test_network_guard.py`
- Test: `tests/offline/test_acceptance.py`
- Test: `tests/npu/test_offline_end_to_end.py`

**Interfaces:**
- Consumes: verified bundle, restricted-data mount, environment probe, smoke config.
- Produces: `OfflineAcceptance` with per-check `COMPLETE`, `PARTIAL`, `MISSING`, `BLOCKED`, or `FAILED` and overall readiness.

- [ ] **Step 1: Write failing hidden-network and readiness tests**

```python
def test_network_guard_blocks_socket_connect() -> None:
    with NetworkGuard.block_all():
        with pytest.raises(NetworkAccessAttempt):
            socket.create_connection(("example.com", 443))


def test_host_success_cannot_be_offline_ready(host_acceptance) -> None:
    assert host_acceptance.overall != "OFFLINE_READY"
```

- [ ] **Step 2: Run offline tests and confirm missing acceptance code**

Run: `python -m pytest tests/offline/test_network_guard.py tests/offline/test_acceptance.py -v`

Expected: FAIL because network guard/acceptance code does not exist.

- [ ] **Step 3: Implement layered offline verification**

Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`; install from wheelhouse in a clean environment; block Python socket connects during tests; verify bundle hashes; then load/infer Qwen and RAD-DINO, run KBCSv2 heads, loaders, caches, each macro action, one update/eval for four P0 algorithms, one intervention, and one bootstrap evaluation.

- [ ] **Step 4: Separate host, NPU, and true network-disabled evidence**

Statuses progress only through `SOURCE_READY`, `ASSETS_VERIFIED`, `HOST_TESTED`, `NPU_SMOKE_PASSED`, and `OFFLINE_READY`. The last state requires execution in a target container launched without network; Python monkeypatches alone are insufficient. Record container image digest and environment hash.

- [ ] **Step 5: Run host acceptance tests and target command**

Run: `python -m pytest tests/offline -v`

Run in target network-disabled container: `python -m iira2.cli.verify_offline --config configs/experiment/offline_smoke.yaml`

Expected: host tests PASS. Target command emits all three compact artifacts and only claims `OFFLINE_READY` when every mandatory check passes.

- [ ] **Step 6: Commit offline acceptance tooling**

```bash
git add src/iira2/offline src/iira2/cli/verify_offline.py docs/OFFLINE_RUNBOOK.md docs/KNOWN_LIMITATIONS.md tests/offline tests/npu/test_offline_end_to_end.py
git commit -m "feat: verify network-disabled NPU execution"
```

### Task 10: End-to-end matrix acceptance and final documentation

**Files:**
- Modify: `configs/experiment/offline_smoke.yaml`
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/MODEL_ASSETS.md`
- Create: `docs/EXTERNAL_VERIFIER.md`
- Create: `docs/INTERVENTION_PLAN.md`
- Create: `docs/AGENT_FINAL_REPORT.md`
- Create: `tests/integration/test_synthetic_end_to_end.py`
- Create: `tests/integration/test_compact_outputs.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all implemented modules and a generated synthetic non-medical fixture set.
- Produces: source-ready reproducible harness plus evidence-backed readiness status.

- [ ] **Step 1: Write failing synthetic end-to-end test**

```python
@pytest.mark.parametrize("algorithm", [
    "supervised_router", "contextual_bandit", "ppo_options", "macro_cispo"
])
def test_synthetic_pipeline_emits_compact_outputs(tmp_path, algorithm) -> None:
    result = run_synthetic_pipeline(tmp_path, algorithm=algorithm, seed=42)
    assert result.status_path.exists()
    assert result.html_path.exists()
    assert result.png_path.exists()
    assert result.controller_gradient_ok if algorithm in {"ppo_options", "macro_cispo"} else True
```

- [ ] **Step 2: Run integration tests and confirm missing pipeline fixture**

Run: `python -m pytest tests/integration -v`

Expected: FAIL because the synthetic full-pipeline helper/config does not exist.

- [ ] **Step 3: Implement a non-clinical synthetic full pipeline**

Generate tiny images, internal records, direct/evidence/revised caches, deterministic labels, and separable controller features without copying real data. Run all actions and four algorithms through the same launcher/evaluator/report path. Mark artifacts `SYNTHETIC_ONLY`; they prove plumbing, not scientific performance.

- [ ] **Step 4: Complete documentation from verified behavior**

Document architecture, asset revisions/hashes/status, KBCSv2 boundaries, four algorithms and exact Macro-CISPO gradient boundary, experiment matrix, intervention protocol, offline commands, compact output schema, known limitations, and the difference between source/host/NPU/offline readiness. Do not copy aspirational claims into the final report as completed evidence.

- [ ] **Step 5: Run complete host verification**

Run: `python -m pytest -v`

Run: `python -m iira2.cli.launch --config configs/experiment/offline_smoke.yaml runtime.profile=probe_1npu`

Expected: all host tests PASS or target-only tests SKIP with explicit reasons; synthetic smoke emits valid compact artifacts and remains labeled `SYNTHETIC_ONLY`.

- [ ] **Step 6: Run target verification and locked experiments in gate order**

```powershell
python -m iira2.cli.launch --matrix configs/experiment/matrix.yaml --through-gate 5
if ($LASTEXITCODE -ne 0) { throw 'A required gate blocked later phases; inspect compact report' }
python -m iira2.cli.launch --matrix configs/experiment/matrix.yaml --arm E1 --require-lock
python -m iira2.cli.launch --matrix configs/experiment/matrix.yaml --arm E2 --require-lock
```

The first command runs Gate 0, Qwen Gate 1, KBCSv2 Gate 2, complementarity Gate 3, macro RL smoke Gate 4, then P0 comparison Gate 5, stopping when policy permits no next phase. Only after the experiment lock is hash-sealed do E1/E2 run once per declared seed/matrix entry. Preserve failed and negative outcomes.

- [ ] **Step 7: Commit final source documentation**

```bash
git add README.md configs/experiment/offline_smoke.yaml docs tests/integration
git commit -m "docs: complete IIRA 2.0 offline research harness"
```

Do not commit downloaded assets, restricted data, model checkpoints, evidence/response caches, prediction tables, or generated run outputs.
