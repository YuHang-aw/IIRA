# IIRA 2.0 Foundation and Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the installable repository, validated configuration surface, compact run-status protocol, resumable asset manager, environment probe, and restricted-safe offline bundler.

**Architecture:** Keep code in `${IIRA2_ROOT}\iira2` and all large assets under sibling directories in `${IIRA2_ROOT}`. Pure dataclasses and filesystem services form the host-testable core; Hugging Face, NPU, and dataset-specific work consumes their manifests later. Every command produces a compact `STATUS.json` before expensive work begins.

**Tech Stack:** Python 3.11, `dataclasses`, `pathlib`, `hashlib`, OmegaConf/Hydra, PyYAML, Hugging Face Hub, pytest, PowerShell, JSON/HTML.

**Spec:** `docs/superpowers/specs/2026-08-27-iira2-design.md`

## Global Constraints

- The old `IIRA-main.zip` is reference-only and must never be imported.
- Code lives in `${IIRA2_ROOT}\iira2`; models, datasets, wheelhouse, downloads, manifests, and bundles live under `${IIRA2_ROOT}`.
- P0 runs use at most 7 Ascend 910C NPUs; no CUDA, `bitsandbytes`, CUDA flash-attn, or ordinary vLLM dependency is allowed in the core path.
- Main Qwen model is `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- KBCSv2 backbone is `microsoft/rad-dino` at revision `110cbc18d5133582e320b43d53bf5c44e410c936`.
- Restricted datasets never enter Git, generic bundles, fixtures, compact reports, or logs.
- The console and exchange artifacts expose only key metrics; full audit artifacts remain under `outputs/<run_id>/internal/`.
- Every implementation task follows red-green-refactor and ends with an isolated commit.

---

### Task 1: Installable package and repository boundaries

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/iira2/__init__.py`
- Create: `src/iira2/cli/__init__.py`
- Create: `src/iira2/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: environment variable `IIRA2_ROOT` when explicitly set.
- Produces: `ProjectPaths.from_root(root: Path) -> ProjectPaths` and `ProjectPaths.discover() -> ProjectPaths`.

- [ ] **Step 1: Write the failing path-boundary test**

```python
from pathlib import Path

from iira2.paths import ProjectPaths


def test_project_paths_keep_code_and_assets_separate(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.code == tmp_path / "iira2"
    assert paths.models == tmp_path / "models"
    assert paths.datasets == tmp_path / "datasets"
    assert paths.outputs == tmp_path / "iira2" / "outputs"
```

- [ ] **Step 2: Run the focused test and confirm the expected import failure**

Run: `python -m pytest tests/test_paths.py -v`

Expected: FAIL because `iira2.paths` does not exist.

- [ ] **Step 3: Create the package metadata and path object**

```python
# src/iira2/paths.py
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    code: Path
    models: Path
    datasets: Path
    downloads: Path
    manifests: Path
    wheelhouse: Path
    offline_bundle: Path
    outputs: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        return cls(root, root / "iira2", root / "models", root / "datasets",
                   root / "downloads", root / "manifests", root / "wheelhouse",
                   root / "offline_bundle", root / "iira2" / "outputs")

    @classmethod
    def discover(cls) -> "ProjectPaths":
        return cls.from_root(Path(os.environ.get("IIRA2_ROOT", r"${IIRA2_ROOT}")))
```

Set `requires-python = ">=3.11,<3.12"`, setuptools package discovery under `src`, and a `dev` extra containing pytest. Ignore `.venv/`, caches, `outputs/`, and every sibling large-asset directory without ignoring `manifests/` inside the code repository.

- [ ] **Step 4: Install editable package and run the test**

Run: `python -m pip install -e .[dev]`

Run: `python -m pytest tests/test_paths.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the repository foundation**

```bash
git add pyproject.toml .gitignore README.md src/iira2/__init__.py src/iira2/cli/__init__.py src/iira2/paths.py tests/test_paths.py
git commit -m "build: scaffold iira2 package"
```

### Task 2: Typed configuration and illegal-combination guard

**Files:**
- Create: `src/iira2/config/schema.py`
- Create: `src/iira2/config/load.py`
- Create: `src/iira2/config/validate.py`
- Create: `src/iira2/config/__init__.py`
- Create: `configs/experiment/macro_cispo_mimic.yaml`
- Test: `tests/config/test_config_validation.py`

**Interfaces:**
- Consumes: YAML path and `list[str]` dot-list overrides.
- Produces: `ResolvedConfig`, `load_config(path: Path, overrides: list[str]) -> ResolvedConfig`, and `validate_config(config: ResolvedConfig) -> None`.

- [ ] **Step 1: Write failing tests for the valid main arm and forbidden trainable Qwen**

```python
import pytest

from iira2.config.schema import default_config
from iira2.config.validate import ConfigError, validate_config


def test_main_macro_config_is_valid() -> None:
    config = default_config()
    validate_config(config)


def test_main_macro_config_rejects_trainable_qwen() -> None:
    config = default_config()
    config.agent.trainable = True
    with pytest.raises(ConfigError, match="frozen Qwen"):
        validate_config(config)
```

- [ ] **Step 2: Run the tests and confirm missing schema failure**

Run: `python -m pytest tests/config/test_config_validation.py -v`

Expected: FAIL because `iira2.config.schema` does not exist.

- [ ] **Step 3: Implement focused dataclasses and validation**

```python
# src/iira2/config/validate.py
class ConfigError(ValueError):
    pass


def validate_config(config: "ResolvedConfig") -> None:
    if config.experiment.arm == "controller" and config.agent.trainable:
        raise ConfigError("main controller arm requires frozen Qwen")
    required = {"DIRECT_COMMIT", "QUERY_AND_REVISE", "ABSTAIN"}
    if config.controller.algorithm == "macro_cispo" and set(config.controller.actions) != required:
        raise ConfigError("Macro-CISPO requires the frozen three-action set")
    if "QUERY_AND_REVISE" in config.controller.actions and not (
        config.evidence.enabled and config.evidence.cache
    ):
        raise ConfigError("QUERY_AND_REVISE requires cached evidence")
    if config.data.external_test == "vindr" and config.experiment.mode in {
        "train", "calibrate", "threshold_fit"
    }:
        raise ConfigError("VinDr external test is locked against fitting")
```

Define nested dataclasses for every field in design section 18, restrict enum-like strings to their declared values, merge YAML and overrides with OmegaConf, resolve interpolations, validate, and compute SHA256 over canonical sorted JSON. The committed default must use `mode: train`, `arm: controller`, frozen Qwen, KBCSv2 cache, the three macro actions, `macro_cispo`, and `qwen_4npu`; it must leave `external_test` unset during training.

- [ ] **Step 4: Add parametrized invalid-combination coverage**

Cover missing evidence, wrong Macro-CISPO actions, VinDr fitting, unprobed `max_7npu`, stress entering a clean arm, unsupported algorithm, and fusion/arm mismatch. Assert the exact `ConfigError` message for each.

- [ ] **Step 5: Run the configuration tests**

Run: `python -m pytest tests/config -v`

Expected: PASS with every invalid combination rejected before model loading.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add src/iira2/config configs/experiment/macro_cispo_mimic.yaml tests/config
git commit -m "feat: add validated experiment configuration"
```

### Task 3: Compact status and run-artifact protocol

**Files:**
- Create: `src/iira2/reporting/status.py`
- Create: `src/iira2/reporting/redaction.py`
- Create: `src/iira2/reporting/run_dir.py`
- Create: `src/iira2/reporting/__init__.py`
- Test: `tests/reporting/test_status.py`
- Test: `tests/reporting/test_redaction.py`

**Interfaces:**
- Consumes: `ResolvedConfig`, phase status, optional key metrics.
- Produces: `RunSummary`, `RunDirectory.create(paths, run_id) -> RunDirectory`, and `write_status(run_dir, summary) -> Path`.

- [ ] **Step 1: Write the failing compact-schema test**

```python
import json

from iira2.reporting.status import RunSummary


def test_status_omits_missing_metrics_and_raw_paths() -> None:
    summary = RunSummary.started(run_id="r1", phase="audit", config_hash="abc")
    payload = summary.to_exchange_dict()
    assert "brier" not in payload["key_metrics"]
    assert "evidence_paths" not in payload
    assert json.dumps(payload).find("subject_id") == -1
```

- [ ] **Step 2: Run reporting tests and confirm missing-module failure**

Run: `python -m pytest tests/reporting -v`

Expected: FAIL because reporting modules do not exist.

- [ ] **Step 3: Implement status enums and atomic JSON writes**

```python
class RunState(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RunSummary:
    schema_version: str
    run_id: str
    status: RunState
    phase: str
    config_hash: str
    key_metrics: dict[str, float]
    warnings: tuple[str, ...]
    blockers: tuple[str, ...]
    evidence_refs: tuple[str, ...]
```

Write JSON to `STATUS.json.tmp`, flush and replace atomically. `RunDirectory` creates `internal/` separately. `to_exchange_dict()` must reject absolute paths, drive letters, `subject_id`, `study_id`, `image_id`, and traceback text in warnings/blockers/evidence refs.

- [ ] **Step 4: Add redaction and atomic-write tests**

Test that a restricted absolute path raises `UnsafeExchangeValue`, a failed write does not replace an existing valid status file, and absent metrics are omitted rather than serialized as zero.

- [ ] **Step 5: Run reporting tests**

Run: `python -m pytest tests/reporting -v`

Expected: PASS.

- [ ] **Step 6: Commit compact reporting primitives**

```bash
git add src/iira2/reporting tests/reporting
git commit -m "feat: add compact run status protocol"
```

### Task 4: Asset registry and storage planning

**Files:**
- Create: `src/iira2/assets/schema.py`
- Create: `src/iira2/assets/registry.py`
- Create: `src/iira2/assets/storage.py`
- Create: `src/iira2/assets/__init__.py`
- Create: `src/iira2/cli/assets.py`
- Create: `manifests/assets.yaml`
- Create: `docs/STORAGE_PLAN.md`
- Test: `tests/assets/test_registry.py`
- Test: `tests/assets/test_storage.py`

**Interfaces:**
- Consumes: committed `manifests/assets.yaml` and `ProjectPaths`.
- Produces: `AssetSpec`, `AssetRecord`, `AssetState`, `load_registry(path)`, and `build_storage_plan(registry, free_bytes) -> StoragePlan`.

- [ ] **Step 1: Write failing state-machine and headroom tests**

```python
from iira2.assets.schema import AssetState
from iira2.assets.storage import build_storage_plan


def test_partial_asset_cannot_be_verified() -> None:
    assert not AssetState.PARTIAL.can_transition_to(AssetState.VERIFIED)


def test_storage_plan_requires_twenty_percent_headroom(registry) -> None:
    plan = build_storage_plan(registry, free_bytes=1200)
    assert plan.required_with_headroom == int(plan.required_bytes * 1.2)
```

- [ ] **Step 2: Run tests and confirm missing asset modules**

Run: `python -m pytest tests/assets/test_registry.py tests/assets/test_storage.py -v`

Expected: FAIL because asset types are undefined.

- [ ] **Step 3: Implement explicit asset records and transitions**

Define states `MISSING`, `DOWNLOADING`, `PARTIAL`, `COMPLETE`, `VERIFIED`, `BLOCKED_BY_TERMS`, and `FAILED`. A verified record requires non-empty file hashes, exact revision/version, license identifier, observed size, and verification evidence ref. Do not allow `PARTIAL -> VERIFIED`.

The initial manifest contains exact model IDs/revisions plus the four dataset versions and access classes. Model estimates must be marked `observed_remote`; dataset estimates must be marked `official_or_unknown`. Unknown sizes remain `null` and force `StoragePlan.complete_estimate=False`.

- [ ] **Step 4: Generate and test the Markdown storage plan**

Run: `python -m iira2.cli.assets storage-plan --root ${IIRA2_ROOT}`

Expected: `docs/STORAGE_PLAN.md` lists each asset, size evidence, subtotal, 20% headroom, free space, and `INCOMPLETE_ESTIMATE` when any official size is unknown.

- [ ] **Step 5: Run asset tests**

Run: `python -m pytest tests/assets -v`

Expected: PASS.

- [ ] **Step 6: Commit asset and storage contracts**

```bash
git add src/iira2/assets src/iira2/cli/assets.py manifests/assets.yaml docs/STORAGE_PLAN.md tests/assets
git commit -m "feat: add asset registry and storage planner"
```

### Task 5: Resumable and verifiable download manager

**Files:**
- Create: `src/iira2/assets/download.py`
- Create: `src/iira2/assets/huggingface.py`
- Create: `src/iira2/assets/hash_manifest.py`
- Modify: `src/iira2/cli/assets.py`
- Test: `tests/assets/test_download.py`
- Test: `tests/assets/test_hash_manifest.py`

**Interfaces:**
- Consumes: `AssetSpec`, destination path, optional Hugging Face token from process environment.
- Produces: `DownloadManager.download(spec) -> AssetRecord` and `build_hash_manifest(root) -> HashManifest`.

- [ ] **Step 1: Write failing lock and partial-state tests using a local HTTP server**

```python
def test_interrupted_download_remains_partial(fake_server, manager) -> None:
    fake_server.disconnect_after(1024)
    record = manager.download(fake_server.asset_spec())
    assert record.state.value == "PARTIAL"
    assert record.destination.with_suffix(".partial").exists()


def test_second_downloader_cannot_take_live_lock(manager, asset_spec) -> None:
    with manager.lock(asset_spec):
        with pytest.raises(DownloadLocked):
            manager.download(asset_spec)
```

- [ ] **Step 2: Run the tests and confirm missing implementation**

Run: `python -m pytest tests/assets/test_download.py tests/assets/test_hash_manifest.py -v`

Expected: FAIL because download services do not exist.

- [ ] **Step 3: Implement safe download state transitions**

Use an exclusive lock file containing PID, host, asset ID, and start time. Download into `.partial`, resume only when the server supports ranges or Hugging Face Hub validates the local cache, then atomically rename after size/hash checks. Catch exceptions into a `FAILED` or `PARTIAL` record without printing credentials or URLs containing tokens.

For model snapshots, call `snapshot_download(repo_id=..., revision=..., local_dir=..., local_files_only=False)` and then build a sorted file-level SHA256 manifest. Never use a moving branch name.

- [ ] **Step 4: Add model manifest and CLI tests with mocked Hub calls**

Assert exact repo IDs and revisions, `local_dir` under `${IIRA2_ROOT}\models`, no token in serialized status, deterministic hash ordering, and no `VERIFIED` state until offline file validation succeeds.

- [ ] **Step 5: Run the complete asset test set**

Run: `python -m pytest tests/assets -v`

Expected: PASS without network access.

- [ ] **Step 6: Commit the download manager**

```bash
git add src/iira2/assets src/iira2/cli/assets.py tests/assets
git commit -m "feat: add resumable verified asset downloads"
```

### Task 6: Target environment probe and wheelhouse contract

**Files:**
- Create: `src/iira2/runtime/probe.py`
- Create: `src/iira2/runtime/compatibility.py`
- Create: `src/iira2/runtime/__init__.py`
- Create: `src/iira2/cli/probe.py`
- Create: `scripts/probe_target.ps1`
- Create: `docs/TARGET_ENVIRONMENT.md`
- Create: `requirements/host.lock`
- Create: `requirements/npu.lock.template`
- Test: `tests/runtime/test_probe.py`
- Test: `tests/runtime/test_compatibility.py`

**Interfaces:**
- Consumes: host subprocess outputs and, in target runtime, `torch_npu` runtime APIs.
- Produces: `EnvironmentProbe`, `CompatibilityDecision`, and `probe_environment() -> EnvironmentProbe`.

- [ ] **Step 1: Write failing probe normalization tests**

```python
def test_probe_without_torch_npu_is_host_only(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    result = probe_environment()
    assert result.npu.available is False
    assert result.readiness == "HOST_ONLY"
```

- [ ] **Step 2: Run runtime tests and confirm missing probe failure**

Run: `python -m pytest tests/runtime -v`

Expected: FAIL because runtime probe modules do not exist.

- [ ] **Step 3: Implement non-throwing environment discovery**

Capture OS, architecture, Python, disk, runtime markers, image digest when available, `torch`, `torch_npu`, CANN, driver, firmware, NPU names/count/memory, and HCCL. Missing commands become explicit `null` plus warning, never fabricated values. A compatibility decision remains `UNRESOLVED` until exact Python/CANN/torch/torch_npu versions are observed.

- [ ] **Step 4: Implement wheelhouse lock generation gate**

The CLI must refuse to render `requirements/npu.lock` from the template until a compatible matrix row exactly matches the target probe. The template lists package names only; no CUDA index, `bitsandbytes`, CUDA flash-attn, or ordinary vLLM is permitted.

- [ ] **Step 5: Run runtime tests and host probe**

Run: `python -m pytest tests/runtime -v`

Run: `python -m iira2.cli.probe --output ${IIRA2_ROOT}\manifests\host_environment.json`

Expected: tests PASS; current machine is truthfully reported without being labeled NPU ready.

- [ ] **Step 6: Commit the probe and dependency gate**

```bash
git add src/iira2/runtime src/iira2/cli/probe.py scripts/probe_target.ps1 docs/TARGET_ENVIRONMENT.md requirements tests/runtime
git commit -m "feat: add target environment compatibility probe"
```

### Task 7: Restricted-safe offline bundle builder

**Files:**
- Create: `src/iira2/offline/bundle.py`
- Create: `src/iira2/offline/verify.py`
- Create: `src/iira2/offline/__init__.py`
- Create: `src/iira2/cli/bundle.py`
- Create: `docs/LICENSE_AND_ACCESS.md`
- Test: `tests/offline/test_bundle.py`
- Test: `tests/offline/test_restricted_exclusion.py`

**Interfaces:**
- Consumes: verified asset records and explicit inclusion list.
- Produces: `build_bundle(spec: BundleSpec) -> BundleResult` and `verify_bundle(path: Path) -> BundleVerification`.

- [ ] **Step 1: Write failing restricted-exclusion tests**

```python
def test_bundle_rejects_restricted_dataset(tmp_path: Path) -> None:
    spec = BundleSpec(output=tmp_path / "bundle", includes=[tmp_path / "datasets" / "restricted"])
    with pytest.raises(RestrictedAssetError):
        build_bundle(spec)


def test_bundle_manifest_is_deterministic(small_bundle_spec) -> None:
    first = build_bundle(small_bundle_spec)
    second = build_bundle(small_bundle_spec)
    assert first.manifest_sha256 == second.manifest_sha256
```

- [ ] **Step 2: Run offline tests and confirm missing bundle code**

Run: `python -m pytest tests/offline -v`

Expected: FAIL because offline bundle modules do not exist.

- [ ] **Step 3: Implement allowlist-based assembly and verification**

Only copy code snapshot, verified non-restricted models, wheelhouse, configs, docs, manifests, scripts, and references. Reject paths under `datasets/restricted`, filenames/metadata containing subject/study/image identifiers, symlinks escaping the root, duplicate archive paths, and any file absent from the manifest. Generate `MANIFEST.sha256` in normalized path order.

- [ ] **Step 4: Add fresh-directory verification**

Verify safe paths, hashes, required licenses, model revisions, wheel completeness marker, code commit, and absence of restricted content from a newly extracted directory. Return `PARTIAL` when a required non-restricted asset is missing; never mark it complete.

- [ ] **Step 5: Run foundation verification**

Run: `python -m pytest tests/test_paths.py tests/config tests/reporting tests/assets tests/runtime tests/offline -v`

Expected: PASS.

- [ ] **Step 6: Commit the offline bundle boundary**

```bash
git add src/iira2/offline src/iira2/cli/bundle.py docs/LICENSE_AND_ACCESS.md tests/offline
git commit -m "feat: build restricted-safe offline bundles"
```

### Task 8: Read-only legacy and paper appendix audit

**Files:**
- Create: `src/iira2/audits/legacy_boundary.py`
- Create: `src/iira2/audits/__init__.py`
- Create: `docs/LEGACY_REFERENCE_AUDIT.md`
- Test: `tests/architecture/test_no_legacy_import.py`
- Test: `tests/architecture/test_legacy_archive_safety.py`

**Interfaces:**
- Consumes: `${IIRA2_ROOT}\IIRA-main.zip`, manuscript/thesis source material, and new `src/iira2` AST.
- Produces: a read-only behavior/formula audit and `audit_legacy_boundary(source_root, archive_path) -> LegacyBoundaryAudit`.

- [ ] **Step 1: Write failing no-import and archive-safety tests**

```python
def test_new_source_does_not_import_legacy_modules() -> None:
    audit = audit_legacy_boundary(Path("src/iira2"), Path(r"${IIRA2_ROOT}\IIRA-main.zip"))
    assert audit.forbidden_imports == ()


def test_legacy_archive_members_are_safe_paths() -> None:
    audit = inspect_archive_members(Path(r"${IIRA2_ROOT}\IIRA-main.zip"))
    assert audit.path_traversal_members == ()
```

- [ ] **Step 2: Run architecture tests and confirm missing audit module**

Run: `python -m pytest tests/architecture -v`

Expected: FAIL because legacy-boundary audit code does not exist.

- [ ] **Step 3: Implement read-only AST/archive inspection**

List ZIP members without importing or executing them, reject traversal/absolute paths, and parse readable Python source directly from ZIP bytes. Parse every new `src/iira2` file with `ast` and report imports or path references to old IIRA package names. The production package must not add the legacy ZIP to `sys.path`.

- [ ] **Step 4: Record appendix and historical behavior mapping**

Document exact manuscript/thesis section/page or source-file/function evidence for action semantics, CISPO terms, KBCS behavior, metrics, and intervention setup. Explicitly record that the old trainer accumulated trajectory-wide log-probability/entropy/KL and therefore is not reused as terminal-macro CISPO. Mark contradictions as historical, not requirements.

- [ ] **Step 5: Run boundary tests and commit**

Run: `python -m pytest tests/architecture -v`

Expected: PASS with zero legacy imports/path injection.

```bash
git add src/iira2/audits docs/LEGACY_REFERENCE_AUDIT.md tests/architecture
git commit -m "docs: audit legacy behavior without importing it"
```

### Task 9: Phase acceptance and real asset download checkpoint

**Files:**
- Modify: `docs/STORAGE_PLAN.md`
- Modify: `docs/TARGET_ENVIRONMENT.md`
- Generate (untracked): `outputs/foundation-audit/STATUS.json`
- Generate (untracked): `outputs/foundation-audit/internal/asset_audit.json`

**Interfaces:**
- Consumes: all prior foundation commands.
- Produces: a committed-code checkpoint plus external asset state under `${IIRA2_ROOT}`.

- [ ] **Step 1: Run the full host suite**

Run: `python -m pytest -v`

Expected: all host tests PASS; target-only NPU tests are explicitly skipped with a reason.

- [ ] **Step 2: Generate current storage and environment evidence**

Run: `python -m iira2.cli.assets storage-plan --root ${IIRA2_ROOT}`

Run: `python -m iira2.cli.probe --output ${IIRA2_ROOT}\manifests\host_environment.json`

Expected: both files contain measured values and unresolved fields are explicit.

- [ ] **Step 3: Download unrestricted model snapshots in declared order**

Run: `python -m iira2.cli.assets download --asset rad-dino --root ${IIRA2_ROOT}`

Run: `python -m iira2.cli.assets download --asset qwen3.8-27b --root ${IIRA2_ROOT}`

Expected: each command ends in `VERIFIED`, `PARTIAL`, or `FAILED` with compact evidence; never start Qwen when the storage plan lacks 20% headroom.

- [ ] **Step 4: Re-run offline file verification**

Run: `python -m iira2.cli.assets verify --all --local-files-only --root ${IIRA2_ROOT}`

Expected: downloaded snapshots have exact revisions and deterministic file hashes. This does not claim model inference readiness.

- [ ] **Step 5: Commit documentation updates only**

```bash
git add docs/STORAGE_PLAN.md docs/TARGET_ENVIRONMENT.md
git commit -m "docs: record foundation asset audit"
```

Do not add generated outputs, downloaded models, sibling manifests, or wheel files to Git.
