# IIRA 2.0 Macro Controller and P0 Algorithms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the leakage-safe terminal macro environment, shared controller interface, non-RL baselines, PPO-Options, and terminal-action-only Macro-CISPO.

**Architecture:** Every algorithm receives the same detached pre-query observation and selects one of three terminal macro actions. The environment alone may access KBCSv2 and Qwen response caches after `QUERY_AND_REVISE`; it emits an immutable one-step transition and reward. Four controllers share common batching, checkpointing, evaluation, and budget accounting.

**Tech Stack:** Python 3.11, PyTorch, NumPy, scikit-learn, PyArrow, pytest; `torch_npu` only behind the runtime adapter.

**Spec:** `docs/superpowers/specs/2026-08-27-iira2-design.md`

## Global Constraints

- Execute the foundation/assets, data/Qwen, and KBCSv2 plans first.
- P0 action set is exactly `DIRECT_COMMIT`, `QUERY_AND_REVISE`, and `ABSTAIN`.
- Before a query, the controller cannot access evidence probability, ROI, localization, reliability, evidence ID, or revised Qwen response.
- All algorithms share identical records, observations, action semantics, rewards, caches, splits, seeds, and evaluators.
- Qwen and KBCSv2 are frozen; only controller policy/value parameters may receive gradients during RL.
- Covered reward is negative Brier minus query cost; abstention is a separate terminal outcome with fixed development-selected utility.
- Macro-CISPO is the main algorithm with `K=3`; only the terminal macro-action decision contributes policy gradient.
- Supervised Router, Contextual Bandit, and PPO-Options are mandatory P0 comparators.
- GRPO, DAPO-inspired Options, Appendix-CISPO, recurrent methods, and Qwen LoRA remain non-P0.
- Every implementation task follows red-green-refactor and ends with an isolated commit.

---

### Task 1: Belief, action, observation, and transition contracts

**Files:**
- Create: `src/iira2/beliefs/schema.py`
- Create: `src/iira2/controllers/schema.py`
- Create: `src/iira2/controllers/features.py`
- Create: `src/iira2/controllers/__init__.py`
- Test: `tests/controllers/test_schema.py`
- Test: `tests/controllers/test_features.py`

**Interfaces:**
- Consumes: `SampleRecord`, direct `AgentResponse`, ontology index.
- Produces: `MacroAction`, `BeliefState`, `ControllerObservation`, `MacroTransition`, and `build_prequery_observation(...)`.

- [ ] **Step 1: Write failing action and feature-boundary tests**

```python
def test_p0_action_set_is_exact() -> None:
    assert {a.value for a in MacroAction} == {
        "DIRECT_COMMIT", "QUERY_AND_REVISE", "ABSTAIN"
    }


def test_prequery_observation_has_no_evidence_fields(sample, direct_response) -> None:
    observation = build_prequery_observation(sample, direct_response)
    names = set(observation.feature_names)
    assert not names.intersection({"evidence_probability", "roi", "reliability", "evidence_id"})
```

- [ ] **Step 2: Run schema tests and confirm missing controller types**

Run: `python -m pytest tests/controllers/test_schema.py tests/controllers/test_features.py -v`

Expected: FAIL because belief/controller schema modules do not exist.

- [ ] **Step 3: Implement immutable macro contracts**

```python
class MacroAction(str, Enum):
    DIRECT_COMMIT = "DIRECT_COMMIT"
    QUERY_AND_REVISE = "QUERY_AND_REVISE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class BeliefState:
    sample_id: str
    pathology: Pathology
    p_initial: float
    p_current: float
    queried_external: bool
    evidence_id: str | None
    covered: bool
    terminal_action: MacroAction | None


@dataclass(frozen=True)
class ControllerObservation:
    sample_id: str
    pathology_index: int
    features: np.ndarray
    feature_names: tuple[str, ...]
    valid_actions: tuple[MacroAction, ...]


@dataclass(frozen=True)
class MacroTransition:
    observation: ControllerObservation
    action: MacroAction
    belief_before: BeliefState
    belief_after: BeliefState
    reward: float
    covered: bool
    queried: bool
    internal_event_refs: tuple[str, ...]
```

`BeliefState` contains sample/pathology, initial/current probability, query flag, evidence ID, covered, and terminal action. Arrays are copied, set read-only, finite-checked, and contain only direct-Qwen detached features, probability/entropy/margin, pathology encoding, and permitted non-outcome metadata.

- [ ] **Step 4: Add deterministic feature-order and finite-value tests**

Assert the same feature schema/hash across records, detached Torch tensors are converted without grad, NaN/Inf fail, and labels/test outcomes are never accepted by the observation builder.

- [ ] **Step 5: Run schema tests and commit**

Run: `python -m pytest tests/controllers/test_schema.py tests/controllers/test_features.py -v`

Expected: PASS.

```bash
git add src/iira2/beliefs/schema.py src/iira2/controllers tests/controllers
git commit -m "feat: define macro controller state contracts"
```

### Task 2: Leakage-safe macro environment and reward

**Files:**
- Create: `src/iira2/rl/env.py`
- Create: `src/iira2/rl/reward.py`
- Create: `src/iira2/rl/__init__.py`
- Test: `tests/rl/test_environment.py`
- Test: `tests/rl/test_reward.py`
- Test: `tests/rl/test_cache_visibility.py`

**Interfaces:**
- Consumes: direct response cache, environment-private evidence cache, revised response cache, label store, `ControllerObservation`.
- Produces: `MacroEnvironment.reset(sample_id, pathology) -> ControllerObservation` and `MacroEnvironment.step(action) -> MacroTransition`.

- [ ] **Step 1: Write failing visibility and action-semantics tests**

```python
def test_reset_does_not_read_evidence_cache(spying_environment) -> None:
    spying_environment.reset("s1", Pathology.EDEMA)
    assert spying_environment.evidence_cache.read_count == 0


def test_query_reads_evidence_once_and_uses_revised_probability(environment) -> None:
    environment.reset("s1", Pathology.EDEMA)
    transition = environment.step(MacroAction.QUERY_AND_REVISE)
    assert transition.queried is True
    assert transition.belief_after.p_current == pytest.approx(0.8)
    assert environment.evidence_cache.read_count == 1


def test_abstain_is_not_probability_half(environment) -> None:
    environment.reset("s1", Pathology.EDEMA)
    transition = environment.step(MacroAction.ABSTAIN)
    assert transition.covered is False
    assert transition.belief_after.p_current != 0.5
```

- [ ] **Step 2: Run environment tests and confirm missing modules**

Run: `python -m pytest tests/rl/test_environment.py tests/rl/test_reward.py tests/rl/test_cache_visibility.py -v`

Expected: FAIL because environment/reward code does not exist.

- [ ] **Step 3: Implement one-step terminal macro execution**

`reset` reads only sample metadata and direct Qwen cache. `DIRECT_COMMIT` returns the calibrated direct probability; `QUERY_AND_REVISE` then reads exact frozen evidence and revised-response keys and returns the revised probability; `ABSTAIN` sets covered false while retaining current probability for internal diagnostics. Every action terminates; a second step raises `EpisodeTerminated`.

- [ ] **Step 4: Implement reward with explicit abstention utility**

```python
def macro_reward(action: MacroAction, probability: float, label: int,
                 query_cost: float, abstain_cost: float) -> float:
    if action is MacroAction.ABSTAIN:
        return -abstain_cost
    return -((probability - label) ** 2) - (
        query_cost if action is MacroAction.QUERY_AND_REVISE else 0.0
    )
```

Select `abstain_cost` only on MIMIC development validation to meet minimum coverage `0.80`; persist the candidate grid, chosen value, split hash, and achieved coverage. Evaluation uses the frozen value.

- [ ] **Step 5: Add deterministic replay and cache mismatch tests**

Assert identical transition/reward for identical row, hashes, and seed; reject revised cache with different evidence/model/prompt hash; reject labels from a locked test store during any tuning method.

- [ ] **Step 6: Run environment tests and commit**

Run: `python -m pytest tests/rl/test_environment.py tests/rl/test_reward.py tests/rl/test_cache_visibility.py -v`

Expected: PASS.

```bash
git add src/iira2/rl tests/rl
git commit -m "feat: add leakage-safe macro environment"
```

### Task 3: Fixed fusion and uncertainty-router baselines

**Files:**
- Create: `src/iira2/baselines/fusion.py`
- Create: `src/iira2/baselines/threshold.py`
- Create: `src/iira2/baselines/__init__.py`
- Create: `configs/experiment/fixed_fusion.yaml`
- Create: `configs/experiment/reliability_fusion.yaml`
- Create: `configs/experiment/uncertainty_router.yaml`
- Test: `tests/baselines/test_fusion.py`
- Test: `tests/baselines/test_threshold.py`

**Interfaces:**
- Consumes: direct probability, external calibrated probability/reliability, development records.
- Produces: `FixedMix`, `ReliabilityLogitFusion`, `UncertaintyRouter`, and frozen `BaselineArtifact`.

- [ ] **Step 1: Write failing fusion and development-only fit tests**

```python
def test_fixed_mix_matches_declared_formula() -> None:
    assert FixedMix(weight=0.25)(0.4, 0.8) == pytest.approx(0.5)


def test_threshold_router_cannot_fit_on_test() -> None:
    with pytest.raises(LockedSplitError):
        UncertaintyRouter.fit(test_records(), minimum_coverage=0.8)
```

- [ ] **Step 2: Run baseline tests and confirm missing code**

Run: `python -m pytest tests/baselines/test_fusion.py tests/baselines/test_threshold.py -v`

Expected: FAIL because baseline modules do not exist.

- [ ] **Step 3: Implement the three fixed baselines**

`FixedMix` applies `(1-w)*p_agent + w*p_external`. `ReliabilityLogitFusion` combines clipped logits with a reliability-derived external weight and development-fitted coefficients. `UncertaintyRouter` queries when direct entropy/margin crosses a development-selected threshold, then uses the frozen revised response. All outputs are bounded and provenance records the fit split.

- [ ] **Step 4: Add identical-budget evaluation helpers**

Given a query budget, choose thresholds/weights on development data only and report achieved query/abstain rates. Evaluation never adjusts them to match test performance.

- [ ] **Step 5: Run baseline tests and commit**

Run: `python -m pytest tests/baselines/test_fusion.py tests/baselines/test_threshold.py -v`

Expected: PASS.

```bash
git add src/iira2/baselines configs/experiment/fixed_fusion.yaml configs/experiment/reliability_fusion.yaml configs/experiment/uncertainty_router.yaml tests/baselines
git commit -m "feat: add fixed macro baselines"
```

### Task 4: Shared policy/value network and algorithm protocol

**Files:**
- Create: `src/iira2/controllers/network.py`
- Create: `src/iira2/controllers/base.py`
- Create: `src/iira2/rl/batches.py`
- Test: `tests/controllers/test_network.py`
- Test: `tests/rl/test_batches.py`

**Interfaces:**
- Consumes: batched `ControllerObservation` features and valid-action masks.
- Produces: `MacroPolicyValueNet`, `PolicyOutput`, `MacroBatch`, and `ControllerAlgorithm` protocol.

- [ ] **Step 1: Write failing action-mask and shape tests**

```python
def test_policy_masks_invalid_action() -> None:
    net = MacroPolicyValueNet(input_dim=4, hidden_dim=8, action_count=3)
    output = net(torch.zeros(2, 4), torch.tensor([[True, False, True], [True, True, True]]))
    assert torch.isneginf(output.logits[0, 1])
    assert output.value.shape == (2,)
```

- [ ] **Step 2: Run shared-controller tests and confirm missing modules**

Run: `python -m pytest tests/controllers/test_network.py tests/rl/test_batches.py -v`

Expected: FAIL because network/batch code does not exist.

- [ ] **Step 3: Implement a small shared controller**

Use `LayerNorm -> Linear -> GELU -> Linear` trunk with separate three-logit policy and scalar value heads. Apply boolean action mask before `Categorical`. Initialize deterministically from the run seed. Inputs must be detached; the controller does not hold agent/evidence module references.

- [ ] **Step 4: Define common algorithm calls**

```python
class ControllerAlgorithm(Protocol):
    def select(self, observation: ControllerObservation, deterministic: bool) -> MacroAction: ...
    def update(self, batch: MacroBatch) -> Mapping[str, float]: ...
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
```

`MacroBatch` stores feature tensors, action indices, rewards, covered/query flags, behavior log-probs/logits when required, group IDs, and detached metadata refs. Validate shapes and finiteness.

- [ ] **Step 5: Run shared tests and commit**

Run: `python -m pytest tests/controllers/test_network.py tests/rl/test_batches.py -v`

Expected: PASS.

```bash
git add src/iira2/controllers src/iira2/rl/batches.py tests/controllers tests/rl/test_batches.py
git commit -m "feat: add shared macro policy network"
```

### Task 5: Cross-fitted Supervised Router

**Files:**
- Create: `src/iira2/baselines/supervised_router.py`
- Create: `src/iira2/baselines/router_targets.py`
- Create: `configs/experiment/supervised_router.yaml`
- Test: `tests/baselines/test_supervised_router.py`
- Test: `tests/baselines/test_router_targets.py`

**Interfaces:**
- Consumes: training observations and cached utility for all three actions.
- Produces: `RouterTarget`, `build_crossfit_router_targets(...)`, and `SupervisedRouter` implementing `ControllerAlgorithm` selection.

- [ ] **Step 1: Write failing target and no-test-label tests**

```python
def test_router_target_selects_highest_utility() -> None:
    target = RouterTarget.from_utilities(direct=-0.25, query=-0.05, abstain=-0.20)
    assert target.action is MacroAction.QUERY_AND_REVISE


def test_router_targets_reject_test_records() -> None:
    with pytest.raises(LockedSplitError):
        build_crossfit_router_targets(test_records(), folds=5, seed=42)
```

- [ ] **Step 2: Run router tests and confirm missing implementation**

Run: `python -m pytest tests/baselines/test_supervised_router.py tests/baselines/test_router_targets.py -v`

Expected: FAIL because router modules do not exist.

- [ ] **Step 3: Build training-only utility targets**

Evaluate all three cached terminal utilities using training labels only. Partition subjects into folds; any learned preprocessing/class weighting for a row is fit without that row's subject. Resolve exact utility ties with declared priority `DIRECT_COMMIT`, then `ABSTAIN`, then `QUERY_AND_REVISE` to avoid inflating query usage.

- [ ] **Step 4: Implement cost-sensitive classifier training**

Train the shared policy logits with weighted cross-entropy, derive class weights from training targets, early-stop on development Brier/coverage under the frozen reward, and save target/fold hashes. Evaluation uses deterministic argmax.

- [ ] **Step 5: Run router tests and commit**

Run: `python -m pytest tests/baselines/test_supervised_router.py tests/baselines/test_router_targets.py -v`

Expected: PASS.

```bash
git add src/iira2/baselines/supervised_router.py src/iira2/baselines/router_targets.py configs/experiment/supervised_router.yaml tests/baselines
git commit -m "feat: add cross-fitted supervised router"
```

### Task 6: Contextual Bandit baseline

**Files:**
- Create: `src/iira2/baselines/contextual_bandit.py`
- Create: `configs/experiment/contextual_bandit.yaml`
- Test: `tests/baselines/test_contextual_bandit.py`

**Interfaces:**
- Consumes: pre-query context, selected action, observed terminal reward.
- Produces: `PerActionRidgeBandit` implementing selection/update/state persistence.

- [ ] **Step 1: Write failing per-action update and deterministic-eval tests**

```python
def test_bandit_updates_only_selected_action() -> None:
    bandit = PerActionRidgeBandit(feature_dim=2, action_count=3, ridge=1.0)
    before = bandit.matrices_copy()
    bandit.update_one(np.array([1.0, 0.0]), action=1, reward=0.5)
    assert np.array_equal(before[0], bandit.matrices_copy()[0])
    assert not np.array_equal(before[1], bandit.matrices_copy()[1])


def test_eval_uses_no_exploration(trained_bandit, observation) -> None:
    assert trained_bandit.select(observation, deterministic=True) == trained_bandit.greedy_action(observation)
```

- [ ] **Step 2: Run bandit tests and confirm missing code**

Run: `python -m pytest tests/baselines/test_contextual_bandit.py -v`

Expected: FAIL because contextual bandit code does not exist.

- [ ] **Step 3: Implement regularized per-action reward models**

Maintain one ridge system per action, update only the selected action, and compute expected reward plus configured UCB bonus. Training supports seeded epsilon or UCB exploration; evaluation has neither. Persist matrices/vectors, feature schema, exploration schedule, seed, and update count.

- [ ] **Step 4: Add numerical and checkpoint tests**

Assert finite solve under collinearity, identical action sequence after restore, no label in input context, and deterministic tie-breaking by action enum order.

- [ ] **Step 5: Run bandit tests and commit**

Run: `python -m pytest tests/baselines/test_contextual_bandit.py -v`

Expected: PASS.

```bash
git add src/iira2/baselines/contextual_bandit.py configs/experiment/contextual_bandit.yaml tests/baselines/test_contextual_bandit.py
git commit -m "feat: add contextual bandit baseline"
```

### Task 7: PPO-Options

**Files:**
- Create: `src/iira2/rl/ppo_options.py`
- Create: `configs/experiment/ppo_options.yaml`
- Test: `tests/rl/test_ppo_options.py`

**Interfaces:**
- Consumes: `MacroBatch` with old log-probs, rewards, and old values.
- Produces: `PPOOptions` implementing `ControllerAlgorithm` and `ppo_loss(...) -> PPOLoss`.

- [ ] **Step 1: Write failing clipped-ratio and one-step-advantage tests**

```python
def test_ppo_clips_policy_ratio() -> None:
    loss = ppo_loss(new_logp=torch.log(torch.tensor([2.0])), old_logp=torch.zeros(1),
                    advantage=torch.ones(1), values=torch.zeros(1), returns=torch.ones(1),
                    clip_ratio=0.2, value_weight=0.5, entropy_weight=0.0)
    assert loss.policy.item() == pytest.approx(-1.2)


def test_one_step_return_equals_reward() -> None:
    assert torch.equal(one_step_returns(torch.tensor([0.4, -0.2])), torch.tensor([0.4, -0.2]))
```

- [ ] **Step 2: Run PPO tests and confirm missing implementation**

Run: `python -m pytest tests/rl/test_ppo_options.py -v`

Expected: FAIL because PPO-Options does not exist.

- [ ] **Step 3: Implement clipped policy/value update**

Compute normalized one-step advantages, clipped surrogate, clipped or MSE value loss, entropy bonus, approximate KL, gradient clipping, minibatch epochs, and KL early stop. Keep sequence-shaped batch support but do not invent intermediate rewards or GAE steps for the one-step environment.

- [ ] **Step 4: Add finite-gradient, mask, and restore tests**

Assert nonzero finite controller gradient, invalid actions never sampled, KL stop triggers, deterministic evaluation, and state restore reproduces logits and optimizer counters.

- [ ] **Step 5: Run PPO tests and commit**

Run: `python -m pytest tests/rl/test_ppo_options.py -v`

Expected: PASS.

```bash
git add src/iira2/rl/ppo_options.py configs/experiment/ppo_options.yaml tests/rl/test_ppo_options.py
git commit -m "feat: add PPO options controller"
```

### Task 8: Terminal-action-only Macro-CISPO

**Files:**
- Create: `src/iira2/rl/macro_cispo.py`
- Create: `src/iira2/rl/group_rollout.py`
- Create: `configs/experiment/macro_cispo.yaml`
- Test: `tests/rl/test_macro_cispo.py`
- Test: `tests/rl/test_macro_cispo_gradient_boundary.py`

**Interfaces:**
- Consumes: `K=3` macro rollouts per observation, frozen behavior-policy logits, current controller logits.
- Produces: `MacroCISPO`, `CISPOLoss`, and `group_relative_advantage(rewards, group_ids)`.

- [ ] **Step 1: Write failing group-advantage and zero-variance tests**

```python
def test_group_advantage_is_reward_minus_group_mean() -> None:
    rewards = torch.tensor([1.0, 0.0, -1.0])
    assert torch.equal(group_relative_advantage(rewards, torch.zeros(3, dtype=torch.long)), rewards)


def test_zero_advantage_group_is_skipped() -> None:
    result = macro_cispo_loss(current_logits=torch.zeros(3, 3, requires_grad=True),
                              behavior_logits=torch.zeros(3, 3), actions=torch.tensor([0, 1, 2]),
                              rewards=torch.ones(3), group_ids=torch.zeros(3, dtype=torch.long),
                              clip_log_ratio=1.0, max_is_weight=2.0, kl_weight=0.01,
                              entropy_weight=0.01)
    assert result.valid_group_count == 0
    assert result.zero_advantage_group_count == 1
```

- [ ] **Step 2: Write the failing gradient-boundary test**

```python
def test_only_current_macro_logits_receive_gradient() -> None:
    current = torch.zeros(3, 3, requires_grad=True)
    behavior = torch.zeros(3, 3, requires_grad=True)
    qwen_internal = torch.tensor(1.0, requires_grad=True)
    evidence_internal = torch.tensor(1.0, requires_grad=True)
    result = macro_cispo_loss(current, behavior.detach(), torch.tensor([0, 1, 2]),
                              torch.tensor([0.0, 1.0, -1.0]), torch.zeros(3, dtype=torch.long),
                              1.0, 2.0, 0.01, 0.01)
    (result.total + 0.0 * qwen_internal.detach() + 0.0 * evidence_internal.detach()).backward()
    assert current.grad is not None and current.grad.norm() > 0
    assert behavior.grad is None
    assert qwen_internal.grad is None
    assert evidence_internal.grad is None
```

- [ ] **Step 3: Run CISPO tests and confirm missing implementation**

Run: `python -m pytest tests/rl/test_macro_cispo.py tests/rl/test_macro_cispo_gradient_boundary.py -v`

Expected: FAIL because Macro-CISPO does not exist.

- [ ] **Step 4: Implement the appendix-aligned macro loss**

For each sample group, compute `A_i = R_i - mean(R_group)`. Gather current and detached behavior log-prob only for the selected terminal macro action. Compute `log_w = current_logp - behavior_logp`, clip log ratio to `[-c_log, c_log]`, exponentiate, cap at `c_is`, and use `-mean(w_hat.detach() * A.detach() * current_logp)`. Add categorical `KL(pi_theta || pi_beta)` and subtract controller entropy. No Qwen token log-prob, KBCSv2 tensor, revised-response token, or internal event is accepted by the loss signature.

- [ ] **Step 5: Implement K=3 rollout collection and behavior snapshots**

Sample three macro actions independently from frozen `pi_beta` for each observation, execute each against immutable caches, group by observation ID, and record exact behavior logits/action log-prob/reward. Refresh `pi_beta` every 50 optimizer updates by default and persist snapshot/update identity. Duplicate sampled actions are retained; zero-reward-variance groups are counted and skipped.

- [ ] **Step 6: Add clipping, KL, entropy, snapshot, and replay tests**

Assert extreme ratios stay finite/capped, behavior tensors remain gradient-free, refresh happens exactly on schedule, resume restores the same behavior snapshot, K is exactly three in the main config, and only terminal macro action fields enter `macro_cispo_loss`.

- [ ] **Step 7: Run CISPO tests and commit**

Run: `python -m pytest tests/rl/test_macro_cispo.py tests/rl/test_macro_cispo_gradient_boundary.py -v`

Expected: PASS.

```bash
git add src/iira2/rl/macro_cispo.py src/iira2/rl/group_rollout.py configs/experiment/macro_cispo.yaml tests/rl
git commit -m "feat: add terminal macro CISPO"
```

### Task 9: Unified trainer, checkpoint/resume, and four-algorithm CLI

**Files:**
- Create: `src/iira2/rl/trainer.py`
- Create: `src/iira2/rl/checkpoint.py`
- Create: `src/iira2/cli/train_controller.py`
- Create: `src/iira2/cli/evaluate_controller.py`
- Create: `docs/RL_DESIGN.md`
- Test: `tests/rl/test_trainer.py`
- Test: `tests/rl/test_checkpoint.py`
- Test: `tests/rl/test_four_algorithm_parity.py`

**Interfaces:**
- Consumes: validated config, frozen caches, algorithm factory, train/development records.
- Produces: `ControllerTrainer`, `ControllerCheckpoint`, `create_algorithm(config) -> ControllerAlgorithm`, and compact training/evaluation status.

- [ ] **Step 1: Write failing algorithm-factory and resume tests**

```python
@pytest.mark.parametrize("name", [
    "macro_cispo", "supervised_router", "contextual_bandit", "ppo_options"
])
def test_factory_builds_each_p0_algorithm(name, config) -> None:
    config.controller.algorithm = name
    assert create_algorithm(config) is not None


def test_resume_restores_rng_and_next_action(trainer, checkpoint_path) -> None:
    trainer.save(checkpoint_path)
    expected = trainer.sample_next_action()
    restored = ControllerTrainer.load(checkpoint_path)
    assert restored.sample_next_action() == expected
```

- [ ] **Step 2: Run trainer tests and confirm missing orchestration**

Run: `python -m pytest tests/rl/test_trainer.py tests/rl/test_checkpoint.py tests/rl/test_four_algorithm_parity.py -v`

Expected: FAIL because trainer/checkpoint/factory code does not exist.

- [ ] **Step 3: Implement shared training/evaluation lifecycle**

Load only frozen cache identities, instantiate the selected algorithm, iterate deterministic training records, write compact update metrics, validate on the development split, and checkpoint policy/value/optimizer/scheduler/RNG/config/feature-schema/cache hashes. Reject checkpoint/config/cache mismatch on resume.

- [ ] **Step 4: Implement frozen-model hash and gradient audits**

At startup and completion, hash Qwen/KBCSv2 artifacts. During one smoke update, assert controller gradient norm is finite/nonzero and no parameter outside controller exists in optimizer groups. Store audit results in `internal/` and compact pass/fail only in status.

- [ ] **Step 5: Run parity and full controller tests**

Run: `python -m pytest tests/controllers tests/baselines tests/rl -v`

Expected: PASS; all four algorithms consume the same observation schema, action enum, reward implementation, and evaluator fixture.

- [ ] **Step 6: Run one synthetic CLI smoke per P0 algorithm**

```powershell
$algorithms = @('supervised_router', 'contextual_bandit', 'ppo_options', 'macro_cispo')
foreach ($algorithm in $algorithms) {
    python -m iira2.cli.train_controller --config configs/experiment/$algorithm.yaml `
        data.synthetic=true data.limit=32 runtime.profile=probe_1npu experiment.mode=train
    if ($LASTEXITCODE -ne 0) { throw "Smoke failed: $algorithm" }
}
```

Expected: each algorithm completes one finite update/evaluation, writes compact status, and leaves model/cache hashes unchanged.

- [ ] **Step 7: Commit the unified controller phase**

```bash
git add src/iira2/rl src/iira2/cli/train_controller.py src/iira2/cli/evaluate_controller.py docs/RL_DESIGN.md tests/rl
git commit -m "feat: train and resume all P0 controllers"
```
