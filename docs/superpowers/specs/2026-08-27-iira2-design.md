# IIRA 2.0 设计规格：KBCSv2、Macro-CISPO 与 Ascend 离线框架

日期：2026-08-27  
状态：待用户书面审阅  
目标平台：Ascend 910C，单卡约 64 GiB，任务最多使用 7 张 NPU  
主模型：`Qwen/Qwen3.8-27B`  
医学视觉 verifier：`microsoft/rad-dino`  

## 1. 目标与科学问题

IIRA 2.0 从零实现，不增量修补旧 IIRA。旧代码与旧论文只用于复现口径、检查历史行为和定位已知错误。

核心问题是：

> 对于冻结且未做医学任务微调的 Qwen3.8-27B，一个以宏动作为决策单位的 RL controller，能否学会在必要时调用独立训练、独立校准并提供空间定位的 KBCSv2，从而改善概率质量、选择性决策和跨数据集可靠性？

主研究链路：

```text
胸片 + pathology
    -> 冻结 Qwen 初始判断 p0
    -> Macro controller 选择动作
       -> DIRECT_COMMIT
       -> QUERY_AND_REVISE -> 冻结 KBCSv2 -> 冻结 Qwen 结构化复判
       -> ABSTAIN
    -> 最终概率/选择性结果
    -> Brier、校准、查询行为、定位依赖和外部验证
```

明确不做：

- 不把报告生成作为主任务；
- 不把旧 KBCS 的 CAM 强度当疾病概率；
- 不把 Qwen SelfCheck 称为 external evidence；
- 不把 VinDr 声称为 patient-disjoint；
- 不在主实验中微调 Qwen；
- 不在数据、证据和校准审计通过前进行大规模 RL；
- 不声称临床部署、医生获益或强因果结论。

## 2. 目录与资产边界

所有项目资产位于 `${IIRA2_ROOT}`，代码和大文件分离：

```text
${IIRA2_ROOT}\
├── iira2\                         # 新 Git 仓库，只含代码、配置、小型测试夹具和文档
├── models\
│   ├── qwen3.8-27b\
│   ├── rad-dino\
│   └── optional\
├── datasets\
│   ├── restricted\
│   │   ├── mimic-cxr-jpg-2.1.0\
│   │   ├── mimic-cxr-reports-2.1.0\
│   │   ├── vindr-cxr-1.0.0\
│   │   └── mimic-cxr-ext-ils-1.0.0\
│   └── derived\
├── wheelhouse\                    # 按目标Ascend 运行环境 OS/架构/Python/CANN 固定
├── offline_bundle\                # 可复制进无网Ascend 运行环境的非受限资产
├── downloads\                     # .partial、断点和下载状态
├── manifests\                     # 全局 SHA256、许可、revision 和状态
├── IIRA-main.zip                  # 旧代码，仅参考
└── 论文和 PRE 文件                # 原始材料，保持不变
```

约束：

- `models/`、`datasets/`、`wheelhouse/`、`offline_bundle/` 不进入 Git；
- restricted data 不进入通用离线包，不复制到日志或测试夹具；
- 新代码不得 import `IIRA-main.zip` 或解压后的旧模块；
- 旧代码中的算法和行为只能通过重新实现、显式测试和来源注释进入新系统。

## 3. 新代码架构

```text
iira2/
├── src/iira2/
│   ├── agents/
│   │   ├── base.py
│   │   ├── qwen38.py
│   │   └── probability.py
│   ├── beliefs/
│   │   ├── schema.py
│   │   └── revision.py
│   ├── evidence/
│   │   ├── base.py
│   │   ├── kbcs_v2.py
│   │   ├── classifier.py
│   │   ├── localizer.py
│   │   ├── reliability.py
│   │   └── cache.py
│   ├── controllers/
│   │   ├── base.py
│   │   ├── features.py
│   │   └── macro_policy.py
│   ├── rl/
│   │   ├── env.py
│   │   ├── trajectories.py
│   │   ├── macro_cispo.py
│   │   └── ppo_options.py
│   ├── baselines/
│   │   ├── supervised_router.py
│   │   └── contextual_bandit.py
│   ├── data/
│   │   ├── ontology.py
│   │   ├── mimic.py
│   │   ├── vindr.py
│   │   ├── dicom.py
│   │   └── audits.py
│   ├── calibration/
│   ├── evaluation/
│   ├── interventions/
│   ├── runtime/
│   │   ├── device.py
│   │   ├── distributed.py
│   │   └── offline.py
│   └── cli/
├── configs/
├── scripts/
├── tests/
├── docs/
├── manifests/
├── experiments/
└── outputs/
```

上层代码只能依赖抽象接口，不得直接依赖 Qwen、RAD-DINO 或 NPU 专用类。

## 4. 冻结与可训练边界

主实验按顺序冻结参数：

1. 训练 KBCSv2 分类头、定位头和可靠性估计器；
2. 在独立 calibration split 拟合 calibrator；
3. 固定 KBCSv2 checkpoint、calibrator 和 evidence cache；
4. 加载并冻结 Qwen3.8；
5. 预计算 Qwen direct response 和 query-conditioned response cache；
6. RL 阶段只训练 macro controller 的 policy/value 参数。

必须审计：

```text
Qwen parameters requiring grad = 0
KBCSv2 parameters requiring grad during RL = 0
Controller parameters requiring grad > 0
Controller gradient norm > 0 on smoke batch
Qwen/KBCSv2 checkpoint hash before RL == hash after RL
```

后续对照实验允许：

- controller supervised warm-start；
- Qwen LoRA-SFT；
- Qwen RL-LoRA；
- RAD-DINO partial/full fine-tuning。

这些均使用独立配置和表格，不能混入 frozen-Qwen 主结果。

## 5. Qwen3.8 官方适配

固定模型：

```text
model_id: Qwen/Qwen3.8-27B
revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
license: Apache-2.0
observed_weight_size: 51.75 GiB safetensors
```

该 revision 的配置声明：

```text
architecture: Qwen3_5ForConditionalGeneration
model_type: qwen3_5
text hidden size: 5120
text layers: 64
vision hidden size: 1152
```

这些数值只进入 manifest 和测试预期，业务代码不得硬编码。Adapter 必须从官方 config 和模型对象动态发现结构。

下载前先验证官方依赖版本。由于模型 config 指向 `transformers 5.8.0.dev0`，不得使用宽泛的 `transformers>=...`。下载器保存：

- 完整 snapshot；
- revision SHA；
- config、processor、tokenizer、chat template、generation config；
- README、LICENSE；
- 官方需要的 repository code；
- 文件级 SHA256 manifest。

`Qwen38Adapter` 提供：

```python
class MultimodalAgentAdapter:
    def prepare_inputs(...): ...
    def initial_probability(...): ...
    def direct_decision(...): ...
    def revise_with_evidence(...): ...
    def detached_state_features(...): ...
```

主配置：

```yaml
agent:
  backend: qwen38
  trainable: false
  local_files_only: true
  thinking:
    enabled: false
    preserve_thinking: false
  probability_mode: constrained_binary_logprob
```

概率提取使用固定 prompt 和正/负候选 token sequence 的归一化 log-prob，不把自由文本百分数作为主概率。记录候选 token、模板、归一化公式和模型 revision。

不保存自由文本 chain-of-thought。结构化复判只保存最终字段、概率和可审计输入摘要。

## 6. KBCSv2

固定 backbone：

```text
model_id: microsoft/rad-dino
revision: 110cbc18d5133582e320b43d53bf5c44e410c936
license: MIT，模型卡同时注明 research only / not clinical practice
architecture: Dinov2Model
hidden size: 768
patch size: 14
official image size: 518
```

KBCSv2 是训练得到的医学视觉 sensor：

```text
RAD-DINO CLS token
    -> multi-label classification head
    -> raw pathology logits
    -> independent calibrator
    -> disease probabilities

RAD-DINO patch tokens
    -> pathology-conditioned localization decoder
    -> spatial map
    -> mask/box proposals
    -> localization score

classification/localization features
    -> reliability estimator
    -> estimated P(evidence is correct)
```

三个量禁止互换：

```text
disease probability != localization score != reliability
```

主训练采用两阶段候选选择：

1. 冻结 RAD-DINO，训练分类和定位 head；
2. 仅在 MIMIC development validation 上比较是否解冻末端 block；
3. 选择一个 checkpoint 后锁死，再生成 evidence cache；
4. VinDr test 不参与训练、阈值、校准或模型选择。

可靠性定义为 calibration split 上训练的 correctness estimator：输入包括 pathology、calibrated margin、预测熵、localization score 和缺失 ROI 标记，输出 `[0,1]`。它不得读取 test outcome。

统一 schema：

```python
@dataclass(frozen=True)
class ExternalEvidence:
    pathology: str
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
    metadata: dict
```

无有效 ROI 时两个 ROI 字段均为 `None`；不得用全图框伪装定位结果。

## 7. 数据与标签

受限数据由用户通过官方账号下载到 `${IIRA2_ROOT}\datasets\restricted`。代码提供官方 URL 清单、断点续传命令、凭据读取方式、目录检查、哈希和状态报告，但不保存凭据、不绕过条款、不使用第三方镜像。

数据优先级：

```text
P0 MIMIC-CXR-JPG 2.1.0
P0 VinDr-CXR 1.0.0
P1 MIMIC-CXR reports 2.1.0
P1 MIMIC-CXR-Ext-ILS 1.0.0
```

报告文本不是图像分类主链路的硬依赖；只有 evidence-conditioned text 或报告实验启用时才进入运行路径。

主 pathology ontology：

```text
Atelectasis
Cardiomegaly
Consolidation
Edema
Pleural Effusion
Pneumonia
Lung Opacity
Pneumothorax
```

每次实验固定 ontology hash、原始映射、uncertain policy 和样本计数。MIMIC 默认 `uncertain_policy=ignore`；`u_zero/u_one` 只能作为独立消融。

Split：

- MIMIC 使用官方 split，并断言 subject/study 不交叉；
- MIMIC train 用于 sensor 训练；validation 分成 model-selection 与 calibration 两个互斥子集；
- MIMIC test 只用于内部 locked evaluation；
- VinDr 使用官方 train/test；
- VinDr radiologist rows 先按 image_id 聚合再处理；
- VinDr test 永久锁死；
- exact hash、perceptual hash 和 image ID overlap 均需审计。

RAD-DINO 已在 MIMIC-CXR、CheXpert 和 NIH-CXR 上预训练。论文只能声称 task-specific development independence，不能称这些数据对 foundation encoder 完全不可见。

## 8. 图像预处理

原始数据只读，派生图像写入 `datasets/derived`。每个派生样本记录：

- 原始路径的匿名化 ID；
- Photometric Interpretation；
- intensity/windowing；
- resize 和 padding；
- 原始与派生尺寸；
- coordinate transform；
- 原始/派生 SHA256；
- preprocessing version。

RAD-DINO 主路径严格复现官方 preprocessing。VinDr DICOM 转换提供 checkerboard、角点和 round-trip coordinate tests，确保 ROI 能映射回原图。

## 9. Belief 与宏动作环境

```python
@dataclass(frozen=True)
class BeliefState:
    sample_id: str
    pathology: str
    p_initial: float
    p_current: float
    queried_external: bool
    evidence_id: str | None
    covered: bool
    terminal_action: str | None
```

主宏动作：

```text
DIRECT_COMMIT
QUERY_AND_REVISE
ABSTAIN
```

语义：

- `DIRECT_COMMIT`：提交冻结 Qwen 的独立校准概率；
- `QUERY_AND_REVISE`：环境查询冻结 KBCSv2，向冻结 Qwen 提供原图、有效 ROI crop、校准概率、定位分数、可靠性和 provenance，随后得到结构化复判；
- `ABSTAIN`：设置 `covered=false`，保留 p_current 供诊断，但不把 0.5 伪装成最终疾病概率。

QUERY 前，controller 不能读取 evidence probability、ROI、reliability 或 query-conditioned Qwen response。Cache 只对环境可见。

为了使训练可重放，先预计算：

```text
EvidenceCache: KBCSv2 输出
AgentResponseCache:
  - direct Qwen probability/state summary
  - query-conditioned Qwen revised probability/decision
```

主环境是高层 option policy 对应的一步 semi-MDP。宏动作内部保留完整事件 trace，但只有宏动作是可训练策略决策。

## 10. Reward 与 selective prediction

对于 covered action：

```text
R = -(p_final - y)^2 - lambda_query * I[QUERY_AND_REVISE]
```

对于 abstain：

```text
R = -lambda_abstain
```

主配置通过 MIMIC development validation 锁定 `lambda_abstain`，使 validation coverage 不低于 0.80。最终同时报告完整 risk-coverage curve，避免单一阈值选择性呈现。

`lambda_query` 是资源 regularizer，不作为论文 novelty。主表至少报告 `lambda_query=0` 和一个在 validation 锁定的预算版本。

## 11. 四个 P0 算法

所有算法共享完全相同的 observation、三个宏动作、reward、cache、split、seed 和评估器。

### 11.1 Macro-CISPO（主方法）

每个样本由当前策略产生 `K=3` 个 macro rollout。行为策略为冻结快照 `pi_beta`，默认每 50 次 optimizer update 刷新一次。

```text
A_i = R_i - mean_j(R_j)

log_w_i = log pi_theta(a_i | s_i) - log pi_beta(a_i | s_i)
w_hat_i = min(exp(clip(log_w_i, -c_log, c_log)), c_is)

L_policy = -mean_i[w_hat_i * A_i * log pi_theta(a_i | s_i)]
L = L_policy + beta_kl * KL(pi_theta || pi_beta) - lambda_h * H(pi_theta)
```

只有 macro action 的 log-prob、KL 和 entropy 进入梯度。Qwen、KBCSv2、cache 和 belief updater 全部 detach。

当组内 reward 方差为零时，该组不更新并记录 `zero_advantage_group`，不得用隐藏 reward 或 proxy 替换。

### 11.2 Supervised Router

只在 training split 上计算三个宏动作的真实 utility，标签为最高 utility action。使用交叉拟合生成 router targets，避免同一模型在同一行上生成并拟合目标。训练 cost-sensitive classifier，处理 ABSTAIN/QUERY 类别不平衡。

### 11.3 Contextual Bandit

实现每动作 ridge reward model 的 contextual bandit：根据冻结 observation 预测三个动作的期望 reward，训练阶段使用 epsilon/UCB exploration，评估阶段选择预测 reward 最大的动作。它是不需要多步 RL 的强简单基线。

### 11.4 PPO-Options

实现共享 macro policy/value controller 的 clipped PPO：

- clipped policy ratio；
- value loss；
- entropy regularization；
- advantage normalization；
- KL early-stop；
- action mask；
- checkpoint/resume。

由于主环境是一阶 option decision，GAE 退化为单步 advantage，但实现保留序列 batch 接口，供后续恢复多步动作。

## 12. 非 P0 算法

以下不阻塞首个完整版本：

- DAPO-inspired Options；
- GRPO-Options；
- Appendix-CISPO 四动作复现；
- Recurrent PPO/R2D2 多步控制器；
- IQL/CQL 离线 RL；
- Qwen LoRA + PPO/GRPO/DAPO。

DAPO 在 macro controller 上只能称 `DAPO-inspired`。只有实际进行 token-level Qwen policy optimization 时才能称完整 DAPO 对照。

## 13. 核心实验矩阵

在 locked external test 之前冻结：

```text
A1 Qwen direct
A2 Qwen direct + calibration
A3 KBCSv2 standalone raw
A4 KBCSv2 standalone calibrated
B1 Qwen + fixed fusion
B2 Qwen + reliability fusion
B3 Qwen + uncertainty threshold router
C1 Supervised Router
C2 Contextual Bandit
C3 PPO-Options
C4 Macro-CISPO
D1 KBCSv2 missing/noisy/flipped evidence stress
D2 Qwen SelfCheck baseline
E1 VinDr external evaluation
E2 localization intervention
```

Qwen 微调对照在主矩阵结束后执行，使用独立表格。

## 14. Evaluation 与统计

概率指标：Brier、NLL、ECE、AUROC、AUPRC、sensitivity、specificity。  
Policy 指标：query rate、abstain rate、coverage、belief change、correction rate、harm rate。  
Selective 指标：selective risk、risk-coverage curve。  
Localization 指标：IoU、pointing accuracy、box recall、适用时 mAP。

每个 pathology 统计：

```text
Qwen correct / KBCSv2 correct
Qwen correct / KBCSv2 wrong
Qwen wrong / KBCSv2 correct
Qwen wrong / KBCSv2 wrong
```

最终表使用 paired bootstrap 95% CI 和 paired permutation/bootstrap test。MIMIC 优先 subject-level bootstrap；VinDr 使用 image-level bootstrap并注明限制。多 seed 报告均值、标准差和每个 seed 原值。

干预覆盖：

- 所有 QUERY；
- 所有 valid ROI；
- evidence-adopted subset 仅作子分析；
- ROI blackout、gray、local mean；
- same-area random、matched control、contralateral control。

主干预固定使用 clean run 的 ROI；重新查询 localizer 是单独实验，避免 ROI 漂移混淆。

## 15. Go/No-Go gates

这些 gate 控制是否继续花费算力，不用于隐藏负结果。

### Gate 0：目标环境

- 目标Ascend 运行环境生成完整环境 JSON；
- Qwen 和 RAD-DINO 均 `local_files_only` 加载；
- 单图 inference 无隐藏网络调用；
- 1 次 controller backward/update 有限且非零；
- NPU/HCCL smoke 通过。

### Gate 1：Qwen baseline

- 结构化输出有效率至少 99%；
- 概率全部有限且位于 `[0,1]`；
- deterministic replay 的 action/probability 在声明容差内一致；
- 产生完整 Brier/NLL/AUROC 报告。

### Gate 2：KBCSv2

- classification 相比 prevalence baseline 的 paired Brier 95% CI 上界小于 0，或 macro AUROC 95% CI 下界大于 0.5；
- calibration 后 Brier/NLL 至少一项改善且另一项不显著恶化；
- localizer 高于同面积随机定位基线；
- evidence schema、ROI transform 和 checkpoint hash 全部有效。

### Gate 3：Complementarity

- development set 上至少一个不使用 test label 的固定融合/router，相比 calibrated Qwen 的 paired Brier 95% CI 上界小于 0；
- 若失败，仍交付 KBCSv2 和负结果报告，但停止大规模 Macro-CISPO。

### Gate 4：Macro RL smoke

- controller 非零梯度和参数变化；
- Qwen/KBCSv2 hash 不变；
- checkpoint/resume 后 trajectory schema 和 RNG 状态可恢复；
- 无 cache 泄漏、proxy source switch 或 test calibration。

### Gate 5：算法比较

- Macro-CISPO 必须与最佳 P0 非 RL router、PPO-Options 在相同预算下比较；
- 若没有优势，报告非改进结论，不追加无界算力；
- 只有 validation 冻结后才运行 VinDr test。

## 16. Ascend 910C 适配

目标环境必须由Ascend 运行环境内 probe 实测，不从 GPU/CUDA 环境推断。Probe 输出：

```text
OS / CPU architecture / Python
NPU model/count/memory
driver / firmware
CANN
torch / torch_npu
HCCL
available storage
runtime image digest
```

核心依赖：

```text
torch + torch_npu（严格兼容版本）
transformers（Qwen 官方兼容 revision）
accelerate / peft（仅在对应实验启用）
safetensors / tokenizers / huggingface_hub
pydicom / Pillow / opencv-python-headless
numpy / scipy / pandas / pyarrow / scikit-learn / scikit-image
hydra-core / omegaconf / pyyaml
pytest / rich / tqdm
```

`bitsandbytes`、CUDA wheel、CUDA flash-attn 不进入主 wheelhouse。vLLM-Ascend、MindIE 或其他 serving runtime 只有在版本矩阵和图像输入 smoke 通过后才加入。

资源 profile：

```text
probe_1npu: RAD-DINO、controller、最小 Qwen feasibility
qwen_2npu: Qwen inference 可行性测试
qwen_4npu: 主 Qwen inference/cache 生成配置
train_4npu: KBCSv2 或未来 LoRA 训练
max_7npu: 仅用于经 HCCL/分片验证的扩展实验
```

不默认采用 7-way tensor parallel。Qwen 的 attention/KV 结构更适合先验证 2/4 路切分。

## 17. 下载、manifest 与离线包

下载顺序：

1. 检查 `${IIRA2_ROOT}` 可用空间并生成 storage plan；
2. 下载 RAD-DINO 完整 snapshot；
3. 下载 Qwen3.8-27B 完整 snapshot；
4. 用户下载 credentialed datasets；
5. 根据目标Ascend 运行环境 probe 构建 wheelhouse；
6. 生成非受限 offline bundle；
7. 在 network-disabled 目标Ascend 运行环境验证。

每个资产使用状态：

```text
MISSING | DOWNLOADING | PARTIAL | COMPLETE | VERIFIED | BLOCKED_BY_TERMS | FAILED
```

`manifests/models.json`、`datasets.json`、`wheelhouse.json` 记录 revision、license、gated 状态、大小、文件哈希、离线加载结果和证据路径。

下载器必须支持断点续传、单实例锁和 `.partial` 状态。禁止把 partial 文件标成 complete。

## 18. 配置开关与紧凑输出协议

所有主实验、基线和消融必须通过同一入口运行：

```text
python -m iira2.cli.launch --config configs/experiment/<name>.yaml [key=value ...]
```

配置使用组合式开关，不为单个实验新增临时脚本。P0 至少提供以下配置面：

```yaml
experiment:
  name: macro_cispo_mimic
  seed: 42
  mode: train                 # train | evaluate | cache | audit
  arm: controller             # qwen_direct | kbcs_standalone | fixed_fusion |
                              # reliability_fusion | uncertainty_router |
                              # controller | self_check | intervention

data:
  development: mimic
  external_test: vindr
  pathology_set: core8
  uncertain_policy: ignore   # ignore | u_zero | u_one

agent:
  backend: qwen38
  trainable: false
  thinking_enabled: false
  probability_mode: constrained_binary_logprob

evidence:
  enabled: true
  source: kbcs_v2
  cache: true
  classifier_head: linear    # linear | mlp
  backbone_mode: frozen      # frozen | last_block | full
  localization_enabled: true
  localization_head: patch_decoder  # patch_decoder | box_head
  reliability_enabled: true

calibration:
  agent: temperature         # raw | temperature | platt | isotonic | beta
  evidence: platt

fusion:
  method: none               # none | fixed_mix | reliability_logit
  fixed_mix_weight: null

controller:
  enabled: true
  algorithm: macro_cispo     # macro_cispo | supervised_router | contextual_bandit | ppo_options
  actions: [DIRECT_COMMIT, QUERY_AND_REVISE, ABSTAIN]

reward:
  brier_weight: 1.0
  query_cost: 0.0
  abstain_cost: 0.0
  minimum_development_coverage: 0.80

stress:
  enabled: false
  evidence_noise: 0.0
  evidence_flip_rate: 0.0
  missing_evidence_rate: 0.0
  roi_corruption: 0.0

evaluation:
  bootstrap: true
  interventions: false
  save_predictions_internal: true

runtime:
  profile: qwen_4npu        # probe_1npu | qwen_2npu | qwen_4npu | train_4npu | max_7npu
  offline: true
  resume: false
  dry_run: false

output:
  console: key_metrics_only
  exchange_report: compact
  retain_internal_details: true
```

CLI 允许 `key=value` 覆盖，但必须保存合并后的 resolved config、config hash 和覆盖列表。Schema 校验在加载模型或申请 NPU 之前执行。非法组合直接失败并给出明确原因，包括：

- 主实验中 `agent.trainable=true`；
- `experiment.arm` 与 controller/fusion/evidence 开关不一致；
- `QUERY_AND_REVISE` 已启用但 evidence/cache 不可用；
- `macro_cispo` 的动作集合不是冻结的三个宏动作；
- `external_test=vindr` 与任何 train/calibrate/threshold-fit 操作同时出现；
- `runtime.profile=max_7npu` 但未提供通过的 HCCL/分片 probe；
- stress 配置进入 clean 主结果；
- 请求不存在的算法、数据源或静默 fallback。

为便于无网Ascend 运行环境内外交流，默认终端不打印逐样本 prediction、trajectory、自由文本生成或完整依赖列表，只打印固定顺序的关键指标行。每次运行生成：

```text
outputs/<run_id>/STATUS.json       # 机器可读，单个紧凑对象
outputs/<run_id>/REPORT.html       # 单屏优先，可展开但默认只显示关键指标
outputs/<run_id>/REPORT.png        # 适合直接截图
outputs/<run_id>/internal/         # Ascend 运行环境内完整审计产物，不进入紧凑交换输出
```

`STATUS.json` 使用版本化 schema，顶层只保留：

```text
schema_version, run_id, timestamp, status, phase, gate
algorithm, dataset, split, pathology_scope, seed
config_hash, code_commit, model_revisions, cache_id
environment_summary, elapsed_seconds, peak_npu_memory_gib
key_metrics, warnings, blockers, evidence_refs
```

`key_metrics` 按阶段只输出适用字段，不用 `0` 伪装缺失值：

```text
Qwen/KBCSv2: brier, nll, auroc, auprc, ece
Controller: reward, delta_brier_vs_qwen, paired_ci95, query_rate,
            abstain_rate, coverage, correction_rate, harm_rate
Localization: pointing_accuracy, box_recall, iou
Training: loss, gradient_norm, updates, zero_advantage_group_rate
Runtime: samples_per_second, peak_npu_memory_gib
```

紧凑报告只展示总体、macro average、主对照差值及其置信区间、Go/No-Go 结论和阻塞原因。`evidence_refs` 只能使用 run 目录内的相对、脱敏 artifact ID，不得包含 restricted data 绝对路径、subject/study/image ID 或原始错误堆栈。逐 pathology、逐 seed、逐样本、完整 risk-coverage curve、bootstrap samples、trajectory 和 provenance 仍写入 `internal/`，供最终统计和复现，但不刷屏，也不进入默认截图。`REPORT.html`/`REPORT.png` 必须明确显示 `COMPLETE`、`PARTIAL`、`MISSING`、`BLOCKED`，不得把未运行指标从版面中隐藏成成功。

输出协议测试至少覆盖：schema 向后可识别、缺失指标为 `null`/省略、终端无逐样本泄漏、紧凑报告与 internal 汇总一致、restricted path/PHI 不进入交换报告，以及固定尺寸截图中关键字段不截断。

## 19. 离线验收

目标Ascend 运行环境设置：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

最终必须在真实无网环境执行：

- clean wheelhouse install；
- Qwen image inference；
- Qwen direct probability；
- Qwen evidence-conditioned revision；
- RAD-DINO CLS/patch extraction；
- KBCSv2 classifier/localizer/reliability；
- MIMIC/VinDr loader；
- evidence 和 agent response cache；
- 三个宏动作；
- 四个 P0 算法各至少一次 update/eval；
- intervention；
- bootstrap evaluation；
- restricted data archive exclusion test。

Ascend 运行环境内输出紧凑的 `STATUS.json`、`REPORT.html` 和 `REPORT.png`，便于无法传出文件时截图。成功、部分、缺失、阻塞必须分开显示。

## 20. 测试

至少覆盖：

```text
test_qwen38_official_adapter.py
test_frozen_qwen.py
test_probability_extraction.py
test_kbcsv2_classifier.py
test_kbcsv2_localizer.py
test_reliability_no_test_leakage.py
test_evidence_schema.py
test_roi_coordinate_roundtrip.py
test_evidence_cache_hidden_before_query.py
test_agent_response_cache.py
test_macro_actions.py
test_macro_cispo_gradient_boundary.py
test_supervised_router.py
test_contextual_bandit.py
test_ppo_options.py
test_reward_and_abstention.py
test_mimic_split.py
test_vindr_aggregation.py
test_calibration_split.py
test_interventions.py
test_statistics.py
test_manifest_hashes.py
test_restricted_data_exclusion.py
test_npu_single_device.py
test_npu_distributed.py
test_no_network.py
test_config_switch_validation.py
test_compact_status_schema.py
test_compact_report_redaction.py
```

Regression tests 必须阻止：

- CAM/localization score 变成疾病概率；
- test 数据进入训练、阈值或校准；
- evidence cache 在 QUERY 前泄漏；
- train/eval evidence source 静默切换；
- Qwen/KBCSv2 在 controller RL 中获得梯度；
- restricted data 进入通用包。

## 21. 实施顺序

```text
P0 设计与资产/环境审计
P1 下载 RAD-DINO、Qwen 和非受限依赖
P2 新 repo 基础设施、schema、manifest 和 CLI
P3 数据 loader、ontology、split 和 DICOM pipeline
P4 Qwen 官方 adapter、概率和 response cache
P5 KBCSv2 分类、定位、可靠性和校准
P6 宏动作环境与 fixed baselines
P7 Supervised Router 和 Contextual Bandit
P8 PPO-Options
P9 Macro-CISPO
P10 locked VinDr external evaluation
P11 intervention、statistics 和 compact reports
P12 后续 Qwen 微调、GRPO、DAPO 等消融
```

## 22. 验收边界

代码完成不等于 NPU ready。交付状态分为：

```text
SOURCE_READY
ASSETS_VERIFIED
HOST_TESTED
NPU_SMOKE_PASSED
OFFLINE_READY
BLOCKED
```

只有真实 Ascend 910C Ascend 运行环境中的无网测试通过后，才能标记 `OFFLINE_READY`。本地静态测试、CUDA 测试或仅下载成功均不能替代该状态。

出现阻塞时必须记录：

```text
status
reason
exact_error
attempted
required_user_action
evidence_path
```

## 23. 已冻结决策

- 新代码完全重写，旧代码只作为参考；
- 所有数据和资产位于 `${IIRA2_ROOT}`；
- Qwen3.8-27B 是主模型，主实验冻结；
- 后续允许 Qwen 微调作为对照；
- RAD-DINO KBCSv2 是独立医学视觉 sensor；
- Macro-CISPO 是主 RL 算法；
- P0 只实现四个算法：Macro-CISPO、Supervised Router、Contextual Bandit、PPO-Options；
- Appendix-CISPO、GRPO、DAPO 和 Qwen-LoRA 属于后续消融；
- 目标平台是 Ascend 910C，最多使用 7 张 NPU；
- restricted data 由用户通过官方渠道下载；
- 所有实验走统一配置开关，非法组合失败而不是静默 fallback；
- 终端和交换报告只显示关键指标，完整审计结果保留在Ascend 运行环境内部；
- 正式大规模训练服从 Go/No-Go gates。
