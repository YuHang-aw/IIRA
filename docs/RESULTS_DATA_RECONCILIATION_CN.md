# 结果与数据口径核对（2026-09-18）

本文档把原论文中已经写出的数字、当前本地数据资产审计结果和仍需从
Ascend 运行记录中确认的事实分开。它不是新的实验结果，也不替代正式的
锁定评估报告。

## A. 原论文已经报告的结果

来源：仓库外部的原论文 PDF 与毕业论文最终版。以下数字属于旧实验设置，
不能直接标记为当前 4B/Macro policy 的结果。

### A.1 Primary Cohort：Qwen3-VL-8B

- 数据源：VinDr-CXR。
- 派生 cohort：4,240 张训练图像、394 张独立测试图像。
- 病种：Atelectasis、Pleural Effusion。
- 非交互 baseline：Brier `0.4422`，ECE `0.4267`。
- Temperature Scaling：Brier `0.3140`，ECE `0.2510`。
- Policy-LoRA + Prior-Mix：Brier `0.3193`，ECE `0.2774`，采用率 `0.58`。
- Policy-LoRA + KBCS-Mix：Brier `0.4568`，ECE `0.4172`，采用率 `0.58`。

### A.2 Edge Cohort：Qwen2.5-VL-3B，4-bit

- 数据源：VinDr-200 派生子集，四个病种：Pneumothorax、Cardiomegaly、
  Pleural Effusion、Consolidation。
- Policy-init、no P&G：Brier `0.4911`，ECE `0.4914`，采用率 `0.000`。
- Policy-init、Prior-Mix：Brier `0.4418`，ECE `0.4136`，采用率 `0.235`。
- Policy-LoRA、Prior-Mix：Brier `0.4029`，ECE `0.3656`，采用率 `0.385`。
- Policy-init、KBCS-Mix：Brier `0.4831`，ECE `0.4699`，采用率 `0.235`。
- Policy-LoRA、KBCS-Mix：Brier `0.4993`，ECE `0.4983`，采用率 `0.350`。

### A.3 旧干预和行为分析

- 8B 主 cohort 的 adopted-ROI masking 子集为 `N=19`，平均 Brier 增量约
  `+0.013`。
- 代表性案例的 clean 到 masked Brier 增量约 `+0.25`，属于单案例展示，
  不能替代总体统计。
- 3B 不确定区间（belief `0.4–0.6`）的 Abstain rate：initial `0.25`，
  RL-aligned `0.51`。
- 3B sweep 中，KBCS-Gate 的 Brier 约 `0.480`，KBCS-Mix 约 `0.483`；
  这些是 thesis-era / legacy 分析，不能与新 Macro-CISPO 表混写。

### A.4 原论文旧训练口径

- action set：`P&G / Claim / Abstain / Stop`。
- 每个状态 `K=3` rollouts。
- `Tmax=3`，importance clip 上限记录为 `10.0`，KL 权重 `0.1`。
- 主论文写的是 Qwen3-VL-8B，边缘实验写的是 Qwen2.5-VL-3B；旧代码还
  包含 LoRA、4-bit NF4 和 Prior-Mix/KBCS-Mix 两条证据路径。
- reward 的理论附录以 terminal negative Brier 为主。

这些设定与当前公开仓库的 9 维缓存状态、3 个 Macro action 和新的
pathwise fusion-head loss 不同。修稿时必须把旧结果标成 historical/legacy，
不能写成当前实现的验证。

## B. 当前本地数据资产的只读审计

审计只记录汇总数量，不写出患者、检查或图像标识符。

### B.1 MIMIC-CXR-JPG 2.1.0

- 当前 JPG 数量：`60,943`。
- 所有本地 JPG 均来自 AP/PA frontal pool。
- 按官方 split：train `59,604`，validation `495`，test `844`。
- train/validation/test 的本地 frontal 计数分别为 `59,604 / 495 / 844`。
- 本地样本覆盖十个患者目录前缀 `p10`–`p19`；每个前缀均有样本。
- 已存在官方 metadata、split、CheXpert、NegBio 和 test-label 小文件。
- 当前审计没有证明这 59,604 张 train JPG 是按最终论文所需的 subject-level
  抽样方案生成，也没有证明它们就是当前 Ascend 实验实际使用的清单。

### B.2 VinDr-CXR 1.0.0

- `image_labels_train.csv` 含 15,000 个训练图像标识的标签行集合，
  `image_labels_test.csv` 含 3,000 个测试图像标识的标签集合。
- 当前本地有 3,000 个 test DICOM；未发现对应的 train DICOM/PNG 图像。
- 当前资产足以核对 VinDr test 标签表和 test 图像数量，但不足以重建原论文
  的 4,240 train cohort 或任何需要 VinDr train 图像的 KBCS 训练。
- VinDr 的病种列多于原论文的两病种/四病种子集；新版必须固定 ontology、
  二值化规则、缺失/不确定标签策略后再计算指标。

## C. 已确认、待确认和缺失

### 已确认

- 原论文旧结果及其模型/数据口径如 A 节所列。
- 本地 MIMIC 与 VinDr 文件的上述汇总数量。
- 公开仓库当前只包含 Macro 核心代码和合成测试，不包含真实结果。
- Qwen snapshot 的文件级完整性记录已存在于外部资产清单中；这不等于
  当前实验已经用该模型完成推理或训练。

### 必须从 Ascend 运行记录确认

1. 当前实际模型 ID、revision、模型加载类和 dtype。特别要区分原论文的
   Qwen3-VL-8B/Qwen2.5-VL-3B 与新版 4B/Qwen3.5 路径。
2. 当前实际使用的数据 manifest：数据源、split、样本数、病种 ontology、
   label policy、subject/study 去重和校准集。
3. 四路 evidence cache 是否来自同一 split，join 后保留多少样本，缺失和
   重复 key 如何处理。
4. 已完成运行的 compact metrics：Brier、NLL、ECE、AUROC/AUPRC、query rate、
   abstain rate、coverage、correction rate、harm rate、seed 和 run hash。
5. 当前 checkpoint 的参数名/shape、训练 epoch/step、optimizer 与 resume
   状态；不需要传出权重。
6. 当前 4B 运行是否仍有 Qwen/LoRA 参与 RL，还是已经切换为只读 evidence
   cache 的 5,125 参数 MLP Macro policy。

## D. 修稿前的最小结果收集顺序

1. 先冻结当前模型 revision、数据 manifest、ontology 和 calibration split，
   生成不含样本标识符的 `run_manifest`。
2. 在同一 locked split 上输出 direct、KBCS standalone、fixed fusion、
   uncertainty router 和 Macro-CISPO 的完整概率指标。
3. 再输出 action metrics：query/abstain/coverage、纠错与伤害率，并报告
   每个 seed 的原值。
4. 在 VinDr official test 上只做一次冻结外部评估；所有阈值和 calibrator
   必须来自 development split。
5. 最后做 ROI blackout、same-area random 和 control-region 干预，报告
   paired delta Brier 与置信区间。

在 C 节事实补齐前，不应把新模型、新数据或新 Macro loss 写入旧论文的
Table 1/2；应先新增一张“旧结果 vs 当前实验口径”对照表。
