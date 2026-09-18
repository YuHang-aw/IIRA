# IIRA2 公开重建规格

## 目标

重建已确认的 Macro policy 训练语义，使代码可以上传 GitHub，并能在 CPU
合成数据与 Ascend 910B/910C NPU 环境中复现。公开版本只实现基于预计算证据
缓存的控制器；Qwen、KBCS、RAD-DINO 和受限数据由外部步骤生成缓存。

## 网络和状态

策略输入固定为 9 维，网络为 `Linear(9,64) -> Tanh -> Linear(64,64) ->
Tanh`，随后连接三个 head：3 类动作、alpha 标量和 gamma 标量。默认
PyTorch 初始化和 Adam 学习率 `5e-3`。动作索引固定为：

```text
0 DIRECT_COMMIT
1 QUERY_AND_REVISE
2 ABSTAIN
```

状态顺序和缺失值规则由 README 的 Macro contract 定义。概率先裁剪到
`[1e-7, 1-1e-7]`，缺失概率使用 `0.5`，缺失定位分数使用 `0.0`。

## 融合与奖励

`DIRECT_COMMIT` 使用 `qp`；`QUERY_AND_REVISE` 使用
`alpha*qp + (1-alpha)*(gamma*kp + (1-gamma)*qroi_p)`；`ABSTAIN` 的内部
概率为 `0.5`。奖励为负 Brier、加权 binary log-likelihood，再减去动作成本：
QUERY 默认 `0.10`，ABSTAIN 默认 `0.50`。

## CISPO 风格损失

每个样本只有一个终止 Macro action。组内 baseline 为同组 reward 均值，
不做标准差归一化。behavior log-prob detached；当前与 behavior 的 log-ratio
指数经过 `[0.8, 1.2]` 对称裁剪。policy loss、uniform KL 和 entropy 只接收
controller logits 的梯度。另有两份相同的 QUERY-masked 可微奖励负均值，
分别记为 alpha_loss 和 gamma_loss，合计系数为 2；其梯度通过最终融合概率
进入两个融合 head 及共享 trunk。Qwen、视觉模型不参与此反向传播。

状态熵按公式计算：`qp=0.72` 时为 `0.855451`，不是示例中的 `0.764`。
当前版本尚未实现完整 trainer、缓存 schema 对接与 checkpoint 兼容验证。
ranking reward 的完整公式尚未确认，不能据此声称复现 ranking_weight=1
的实验结果。历史计划仅供背景参考，不覆盖当前合同。

## 数据和公开性

四路 JSON 缓存按公开的相对 key 对齐，重复 key 应报错，缺失行要计数并写入
汇总报告。真实患者、检查、图像标识符和绝对路径不得进入日志、测试夹具或
交换报告。数据和模型路径仅通过 `IIRA2_DATA_ROOT`、`IIRA2_MODEL_ROOT`、
`IIRA2_OUTPUT_ROOT` 提供。

## 验证边界

本仓库的 CPU 测试只证明状态、融合、奖励和梯度边界。真实模型加载、数据
分布、NPU 多卡和外部测试集结果必须在目标 Ascend 设备上单独记录，不能由
静态测试推断。
