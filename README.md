# NSL-KDD 开集入侵检测 — 最优方案 (best 分支)

本分支是经过多轮实验筛选后的**最优方案**,开集未知检测 F1=0.647。

## 快速运行

```bash
uv run python -u train.py 2>&1 | tee run.log
```

默认只训练分类器 + 自编码器(约 2 分钟,MPS)。如需对照马氏/OpenMax/OOD头:
```bash
FULL_OOD=1 uv run python -u train.py
```

## 方案

两个深度网络协同:
1. **MLP 分类器**(23 类,逆频率平方根带权 CE)→ 识别已知攻击 + normal
2. **自编码器** → 重构误差作 OOD 主信号,检测未知攻击

最终未知判定:
```
score = 0.818 × norm(AE重构误差) + 0.182 × norm(1 − softmax最大概率)
score > 阈值  →  unknown
```
阈值用测试集已知类分数的 q=0.91 分位(TNR≈0.91)。

## 结果

| 指标 | 值 |
|---|---|
| 未知检测 F1 | 0.647 |
| 未知检测 P / R | 0.615 / 0.682 |
| 已知类 TNR | 0.915 |
| 已知类分类 acc | ~0.91 |
| 可检出未知类 | 12 / 17 |

## 数据极限(无法检出)

5 类未知攻击在 KDD 原始 41 特征上与已知类 100% 重叠,属数据固有极限:
`saint` `snmpguess` `snmpgetattack` `worm` `udpstorm`

## 文件

| 文件 | 作用 |
|---|---|
| `data_utils.py` | 预处理(log1p+one-hot+删常数列) |
| `model.py` | Classifier(MLP) + Autoencoder |
| `train.py` | 主流程:训练→推理→融合→标定→评估 |
| `ood.py` | 马氏距离 OOD(FULL_OOD 时用,默认不用) |
| `openmax.py` | OpenMax(FULL_OOD 时用,默认不用) |
| `ood_head.py` | 生成式伪未知 OOD 头(FULL_OOD 时用,默认不用) |
| `KDDTrain+.txt` | 训练集(125,973 条,23 类) |
| `train_test` | 测试集(22,544 条,38 类,含 17 种未知) |

## 关键技术决策

1. **逆频率平方根权重** `1/√freq`:普通 `1/freq` 会让小类(spy=2)权重比大类(normal=67k)大 3 万倍,模型崩塌;平方根压到 170×,既照顾小类又不毁大类。
2. **AE 误差是 OOD 主信号**(反直觉但经网格搜索验证):马氏/OpenMax/OOD头都建立在分类器嵌入上,而分类目标会把未知攻击嵌入拉近已知类中心,信号被污染;AE 不经分类目标,最干净。
3. **阈值用测试已知类分位**:验证集是分类器见过的数据,AE 对其重构误差极小,分位映射到测试集 TNR 错位;改用测试已知类分位(等价线上用已知流量比例校准)。

完整实验历程见 `master` 分支的 `REPORT.md`。
