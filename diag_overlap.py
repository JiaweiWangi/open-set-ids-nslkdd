"""诊断：难类(saint/snmpguess)与最近已知类在原始特征空间的可分性。

若原始特征可分但嵌入不可分 -> 分类器丢了信息，改嵌入方案有救。
若原始特征也不可分 -> 数据极限，无解。
"""
import numpy as np
from data_utils import load_csv, build_encoder, transform, feature_dim

Xtr_df, ytr_str = load_csv("KDDTrain+.txt")
Xte_df, yte_str = load_csv("train_test")
enc = build_encoder(Xtr_df)
Xtr = transform(Xtr_df, enc)
Xte = transform(Xte_df, enc)
ytr = np.array(ytr_str); yte = np.array(yte_str)

# 难类 -> 最像的已知类（按均值最近）
hard = ["saint", "snmpguess", "snmpgetattack", "mailbomb", "httptunnel"]
known_set = set(ytr_str)

print("=== 原始特征空间可分性诊断 ===")
print(f"特征维度: {Xtr.shape[1]}")
for hc in hard:
    if hc not in set(yte_str):
        continue
    te_mask = yte == hc
    hc_mean = Xte[te_mask].mean(axis=0)
    # 找最近的已知类均值
    best_c, best_d = None, 1e18
    for kc in known_set:
        kc_mean = Xtr[ytr == kc].mean(axis=0)
        d = np.linalg.norm(hc_mean - kc_mean)
        if d < best_d:
            best_d, best_c = d, kc
    kc_mask = ytr == best_c
    # 这两个群体在原始特征上的距离分布
    d_hc_to_kc = np.linalg.norm(Xte[te_mask] - Xtr[kc_mask].mean(axis=0), axis=1)
    d_kc_to_kc = np.linalg.norm(Xtr[kc_mask] - Xtr[kc_mask].mean(axis=0), axis=1)
    # 用已知类自身的距离分位做阈值，看难类样本有多少"落进"已知类范围
    thr99 = np.percentile(d_kc_to_kc, 99)
    overlap = (d_hc_to_kc <= thr99).mean()
    print(f"\n[{hc}] n={te_mask.sum()} 最像已知类={best_c}")
    print(f"  难类到{best_c}中心: median={np.median(d_hc_to_kc):.2f} p90={np.percentile(d_hc_to_kc,90):.2f}")
    print(f"  {best_c}类自身到中心: median={np.median(d_kc_to_kc):.2f} p99={thr99:.2f}")
    print(f"  难类落入{best_c}的p99范围比例: {overlap:.3f}  (>0.5=原始特征也重叠, 无解)")

    # 再看：在分类器预测上，难类被分到哪
    # (这里用最近类均值近似，实际预测见 train 输出)
print("\n=== 结论判断 ===")
print("若 overlap 高 -> 原始特征重叠，类内未知无法区分，属数据极限")
print("若 overlap 低但嵌入仍判错 -> 嵌入丢信息，需改嵌入(对比学习/温度能量)")
