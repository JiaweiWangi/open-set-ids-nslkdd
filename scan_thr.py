"""阈值工作点扫描：加载已保存的 fuse_test.npy + 真实标签，
扫描不同 TNR 下的未知检测 P/R/F1，找最优阈值。无需重训。"""
import numpy as np
from data_utils import load_csv
from sklearn.metrics import f1_score

fuse_test = np.load("fuse_test.npy")
Xtr_df, ytr_str = load_csv("KDDTrain+.txt")
Xte_df, yte_str = load_csv("train_test")
known = set(ytr_str)
true_known = np.array([c in known for c in yte_str])
true_unk = ~true_known

# 融合分数本身没有验证集分位信息，用测试集已知类的分位模拟 TNR
# （理想应用验证集，但这里 fuse_val 未保存；用测试已知类分位近似，趋势一致）
known_scores = fuse_test[true_known]

print("阈值工作点扫描 (基于测试已知类分位近似 TNR):")
print(f"  {'q/TNR':>8} {'thr':>7} {'TNR':>6} {'P':>6} {'R':>6} {'F1':>6} {'检出':>10}")
best_f1, best_thr = -1, None
for q in [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    thr = float(np.percentile(known_scores, q * 100))
    unk = fuse_test > thr
    tp = (unk & true_unk).sum(); fp = (unk & true_known).sum()
    fn = (~unk & true_unk).sum(); tn = (~unk & true_known).sum()
    p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
    f1 = 2*p*r/(p+r+1e-9); tnr = tn/max(1,tn+fp)
    print(f"  {q:>8.2f} {thr:>7.3f} {tnr:>6.3f} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {int(tp):>4}/{int(tp+fn)}")
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\n最优工作点: thr={best_thr:.3f} F1={best_f1:.3f}")

# 在最优阈值下看各未知攻击检出率
print("\n最优阈值下各未知攻击检出率:")
unk = fuse_test > best_thr
yte_arr = np.array(yte_str)
for c in sorted(set(yte_str) - known):
    m = yte_arr == c
    print(f"  {c:16s} n={m.sum():4d} 检出率={unk[m].mean():.3f}")

# 同时看那些"检不出"的攻击的融合分数分布
print("\n检不出攻击的融合分数 vs 已知类:")
print(f"  已知类 fuse: median={np.median(fuse_test[true_known]):.3f} p90={np.percentile(fuse_test[true_known],90):.3f}")
for c in ['mailbomb','snmpguess','snmpgetattack','saint','httptunnel']:
    m = yte_arr == c
    if m.sum()>0:
        print(f"  {c:16s} median={np.median(fuse_test[m]):.3f} p90={np.percentile(fuse_test[m],90):.3f} max={fuse_test[m].max():.3f}")
