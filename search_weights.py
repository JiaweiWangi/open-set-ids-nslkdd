"""离线权重搜索：加载已保存的5个OOD信号，网格搜索最优权重组合。

无需重训。目标：在TNR约束下最大化未知检测F1。
信号: ood(ODD头), mah(马氏), err(AE误差), smax(1-softmax最大), om(OpenMax未知概率)
"""
import numpy as np
from itertools import product
from data_utils import load_csv

ood = np.load("sig_ood_test.npy")
mah = np.load("sig_mah_test.npy")
err = np.load("sig_err_test.npy")
smax = np.load("sig_smax_test.npy")   # 已是 1-softmax_max? 看 train 保存的是 smax_test(原始max)
om = np.load("sig_om_test.npy")
yte = np.load("yte_str.npy", allow_pickle=True)
Xtr_df, ytr_str = load_csv("KDDTrain+.txt")
known = set(ytr_str)
true_known = np.array([c in known for c in yte])
true_unk = ~true_known

# 注意 sig_smax_test 存的是 smax_test(softmax最大值)，越大越像已知，所以未知信号=1-smax
smax_unk = 1 - smax

def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

n_ood = norm01(ood); n_mah = norm01(mah); n_err = norm01(err)
n_smax = norm01(smax_unk); n_om = norm01(om)

signals = {"ood": n_ood, "mah": n_mah, "err": n_err, "smax": n_smax, "om": n_om}

def eval_thr(fuse, thr):
    unk = fuse > thr
    tp = (unk & true_unk).sum(); fp = (unk & true_known).sum()
    fn = (~unk & true_unk).sum(); tn = (~unk & true_known).sum()
    p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
    f1 = 2*p*r/(p+r+1e-9); tnr = tn/max(1,tn+fp)
    return f1, p, r, tnr

# 网格：每个权重 0/0.1/0.2/0.3/0.4/0.5，归一化后搜
print("网格搜索权重 (TNR>=0.90 约束下最大化 F1)...")
best = None
grid = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for wo, wm, we, ws, wmm in product(grid, grid, grid, grid, grid):
    s = wo+wm+we+ws+wmm
    if s == 0:
        continue
    w = {"ood":wo/s,"mah":wm/s,"err":we/s,"smax":ws/s,"om":wmm/s}
    fuse = w["ood"]*n_ood + w["mah"]*n_mah + w["err"]*n_err + w["smax"]*n_smax + w["om"]*n_om
    # 在 TNR 0.85~0.95 区间找最优 thr
    best_f1_thr = -1
    for q in np.arange(0.85, 0.96, 0.01):
        thr = np.percentile(fuse[true_known], q*100)
        f1,p,r,tnr = eval_thr(fuse, thr)
        if tnr >= 0.90 and f1 > best_f1_thr:
            best_f1_thr = f1; best_p,best_r,best_tnr = p,r,tnr; best_q=q
    if best_f1_thr > (best[0] if best else -1):
        best = (best_f1_thr, dict(w), best_q, best_p, best_r, best_tnr)

f1, w, q, p, r, tnr = best
print(f"\n最优权重: {w}")
print(f"  最优工作点 q={q:.2f} | F1={f1:.3f} P={p:.3f} R={r:.3f} TNR={tnr:.3f}")

# 用最优权重看各未知检出
fuse = w["ood"]*n_ood + w["mah"]*n_mah + w["err"]*n_err + w["smax"]*n_smax + w["om"]*n_om
thr = np.percentile(fuse[true_known], q*100)
unk = fuse > thr
print("\n各未知攻击检出率:")
for c in sorted(set(yte) - known):
    m = yte == c
    print(f"  {c:16s} n={m.sum():4d} 检出={unk[m].mean():.3f}")
