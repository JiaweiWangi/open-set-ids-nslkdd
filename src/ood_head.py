"""
生成式未知增强 OOD 头。

动机：分类器为"区分已知23类"优化，未知攻击被强行拉近最近已知类中心，
导致 saint(≈satan)、snmpguess(≈guess_passwd) 这类"重叠未知"在嵌入空间
和已知类重合，任何基于距离/重构的 OOD 都检不出。

解法：在已知类嵌入空间显式生成"伪未知"样本，训一个二分类 OOD 头
（0=已知 inlier，1=unknown），让类间空隙被标为未知区域。

伪未知生成策略：
1) 类间插值：取两个不同类样本 e_a, e_b，在它们之间凸组合（alpha~U(0.4,0.6)），
   落在类间边界——分类器最不确定的区域。
2) 类内离群：取类样本 + 高斯噪声，模拟偏离类中心但仍附近的样本。
3) 均匀扰动：从嵌入空间全局分布采样 + 大噪声，模拟完全离群。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def generate_pseudo_unknown(av, labels, n_classes, n_gen, seed=42):
    """生成伪未知嵌入样本。返回 (X, y)，y=1 表 unknown。"""
    rng = np.random.RandomState(seed)
    D = av.shape[1]
    out_x = []

    # 类中心
    centers = np.zeros((n_classes, D), dtype=np.float32)
    for c in range(n_classes):
        m = labels == c
        if m.sum() > 0:
            centers[c] = av[m].mean(axis=0)

    # 全局统计（用于扰动尺度）
    global_std = av.std(axis=0)

    n1 = int(n_gen * 0.55)  # 类间插值
    n2 = int(n_gen * 0.30)  # 类内离群
    n3 = n_gen - n1 - n2     # 全局离群

    # 1) 类间插值
    for _ in range(n1):
        ca, cb = rng.randint(n_classes), rng.randint(n_classes)
        if ca == cb:
            cb = (cb + 1) % n_classes
        alpha = rng.uniform(0.35, 0.65)
        # 在真实样本间插值，再加噪声
        ia = rng.choice(np.where(labels == ca)[0])
        ib = rng.choice(np.where(labels == cb)[0])
        e = alpha * av[ia] + (1 - alpha) * av[ib]
        e = e + rng.normal(0, 0.3) * global_std
        out_x.append(e.astype(np.float32))

    # 2) 类内离群（偏离中心）
    for _ in range(n2):
        c = rng.randint(n_classes)
        e = centers[c] + rng.normal(0, 3.0) * global_std
        out_x.append(e.astype(np.float32))

    # 3) 全局离群
    mean = av.mean(axis=0)
    for _ in range(n3):
        e = mean + rng.normal(0, 5.0) * global_std
        out_x.append(e.astype(np.float32))

    X = np.array(out_x, dtype=np.float32)
    y = np.ones(len(X), dtype=np.int64)
    return X, y


class OODHead(nn.Module):
    """在 penultimate embedding 上的二分类头：inlier vs unknown。"""
    def __init__(self, in_dim, hidden=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_ood_head(av_train, labels_train, n_classes, device, epochs=25, batch=512):
    """训 OOD 头。已知样本标 0，伪未知标 1。"""
    from train import DEVICE
    rng = np.random.RandomState(0)
    D = av_train.shape[1]

    # 已知样本（标签0），下采样到与伪未知相当
    n_known = min(len(av_train), 30000)
    idx = rng.choice(len(av_train), n_known, replace=False)
    X_known = av_train[idx]
    y_known = np.zeros(n_known, dtype=np.int64)

    # 生成等量伪未知
    X_unk, y_unk = generate_pseudo_unknown(av_train, labels_train, n_classes,
                                           n_gen=n_known, seed=42)

    X = np.vstack([X_known, X_unk])
    y = np.concatenate([y_known, y_unk])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # 类别权重（伪未知可能略少）
    w = np.array([(y==0).sum(), (y==1).sum()], dtype=np.float32)
    w = (1.0 / w) * 2
    w = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    model = OODHead(D).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=w[1] / w[0])
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)

    for ep in range(epochs):
        model.train(); tot = 0
        for xb, yb in dl:
            opt.zero_grad()
            logit = model(xb)
            loss = crit(logit, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if (ep+1) % 10 == 0:
            print(f"  [ood] epoch {ep+1}/{epochs} loss={tot/len(X):.4f}", flush=True)
    return model


@torch.no_grad()
def ood_head_scores(model, av, device=None):
    from train import DEVICE
    model.eval()
    Xt = torch.tensor(av, dtype=torch.float32, device=DEVICE)
    scores = []
    for i in range(0, len(Xt), 4096):
        logit = model(Xt[i:i+4096])
        scores.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(scores)
