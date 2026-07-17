"""
NSL-KDD 开集入侵检测 WebUI 后端 (FastAPI)。

接口：
  GET  /                 → 前端页面
  GET  /api/overview     → 数据集概览 (训练/测试分布、未知攻击、数据极限)
  POST /api/evaluate     → 批量评估 (跑测试集,返回 F1/P/R、各攻击检出率、混淆矩阵)

启动: uv run uvicorn webapp.app:app --reload
"""
import os, sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_csv

app = FastAPI(title="NSL-KDD 开集入侵检测")

# 静态文件 (前端)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(ROOT, "KDDTrain+.txt")
TEST_FILE = os.path.join(ROOT, "train_test")

# 懒加载模型 (首次 evaluate 时加载)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from infer import Predictor
        _predictor = Predictor()
    return _predictor


class EvaluateRequest(BaseModel):
    test_file: str | None = None  # 默认用 train_test


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/overview")
def overview():
    """数据集概览：训练/测试集分布、未知攻击、数据极限说明。"""
    Xtr_df, ytr = load_csv(TRAIN_FILE)
    Xte_df, yte = load_csv(TEST_FILE)
    known = sorted(set(ytr))
    unknown = sorted(set(yte) - set(ytr))

    from collections import Counter
    ctr = Counter(ytr); cte = Counter(yte)
    train_dist = [{"attack": c, "n": ctr[c]} for c in known]
    test_known_dist = [{"attack": c, "n": cte.get(c, 0)} for c in known]

    # 未知攻击详情
    unknown_detail = []
    # 已知数据极限说明：与已知类特征重叠的未知攻击
    overlap_attacks = {"mailbomb", "saint", "snmpguess", "snmpgetattack", "worm", "udpstorm"}
    for c in unknown:
        unknown_detail.append({
            "attack": c, "n": cte[c],
            "is_overlap": c in overlap_attacks,
        })

    return {
        "train_size": len(ytr),
        "test_size": len(yte),
        "n_known": len(known),
        "n_unknown": len(unknown),
        "known_classes": known,
        "unknown_attacks": unknown_detail,
        "train_dist": train_dist,
        "test_known_dist": test_known_dist,
        "feature_dim": Xtr_df.shape[1],
        "overlap_attacks": sorted(overlap_attacks),
        "note": "重叠未知攻击在原始41特征上与已知类100%重叠，属NSL-KDD数据极限，无法检出",
    }


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    """批量评估：加载模型跑测试集，返回完整指标。"""
    test_file = req.test_file or TEST_FILE
    if not os.path.exists(test_file):
        raise HTTPException(404, f"测试集不存在: {test_file}")
    try:
        p = get_predictor()
        res = p.evaluate(test_file)
        return res
    except Exception as e:
        import traceback
        raise HTTPException(500, f"评估失败: {e}\n{traceback.format_exc()}")


@app.get("/api/test-files")
def test_files():
    """列出可用的测试集文件。"""
    files = []
    # 根目录 train_test
    if os.path.exists(TEST_FILE):
        files.append({"name": "train_test (KDDTest+)", "path": TEST_FILE})
    # Test/ 目录下
    test_dir = os.path.join(ROOT, "Test")
    if os.path.isdir(test_dir):
        for f in sorted(os.listdir(test_dir)):
            if f.endswith(".txt"):
                files.append({"name": f"Test/{f}", "path": os.path.join(test_dir, f)})
    return files


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
