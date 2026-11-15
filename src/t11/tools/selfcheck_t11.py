"""Tự kiểm T11 khi repo KHÔNG có ground truth (gt/ chỉ có ở giảng viên).

Cách làm (ghi rõ trong báo cáo):
  1. Chạy PoC trên fixture, lấy các annotation "trông sạch": ghép stage 1, cùng class, IoU >= 0.5, không nằm trong cặp trùng.
     Các cặp (annotation, prediction) đó + các prediction mồ côi conf thấp tạo thành một "thế giới sạch".
  2. CHÈN lỗi có log với seed cố định: class_swap, box_shift, duplicate (id trung tính, KHÔNG dùng tiền tố ann_d),
     missing (xoá annotation). Đảo thứ tự dòng để PoC không thể dựa vào thứ tự.
  3. Chấm lại bằng đúng score_items của PoC, so với random và với các biến thể ablation. Metric = Recall@B,
     Precision@B (B = max(10, ceil(10% N))) — cùng công thức của evaluator.

GIỚI HẠN (trung thực): lỗi do ta tự chèn, trên dữ liệu synthetic, nên con số LẠC QUAN hơn eval thật; nó dùng để
(a) bắt bug, (b) so sánh các thành phần với nhau, (c) kiểm tra phương sai. Nó KHÔNG thay cho điểm của giảng viên.
Không đọc fixtures/*/gt/.

    python -m tools.selfcheck_t11 --seeds 30 --out reports/selfcheck_t11.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import load_config
from src.metrics import ranking as RK
from src.t11.scoring import score_items

FIX = Path("fixtures/t11_label_errors")
TYPES = ("class_swap", "box_shift", "duplicate", "missing")


def clean_world(ann, preds, images, params):
    s = score_items(ann, preds, images, params)
    a = ann.merge(s[["item_id", "iou", "pred_class", "match_stage", "s_duplicate"]], left_on="ann_id", right_on="item_id")
    ok = a[(a.match_stage == 1) & (a["class"] == a.pred_class) & (a.iou >= 0.5) & (a.s_duplicate == 0)]
    clean_ann = ok[["ann_id", "image_id", "class", "x", "y", "w", "h"]].reset_index(drop=True)
    # prediction nào đi cùng annotation sạch: ghép lại bằng IoU cao nhất trong ảnh
    from src.t11.geometry import iou_matrix
    keep = set()
    for img, g in clean_ann.groupby("image_id"):
        pp = preds[preds.image_id == img]
        if len(pp):
            M = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), pp[["x", "y", "w", "h"]].to_numpy(float))
            keep |= set(pp.index[M.argmax(axis=1)])
    orphan_low = set(preds.index[(~preds.index.isin(set(preds.index))) | False])
    # prediction không ghép được với annotation nào VÀ conf thấp = nhiễu detector thực tế, giữ lại
    all_a = {img: g[["x", "y", "w", "h"]].to_numpy(float) for img, g in ann.groupby("image_id")}
    for j, q in preds.iterrows():
        if j in keep:
            continue
        ba = all_a.get(q.image_id)
        ov = iou_matrix(np.array([[q.x, q.y, q.w, q.h]]), ba).max() if ba is not None else 0.0
        if ov < 0.3 and q.conf < 0.5:
            keep.add(j)
    return clean_ann, preds.loc[sorted(keep)].reset_index(drop=True)


def inject(clean_ann, images, rng, rate=0.15):
    n = len(clean_ann)
    n_err = max(8, int(round(rate * n)))
    idx = rng.permutation(n)[:n_err]
    kinds = [TYPES[i % 4] for i in range(n_err)]
    W = dict(zip(images.image_id, images.width)); H = dict(zip(images.image_id, images.height))
    a = clean_ann.copy().astype({"x": float, "y": float, "w": float, "h": float})
    gt, drop, extra = [], [], []
    for k, (i, kind) in enumerate(zip(idx, kinds)):
        r = a.iloc[i]
        if kind == "class_swap":
            others = [c for c in a["class"].unique() if c != r["class"]]
            a.loc[a.index[i], "class"] = rng.choice(others)
            gt.append((r.ann_id, kind))
        elif kind == "box_shift":
            f = rng.uniform(0.3, 1.0); ang = rng.uniform(0, 2 * math.pi); s = max(r.w, r.h) * f
            nx = float(np.clip(r.x + s * math.cos(ang), 0, W[r.image_id] - r.w)); ny = float(np.clip(r.y + s * math.sin(ang), 0, H[r.image_id] - r.h))
            a.loc[a.index[i], ["x", "y"]] = [nx, ny]
            gt.append((r.ann_id, kind))
        elif kind == "duplicate":
            j = rng.uniform(-3, 3, 4)
            extra.append(dict(ann_id=f"ann_i{k:03d}", image_id=r.image_id, **{"class": r["class"]}, x=r.x + j[0], y=r.y + j[1], w=max(4, r.w + j[2]), h=max(4, r.h + j[3])))
            gt.append((f"ann_i{k:03d}", kind))
        else:
            drop.append(a.index[i]); gt.append((f"img:{r.image_id}", kind))
    a = a.drop(index=drop)
    if extra:
        a = pd.concat([a, pd.DataFrame(extra)], ignore_index=True)
    a = a.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)   # đảo thứ tự dòng
    return a, pd.DataFrame(gt, columns=["item_id", "error_type"]).drop_duplicates("item_id")


def evaluate(scored, gt, n_ann):
    B = RK.budget_b(n_ann)
    order = scored.sort_values("score", ascending=False, kind="mergesort").item_id
    r = RK.recall_at_k(order, gt.item_id, B)["value"]
    p = RK.precision_at_k(order, gt.item_id, B)["value"]
    top = set(order.head(B))
    per = {t: float(gt.loc[gt.error_type == t, "item_id"].isin(top).mean()) for t in TYPES if (gt.error_type == t).any()}
    return dict(B=B, recall=r, precision=p, ceiling=min(1.0, B / len(gt)), per_type=per)


VARIANTS = {
    "PoC (đầy đủ)": {},
    "bỏ duplicate": {"w_dup": 0.0}, "bỏ class_swap": {"w_swap": 0.0}, "bỏ box_shift": {"w_shift": 0.0},
    "bỏ missing": {"w_missing": 0.0}, "bỏ unmatched": {"w_unmatched": 0.0},
    "bỏ stage-2 (ghép theo tâm)": {"shift_gate": 0.0},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--config", default="configs/lab.yaml")
    ap.add_argument("--out", default="reports/selfcheck_t11.json")
    a = ap.parse_args(argv)
    params = (load_config(a.config).get("params") or {})
    ann, preds, imgs = (pd.read_csv(FIX / f) for f in ("annotations.csv", "model_predictions.csv", "images.csv"))
    cann, cpreds = clean_world(ann, preds, imgs, params)
    print(f"thế giới sạch: {len(cann)} annotation, {len(cpreds)} prediction (từ {len(ann)} / {len(preds)})")
    res = {k: [] for k in list(VARIANTS) + ["random"]}
    for seed in range(42, 42 + a.seeds):
        rng = np.random.default_rng(seed)
        ia, gt = inject(cann, imgs, rng, a.rate)
        for name, over in VARIANTS.items():
            s = score_items(ia, cpreds, imgs, {**params, **over})
            res[name].append(evaluate(s, gt, len(ia)))
        s = score_items(ia, cpreds, imgs, params)
        s["score"] = np.random.default_rng(seed).random(len(s))
        res["random"].append(evaluate(s, gt, len(ia)))
    summary = {}
    print(f"\n{'biến thể':32s} {'Recall@B':>16s} {'Precision@B':>16s}  recall theo loại lỗi (swap/shift/dup/missing)")
    for name, lst in res.items():
        rec = np.array([x["recall"] for x in lst]); pre = np.array([x["precision"] for x in lst])
        per = {t: float(np.mean([x["per_type"].get(t, np.nan) for x in lst])) for t in TYPES}
        summary[name] = dict(recall_mean=float(rec.mean()), recall_std=float(rec.std()), precision_mean=float(pre.mean()),
                             precision_std=float(pre.std()), per_type=per, ceiling_mean=float(np.mean([x["ceiling"] for x in lst])), B=lst[0]["B"])
        print(f"{name:32s} {rec.mean():6.3f} ± {rec.std():5.3f}   {pre.mean():6.3f} ± {pre.std():5.3f}   "
              + " / ".join(f"{per[t]:.2f}" for t in TYPES))
    print(f"\ntrần oracle trung bình (min(1, B/#lỗi)): {summary['random']['ceiling_mean']:.3f}; B={summary['random']['B']}; seeds={a.seeds}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(seeds=a.seeds, rate=a.rate, params=params, summary=summary,
                                           note="Lỗi tự chèn trên dữ liệu synthetic: con số lạc quan, không thay cho eval của giảng viên."),
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
