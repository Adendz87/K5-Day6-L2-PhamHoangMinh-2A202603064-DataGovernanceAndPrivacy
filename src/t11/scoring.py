"""Suspicion score cho T11: điểm cao = nên đưa người review trước.

Mỗi item là một `ann_id` (nghi lỗi trên box có sẵn) hoặc `img:<image_id>` (nghi thiếu box).
Mỗi loại lỗi cho một điểm s ∈ [0,1] từ bằng chứng của detector, nhân với trọng số độ tin cậy
w_type (cấu hình, đóng băng trước eval). Điểm cuối = max các loại + 0.02 * loại mạnh thứ hai
(item có nhiều dấu hiệu xếp trên) + hạng tử rất nhỏ để phá hoà một cách xác định.

KHÔNG dùng: ID của annotation, thứ tự dòng, hay bất kỳ file gt/ nào.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import iou_matrix
from .matching import match_image

DEFAULTS = dict(
    match_iou=0.30, shift_gate=2.0, dup_iou=0.50, orphan_overlap=0.30,
    w_dup=0.95, w_swap=0.90, w_shift=0.85, w_missing=0.85, w_unmatched=0.35, missing_conf_floor=0.40,
)
TYPES = ("duplicate", "class_swap", "box_shift", "missing", "unmatched")
LABEL_MIN = 0.15   # dưới ngưỡng này item không được gán loại lỗi (chỉ để hiển thị; không ảnh hưởng thứ hạng)


def _p(params: dict | None) -> dict:
    return {**DEFAULTS, **{k: v for k, v in (params or {}).items() if k in DEFAULTS}}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fmt_box(b) -> str:
    return ",".join(f"{float(v):.1f}" for v in b)


@dataclass
class _AnnRec:
    pos: int
    img: str
    cls: str
    box: np.ndarray
    pred_idx: int
    stage: int
    iou: float
    best_any: float
    pred_cls: str = ""
    pred_conf: float = 0.0
    pred_box: np.ndarray | None = None


def score_items(ann: pd.DataFrame, preds: pd.DataFrame, images: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Trả về DataFrame, CỘT ĐẦU là item_id, gồm cả mọi annotation và mọi ảnh (`img:<id>`)."""
    P = _p(params)
    ann = ann.reset_index(drop=True)
    img_split = dict(zip(images.image_id, images["split"])) if "split" in images.columns else {}
    img_path = dict(zip(images.image_id, images["path"])) if "path" in images.columns else {}

    recs: list[_AnnRec] = []
    orphans: dict[str, list[dict]] = defaultdict(list)
    ann_by_img = {k: g for k, g in ann.groupby("image_id")}
    pred_by_img = {k: g.reset_index(drop=True) for k, g in preds.groupby("image_id")}
    empty_p = preds.iloc[0:0]

    for img_id in images.image_id:
        ga = ann_by_img.get(img_id, ann.iloc[0:0])
        gp = pred_by_img.get(img_id, empty_p)
        ab = ga[["x", "y", "w", "h"]].to_numpy(float)
        pb = gp[["x", "y", "w", "h"]].to_numpy(float) if len(gp) else np.zeros((0, 4))
        m = match_image(ab, ga["class"].tolist(), pb, gp["class"].tolist() if len(gp) else [], P["match_iou"], P["shift_gate"])
        for k, pos in enumerate(ga.index):
            j = int(m.pred_of_ann[k])
            r = _AnnRec(pos=pos, img=img_id, cls=str(ga.loc[pos, "class"]), box=ab[k], pred_idx=j,
                        stage=int(m.stage[k]), iou=float(m.iou[k]), best_any=float(m.best_iou_any[k]))
            if j >= 0:
                r.pred_cls, r.pred_conf, r.pred_box = str(gp.loc[j, "class"]), float(gp.loc[j, "conf"]), pb[j]
            recs.append(r)
        if m.orphan_preds and len(ab):
            ov = iou_matrix(pb[m.orphan_preds], ab).max(axis=1)
        else:
            ov = np.zeros(len(m.orphan_preds))
        for o, j in enumerate(m.orphan_preds):
            if ov[o] < P["orphan_overlap"]:
                orphans[img_id].append({"cls": str(gp.loc[j, "class"]), "conf": float(gp.loc[j, "conf"]), "box": pb[j]})

    # --- thống kê bền vững của chất lượng định vị trên các cặp "trông sạch" (stage 1, cùng class)
    clean_ious = np.array([r.iou for r in recs if r.stage == 1 and r.pred_cls == r.cls])
    med = float(np.median(clean_ious)) if len(clean_ious) else 0.75
    sigma = max(1.4826 * float(np.median(np.abs(clean_ious - med))), 0.05) if len(clean_ious) else 0.10

    rows: dict[str, dict] = {}
    by_pos = {r.pos: r for r in recs}
    for r in recs:
        aid = str(ann.loc[r.pos, "ann_id"])
        sc = {t: 0.0 for t in TYPES}
        why: dict[str, str] = {}
        act: dict[str, str] = {}
        if r.stage == 0:
            size = float(np.sqrt(max(r.box[2] * r.box[3], 1e-6)))
            sc["unmatched"] = P["w_unmatched"] * (0.7 + 0.3 * min(1.0, size / 32.0))
            why["unmatched"] = "không có bằng chứng từ detector (có thể chỉ là detector bỏ sót)"
            act["unmatched"] = "verify: đối chiếu bằng mắt"
        else:
            conf = r.pred_conf
            z = (med - r.iou) / sigma
            s_shift = float(_sigmoid(z - 3.0))
            if r.stage == 2:
                s_shift = max(s_shift, 0.60)
            sc["box_shift"] = P["w_shift"] * s_shift * (0.6 + 0.4 * conf)
            why["box_shift"] = f"box lệch so với prediction cùng vị trí (IoU {r.iou:.2f}, trung vị sạch {med:.2f})"
            act["box_shift"] = f"edit box:{_fmt_box(r.pred_box)}"
            if r.pred_cls != r.cls:
                base = conf * (0.5 + 0.5 * min(1.0, r.iou / 0.7))
                s_swap = base if r.stage == 1 else 0.8 * base
                sc["class_swap"] = P["w_swap"] * s_swap
                why["class_swap"] = f"nhãn '{r.cls}' nhưng detector thấy '{r.pred_cls}' (conf {conf:.2f}, IoU {r.iou:.2f})"
                act["class_swap"] = f"edit class:{r.pred_cls}"
        rows[aid] = dict(item_id=aid, image_id=r.img, _rec=r, _sc=sc, _why=why, _act=act, partner="")

    # --- duplicate: hai annotation cùng ảnh, cùng class, IoU cao
    by_img_cls: dict[tuple, list[_AnnRec]] = defaultdict(list)
    for r in recs:
        by_img_cls[(r.img, r.cls)].append(r)
    for (img_id, _cls), grp in by_img_cls.items():
        if len(grp) < 2:
            continue
        boxes = np.stack([g.box for g in grp])
        M = iou_matrix(boxes, boxes)
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                if M[a, b] < P["dup_iou"]:
                    continue
                strength = 0.6 + 0.4 * min(1.0, (M[a, b] - P["dup_iou"]) / 0.4)   # P(cặp thật sự là trùng)
                for g, other in ((grp[a], grp[b]), (grp[b], grp[a])):
                    # P(box này là bản sao): khớp detector kém hơn bản kia => khả năng cao hơn; hoà => 0.5.
                    # Mỗi cặp chỉ có MỘT lỗi trong GT nên điểm phải phản ánh kỳ vọng trúng trên mỗi slot review.
                    p_copy = float(np.clip(_sigmoid((other.best_any - g.best_any) / 0.03), 0.05, 0.95))
                    aid, oid = str(ann.loc[g.pos, "ann_id"]), str(ann.loc[other.pos, "ann_id"])
                    sc_dup = P["w_dup"] * strength * p_copy
                    if sc_dup > rows[aid]["_sc"]["duplicate"]:
                        rows[aid]["_sc"]["duplicate"] = sc_dup
                        rows[aid]["_why"]["duplicate"] = f"trùng với {oid} (cùng class '{g.cls}', IoU {M[a, b]:.2f}, P(bản sao)≈{p_copy:.2f})"
                        rows[aid]["_act"]["duplicate"] = f"reject (duplicate của {oid})"
                        rows[aid]["partner"] = oid

    # --- missing: prediction mồ côi, conf cao
    # prediction thật của vật thể có conf cao; nhiễu detector có conf thấp. Điểm tâm = phân vị 10% conf của các
    # prediction đã ghép sạch (stage 1, cùng class, IoU >= 0.5) — ước lượng không cần nhãn; fallback = missing_conf_floor.
    good = np.array([r.pred_conf for r in recs if r.stage == 1 and r.pred_cls == r.cls and r.iou >= 0.5])
    c50 = float(np.percentile(good, 10)) if len(good) >= 10 else float(P["missing_conf_floor"])
    for img_id in images.image_id:
        sc = {t: 0.0 for t in TYPES}
        why, act = {}, {}
        lst = orphans.get(img_id, [])
        if lst:
            best = max(lst, key=lambda o: o["conf"])
            sc["missing"] = P["w_missing"] * float(_sigmoid((best["conf"] - c50) / 0.08))
            why["missing"] = f"detector thấy '{best['cls']}' (conf {best['conf']:.2f}) nhưng không có annotation nào"
            act["missing"] = f"add {best['cls']} box:{_fmt_box(best['box'])}"
        iid = f"img:{img_id}"
        rows[iid] = dict(item_id=iid, image_id=img_id, _rec=None, _sc=sc, _why=why, _act=act, partner="")

    out = []
    for iid, d in rows.items():
        sc = d["_sc"]
        order = sorted(sc, key=lambda t: sc[t], reverse=True)
        top, second = sc[order[0]], sc[order[1]]
        r = d["_rec"]
        tie = 0.001 * (1.0 - r.best_any) if r is not None else 0.0
        score = top + 0.02 * second + tie
        flagged = top >= LABEL_MIN
        etype = order[0] if flagged else "none"
        out.append(dict(
            item_id=iid, score=round(float(score), 6), error_type=etype,
            reason=(f"{etype}: " + d["_why"].get(etype, "")) if flagged else "không có dấu hiệu rõ ràng",
            suggested_action=d["_act"].get(etype, "") if flagged else "",
            image_id=d["image_id"], split=img_split.get(d["image_id"], ""), evidence_path=img_path.get(d["image_id"], ""),
            iou=round(r.iou, 4) if r is not None else np.nan,
            pred_class=r.pred_cls if r is not None else "", pred_conf=r.pred_conf if r is not None else np.nan,
            match_stage=r.stage if r is not None else -1, partner=d["partner"],
            **{f"s_{t}": round(sc[t], 4) for t in TYPES},
        ))
    return pd.DataFrame(out)


def evidence_boxes(scored: pd.DataFrame, ann: pd.DataFrame, preds: pd.DataFrame) -> dict[str, list[dict]]:
    """Box vẽ lên contact sheet: annotation = đỏ, prediction liên quan = xanh lá, prediction mồ côi = xanh dương."""
    ann_i = ann.set_index("ann_id")
    by_img_a = {k: g for k, g in ann.groupby("image_id")}
    by_img_p = {k: g for k, g in preds.groupby("image_id")}
    out: dict[str, list[dict]] = {}
    for r in scored.itertuples(index=False):
        boxes = []
        if str(r.item_id).startswith("img:"):
            for _, a in by_img_a.get(r.image_id, ann.iloc[0:0]).iterrows():
                boxes.append(dict(x=a.x, y=a.y, w=a.w, h=a.h, label=str(a["class"]), color="red"))
            for _, p in by_img_p.get(r.image_id, preds.iloc[0:0]).iterrows():
                boxes.append(dict(x=p.x, y=p.y, w=p.w, h=p.h, label=f"pred {p['class']} {p.conf:.2f}", color="deepskyblue"))
        else:
            a = ann_i.loc[r.item_id]
            boxes.append(dict(x=a.x, y=a.y, w=a.w, h=a.h, label=f"label {a['class']}", color="red"))
            near = by_img_p.get(r.image_id, preds.iloc[0:0])
            if len(near):
                best = iou_matrix(np.array([[a.x, a.y, a.w, a.h]]), near[["x", "y", "w", "h"]].to_numpy(float))[0]
                p = near.iloc[int(np.argmax(best))]
                boxes.append(dict(x=p.x, y=p.y, w=p.w, h=p.h, label=f"pred {p['class']} {p.conf:.2f}", color="limegreen"))
        out[str(r.item_id)] = boxes
    return out
