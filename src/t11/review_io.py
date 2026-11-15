"""Áp dụng quyết định của reviewer lên annotation. Raw KHÔNG bị ghi đè: sinh file mới + diff + audit log.

Quy ước `override_value` (chuỗi, 1 ô trong CSV của contact sheet):
    class:<tên>               đổi class của annotation            (decision = edit)
    box:x,y,w,h               đổi box của annotation               (decision = edit)
    class:<tên>;box:x,y,w,h   đổi cả hai
    add:<tên>:x,y,w,h         thêm box còn thiếu cho item `img:<id>` (decision = edit)
Quyết định:
    approve  nhãn đúng  -> không đổi; tính là cảnh báo sai (false positive) của hệ thống
    reject   annotation sai hẳn / trùng -> XOÁ annotation (với item `img:` thì không đổi gì)
    edit     sửa theo override_value (bắt buộc hợp lệ, sai định dạng -> ValueError, không im lặng bỏ qua)
    escalate/skip  không đổi; không tính vào tỉ lệ lỗi
"""
from __future__ import annotations

import pandas as pd

from ..common.review import AuditLog

CHANGE_COLUMNS = ["item_id", "change", "field", "before", "after", "reviewer", "review_note", "reviewed_at"]


def parse_override(value: str) -> dict:
    v = (value or "").strip()
    if not v:
        raise ValueError("edit cần override_value (vd: class:car | box:x,y,w,h | add:car:x,y,w,h)")
    if v.startswith("add:"):
        parts = v.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"override không hợp lệ: {value!r}")
        return {"add": (parts[1].strip(), _parse_box(parts[2]))}
    out: dict = {}
    for tok in v.split(";"):
        tok = tok.strip()
        if tok.startswith("class:"):
            out["class"] = tok[6:].strip()
        elif tok.startswith("box:"):
            out["box"] = _parse_box(tok[4:])
        elif tok:
            raise ValueError(f"override không hợp lệ: {value!r}")
    if not out:
        raise ValueError(f"override không hợp lệ: {value!r}")
    return out


def _parse_box(s: str) -> tuple[float, float, float, float]:
    try:
        x, y, w, h = (float(t) for t in s.split(","))
    except ValueError as e:
        raise ValueError(f"box phải có dạng x,y,w,h: {s!r}") from e
    if w <= 0 or h <= 0:
        raise ValueError(f"box có w/h không dương: {s!r}")
    return x, y, w, h


def apply_review(ann: pd.DataFrame, reviewed: pd.DataFrame, audit: AuditLog | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Trả về (annotation đã sửa, bảng thay đổi, thống kê review)."""
    out = ann.copy()
    out["ann_id"] = out["ann_id"].astype(str)
    changes: list[dict] = []
    stats = {"reviewed": 0, "confirmed_error": 0, "false_positive": 0, "escalated": 0, "skipped": 0}
    n_add = 0

    def log(r, action, before, after):
        if audit:
            audit.log(actor=r.reviewer, action=action, item_id=r.item_id, before=before, after=after, reason=r.review_note)

    for r in reviewed.itertuples(index=False):
        dec, iid = r.reviewer_decision, str(r.item_id)
        is_img = iid.startswith("img:")
        row = {"item_id": iid, "reviewer": r.reviewer, "review_note": r.review_note, "reviewed_at": r.reviewed_at}
        if dec == "approve":
            stats["reviewed"] += 1
            stats["false_positive"] += 1
            log(r, "review:approve", "flagged", "label_ok")
        elif dec == "escalate":
            stats["escalated"] += 1
            log(r, "review:escalate", "flagged", "escalated")
        elif dec == "skip":
            stats["skipped"] += 1
            log(r, "review:skip", "flagged", "skipped")
        elif dec == "reject":
            stats["reviewed"] += 1
            if is_img:
                stats["false_positive"] += 1
                log(r, "review:reject_img", "flagged", "nothing_missing")
                continue
            mask = out["ann_id"] == iid
            if not mask.any():
                raise ValueError(f"reject: không tìm thấy annotation {iid}")
            before = out.loc[mask].iloc[0].to_dict()
            out = out.loc[~mask].reset_index(drop=True)
            stats["confirmed_error"] += 1
            changes.append({**row, "change": "delete", "field": "annotation", "before": str(before), "after": ""})
            log(r, "override:reject", before, "deleted")
        elif dec == "edit":
            stats["reviewed"] += 1
            ov = parse_override(r.override_value)
            if is_img:
                if "add" not in ov:
                    raise ValueError(f"{iid}: item img: chỉ nhận override dạng add:<class>:x,y,w,h")
                cls, (x, y, w, h) = ov["add"]
                n_add += 1
                new_id = f"ann_add_{n_add:03d}"
                out = pd.concat([out, pd.DataFrame([{"ann_id": new_id, "image_id": iid[4:], "class": cls, "x": x, "y": y, "w": w, "h": h}])], ignore_index=True)
                stats["confirmed_error"] += 1
                changes.append({**row, "change": "add", "field": "annotation", "before": "", "after": f"{new_id}:{cls}:{x},{y},{w},{h}"})
                log(r, "override:add", "", new_id)
            else:
                if "add" in ov:
                    raise ValueError(f"{iid}: dùng class:/box: để sửa annotation có sẵn")
                mask = out["ann_id"] == iid
                if not mask.any():
                    raise ValueError(f"edit: không tìm thấy annotation {iid}")
                if "class" in ov:
                    before = out.loc[mask, "class"].iloc[0]
                    out.loc[mask, "class"] = ov["class"]
                    changes.append({**row, "change": "edit", "field": "class", "before": before, "after": ov["class"]})
                    log(r, "override:edit_class", before, ov["class"])
                if "box" in ov:
                    before = ",".join(f"{v:g}" for v in out.loc[mask, ["x", "y", "w", "h"]].iloc[0])
                    out.loc[mask, ["x", "y", "w", "h"]] = list(ov["box"])
                    changes.append({**row, "change": "edit", "field": "box", "before": before, "after": ",".join(f"{v:g}" for v in ov["box"])})
                    log(r, "override:edit_box", before, ov["box"])
                stats["confirmed_error"] += 1
    return out, pd.DataFrame(changes, columns=CHANGE_COLUMNS), stats
