"""Test của nhóm cho T11 (success + failure/edge). Không đọc gt/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.review import AuditLog, load_review_decisions
from src.main import main as run_main
from src.t11.geometry import iou_matrix
from src.t11.matching import match_image
from src.t11.review_io import apply_review, parse_override
from src.t11.scoring import score_items

IMGS = pd.DataFrame({"image_id": ["a"], "path": ["images/a.jpg"], "width": [320], "height": [180], "split": ["heldout"]})


def _world(ann_rows, pred_rows):
    ann = pd.DataFrame(ann_rows, columns=["ann_id", "image_id", "class", "x", "y", "w", "h"])
    preds = pd.DataFrame(pred_rows, columns=["image_id", "class", "x", "y", "w", "h", "conf"])
    return ann, preds


# 3 đối tượng sạch: annotation khớp prediction (đủ để ước lượng phân phối "sạch")
CLEAN_A = [("c1", "a", "car", 20, 100, 50, 30), ("c2", "a", "car", 120, 90, 50, 30), ("c3", "a", "pedestrian", 220, 80, 15, 40)]
CLEAN_P = [("a", "car", 21, 101, 50, 30, 0.9), ("a", "car", 119, 91, 50, 30, 0.85), ("a", "pedestrian", 221, 81, 15, 40, 0.8)]


def test_iou_identity_and_disjoint():
    b = np.array([[0, 0, 10, 10.0]])
    assert iou_matrix(b, b)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(b, np.array([[50, 50, 10, 10.0]]))[0, 0] == 0.0


def test_stage2_matches_far_shifted_box_so_no_phantom_orphan():
    # box bị dịch ~1 chiều rộng: IoU = 0 nhưng cùng class và tâm gần => phải ghép (stage 2), không sinh "missing" giả
    m = match_image([[100, 100, 40, 30]], ["car"], [[145, 100, 40, 30]], ["car"], 0.3, 2.0)
    assert m.stage[0] == 2 and m.orphan_preds == []


def test_class_swap_is_top_and_suggests_detector_class():
    ann, preds = _world(CLEAN_A[:2] + [("x", "a", "motorbike", 220, 80, 15, 40)], CLEAN_P)   # nhãn sai: pedestrian -> motorbike
    s = score_items(ann, preds, IMGS).sort_values("score", ascending=False)
    top = s.iloc[0]
    assert top.item_id == "x" and top.error_type == "class_swap" and top.suggested_action == "edit class:pedestrian"


def test_box_shift_flagged_over_clean():
    ann, preds = _world([CLEAN_A[0], CLEAN_A[1], ("s", "a", "pedestrian", 262, 80, 15, 40)], CLEAN_P)
    s = score_items(ann, preds, IMGS).set_index("item_id")
    assert s.loc["s", "score"] > 0.5 > s.loc["c1", "score"]
    assert s.loc["s", "error_type"] == "box_shift" and s.loc["s", "suggested_action"].startswith("edit box:")


def test_duplicate_flags_pair_and_missing_uses_confidence():
    ann, preds = _world(CLEAN_A + [("dup", "a", "car", 22, 102, 50, 30)], CLEAN_P)
    s = score_items(ann, preds, IMGS).set_index("item_id")
    assert s.loc["dup", "error_type"] == "duplicate" and s.loc["dup", "partner"] == "c1"
    # missing: prediction mồ côi conf cao => ảnh bị nghi; conf thấp => bỏ qua
    hi = _world(CLEAN_A, CLEAN_P + [("a", "sign", 280, 20, 15, 15, 0.97)])
    lo = _world(CLEAN_A, CLEAN_P + [("a", "sign", 280, 20, 15, 15, 0.15)])
    s_hi = score_items(*hi, IMGS).set_index("item_id").loc["img:a"]
    s_lo = score_items(*lo, IMGS).set_index("item_id").loc["img:a"]
    assert s_hi.error_type == "missing" and s_hi.score > 0.5 and s_lo.score < 0.1


def test_scores_do_not_depend_on_ids_or_row_order():
    ann, preds = _world(CLEAN_A + [("dup", "a", "car", 22, 102, 50, 30)], CLEAN_P)
    base = score_items(ann, preds, IMGS).set_index("item_id").score
    shuffled = ann.sample(frac=1.0, random_state=0).reset_index(drop=True)
    renamed = shuffled.assign(ann_id=[f"zz{i}" for i in range(len(shuffled))])
    s2 = score_items(renamed, preds, IMGS)
    # cùng hình học => cùng bộ điểm (so sánh đa tập hợp, vì ID đã đổi)
    assert np.allclose(np.sort(base.to_numpy()), np.sort(s2.score.to_numpy()), atol=1e-4)


def test_empty_image_edge_cases():
    ann, preds = _world([], [("a", "car", 10, 10, 30, 20, 0.9)])       # ảnh không có annotation nào
    s = score_items(ann, preds, IMGS)
    assert list(s.item_id) == ["img:a"] and s.score.iloc[0] > 0.5
    ann, preds = _world([("o", "a", "car", 10, 10, 30, 20)], [])         # không có prediction nào
    s = score_items(ann, preds, IMGS).set_index("item_id")
    assert s.loc["o", "error_type"] in ("unmatched", "none") and s.loc["o", "score"] < 0.5


# ------------------------------------------------------------------ review
def _reviewed(rows):
    cols = ["rank", "item_id", "score", "reason", "suggested_action", "slice", "evidence_path", "reviewer", "reviewer_decision", "override_value", "review_note", "reviewed_at"]
    return pd.DataFrame([{**{c: "" for c in cols}, **r} for r in rows])


def test_apply_review_edits_deletes_adds_and_keeps_raw(tmp_path):
    ann, _ = _world(CLEAN_A + [("dup", "a", "car", 22, 102, 50, 30)], CLEAN_P)
    raw = ann.copy()
    rv = _reviewed([
        {"item_id": "c3", "reviewer": "minh", "reviewer_decision": "edit", "override_value": "class:sign", "review_note": "sai nhãn"},
        {"item_id": "dup", "reviewer": "minh", "reviewer_decision": "reject", "review_note": "trùng"},
        {"item_id": "img:a", "reviewer": "minh", "reviewer_decision": "edit", "override_value": "add:sign:280,20,15,15"},
        {"item_id": "c1", "reviewer": "minh", "reviewer_decision": "approve"},
    ])
    audit = AuditLog(tmp_path / "audit.jsonl")
    out, changes, stats = apply_review(ann, rv, audit)
    pd.testing.assert_frame_equal(ann, raw)                                   # raw không bị sửa tại chỗ
    assert "dup" not in set(out.ann_id) and out.loc[out.ann_id == "c3", "class"].iloc[0] == "sign"
    assert (out.ann_id == "ann_add_001").sum() == 1
    assert stats == {"reviewed": 4, "confirmed_error": 3, "false_positive": 1, "escalated": 0, "skipped": 0}
    assert set(changes.change) == {"edit", "delete", "add"} and len(audit.read()) == 4


def test_apply_review_fails_loudly_on_bad_input(tmp_path):
    ann, _ = _world(CLEAN_A, CLEAN_P)
    with pytest.raises(ValueError):
        parse_override("")                                                    # edit không có override
    with pytest.raises(ValueError):
        parse_override("box:1,2,3")                                           # thiếu số
    with pytest.raises(ValueError):
        apply_review(ann, _reviewed([{"item_id": "nope", "reviewer": "m", "reviewer_decision": "reject"}]))
    p = tmp_path / "r.csv"
    _reviewed([{"item_id": "c1", "reviewer_decision": "approve"}]).to_csv(p, index=False)   # thiếu reviewer
    with pytest.raises(ValueError):
        load_review_decisions(p)


# ------------------------------------------------------------------ end-to-end (fixture thật, offline)
def _run(out: Path):
    return run_main(["--config", "configs/lab.yaml", "--track", "t11", "--output", str(out), "--run-name", "poc", "--offline", "--seed", "42"])


def test_end_to_end_then_review_loop(tmp_path):
    out = tmp_path / "poc"
    assert _run(out) == 0
    rk = pd.read_csv(out / "ranking.csv")
    assert {"item_id", "score"} <= set(rk.columns) and rk.item_id.is_unique and len(rk) == 260
    assert json.loads((out / "decision.json").read_text(encoding="utf-8"))["reason_code"] == "REVIEW_PENDING"
    assert (out / "contact_sheet.html").exists()
    # tính xác định: chạy lại ra đúng cùng thứ hạng
    out2 = tmp_path / "poc2"
    assert _run(out2) == 0
    assert list(pd.read_csv(out2 / "ranking.csv").item_id) == list(rk.item_id)

    # vòng review: reviewer làm theo gợi ý cho các mục class/duplicate
    q = pd.read_csv(out / "review_queue.csv", dtype=str).fillna("")
    q["reviewer"] = "tester"
    cls = q.suggested_action.str.startswith("edit class:")
    dup = q.suggested_action.str.startswith("reject")
    q.loc[cls, "reviewer_decision"] = "edit"
    q.loc[cls, "override_value"] = q.loc[cls, "suggested_action"].str.replace("edit ", "", regex=False)
    q.loc[dup, "reviewer_decision"] = "reject"
    q.loc[~(cls | dup), "reviewer_decision"] = "skip"
    q.to_csv(out / "review_queue_reviewed.csv", index=False)
    assert _run(out) == 0
    ch = pd.read_csv(out / "changed_labels.csv")
    assert len(ch) >= 5 and (out / "annotations_corrected.csv").exists() and (out / "audit_log.jsonl").exists()
    assert json.loads((out / "decision.json").read_text(encoding="utf-8"))["reason_code"] != "REVIEW_PENDING"
