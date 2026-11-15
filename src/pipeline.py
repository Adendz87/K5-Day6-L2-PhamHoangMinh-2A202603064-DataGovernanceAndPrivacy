"""T11 · Post-label Error Review Prioritizer — PoC pipeline của nhóm.

5 bước bắt buộc:
    01 load     đọc annotations / model_predictions / images + kiểm hash provenance
    02 process  ghép annotation<->prediction rồi chấm suspicion score (xem src/t11/)
    03 rank     top-B theo score -> review_queue.csv + contact_sheet.html (box label đỏ / prediction xanh)
    04 review   nếu có out/review_queue_reviewed.csv: áp quyết định -> changed_labels.csv + annotations_corrected.csv
                + audit_log.jsonl (raw annotations.csv không bị đụng)
    05 decide   GO / HOLD / REWORK với reason_code + owner (ngưỡng đóng băng trong configs/lab.yaml)

Không đọc fixtures/*/gt/, không mạng, mọi ngẫu nhiên đi qua ctx.rng (PoC này hoàn toàn xác định).
"""
from __future__ import annotations

import math

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .t11.review_io import apply_review
from .t11.scoring import evidence_boxes, score_items


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    d = ctx.input_dir
    data = {
        "manifest": pd.read_csv(manifest) if manifest.exists() else None,
        "ann": pd.read_csv(d / "annotations.csv"),
        "preds": pd.read_csv(d / "model_predictions.csv"),
        "images": pd.read_csv(d / "images.csv"),
    }
    need = {"ann": ["ann_id", "image_id", "class", "x", "y", "w", "h"], "preds": ["image_id", "class", "x", "y", "w", "h", "conf"]}
    for k, cols in need.items():
        miss = [c for c in cols if c not in data[k].columns]
        if miss:
            raise ValueError(f"{k}: thiếu cột {miss}")
    return data


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    ctx.stats["n_annotations"] = len(data["ann"])
    ctx.stats["n_images"] = len(data["images"])
    return score_items(data["ann"], data["preds"], data["images"], ctx.config.get("params"))


# ---------------------------------------------------------------- 03 rank
def _budget(ctx: RunContext, n_ann: int) -> int:
    b = ctx.param("review_budget", "auto")
    return max(10, math.ceil(0.10 * n_ann)) if str(b).lower() == "auto" else int(b)


def rank(ctx: RunContext, scored: pd.DataFrame, data: dict | None = None) -> pd.DataFrame:
    budget = _budget(ctx, ctx.stats.get("n_annotations", len(scored)))
    ctx.stats["review_budget"] = budget
    q = make_review_queue(scored, id_col="item_id", score_col="score", budget=budget, reason_col="reason",
                          action_col="suggested_action", slice_col="error_type", evidence_col="evidence_path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    if data is not None:
        boxes = evidence_boxes(scored[scored.item_id.isin(q.item_id)], data["ann"], data["preds"])
        render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir, boxes_by_item=boxes,
                             title=f"T11 review queue (B={budget}) — đỏ = nhãn hiện tại, xanh lá/xanh dương = detector",
                             fields=("rank", "item_id", "score", "slice", "reason", "suggested_action"))
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame, data: dict | None = None) -> pd.DataFrame:
    """Điểm ranking KHÔNG bị sửa bởi review (ranking.csv đo chất lượng ưu tiên); review chỉ sửa nhãn."""
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    final = scored.copy()
    final["review_decision"] = ""
    if reviewed_path.exists() and data is not None:
        reviewed = load_review_decisions(reviewed_path)
        corrected, changes, stats = apply_review(data["ann"], reviewed, ctx.audit)
        corrected.to_csv(ctx.out("annotations_corrected.csv"), index=False)
        changes.to_csv(ctx.out("changed_labels.csv"), index=False)
        data["ann"].to_csv(ctx.out("annotations_raw.csv"), index=False)   # raw giữ nguyên để so diff
        ctx.stats.update({f"review_{k}": v for k, v in stats.items()})
        ctx.stats["n_reviewed"] = len(reviewed)
        final["review_decision"] = final.item_id.map(dict(zip(reviewed.item_id.astype(str), reviewed.reviewer_decision))).fillna("")
    return final


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    owner = ctx.config.get("owner", "TODO-owner")
    P = ctx.config.get("params") or {}
    n_rev = int(ctx.stats.get("review_reviewed", 0))
    ev = [str(ctx.out("ranking.csv")), str(ctx.out("review_queue.csv"))]
    if n_rev == 0:
        return Decision("HOLD", "REVIEW_PENDING", "Chưa có quyết định của reviewer cho review queue; không thể phát hành nhãn.",
                        owner, scope="t11", evidence=ev, metrics={"review_budget": ctx.stats.get("review_budget")})
    conf = int(ctx.stats.get("review_confirmed_error", 0))
    rate = conf / n_rev
    pending = final[final.review_decision == ""]
    resid = float(pending.score.max()) if len(pending) else 0.0
    m = {"reviewed": n_rev, "confirmed_errors": conf, "error_rate_in_queue": round(rate, 4), "max_unreviewed_score": round(resid, 4)}
    ev += [str(ctx.out("changed_labels.csv")), str(ctx.out("audit_log.jsonl"))]
    if rate >= float(P.get("rework_error_rate", 0.30)):
        return Decision("REWORK", "HIGH_ERROR_DENSITY", f"{conf}/{n_rev} mục review là lỗi thật: mật độ lỗi cao, cần rà soát rộng hơn budget hiện tại.", owner, "t11", ev, m)
    if rate < float(P.get("go_error_rate", 0.10)) and resid < float(P.get("go_residual_score", 0.50)):
        return Decision("GO", "LOW_ERROR_DENSITY", "Mật độ lỗi thấp và không còn mục điểm cao chưa review.", owner, "t11", ev, m)
    return Decision("HOLD", "RESIDUAL_RISK", "Còn mục nghi ngờ điểm cao chưa review hoặc mật độ lỗi ở mức trung bình.", owner, "t11", ev, m)


def run(ctx: RunContext) -> None:
    if ctx.track != "t11":
        raise NotImplementedError(
            f"Pipeline hiện chỉ hỗ trợ T11, chưa triển khai track {ctx.track}."
        )
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored, data)
    final = review(ctx, scored, queue, data)
    final = final.sort_values("score", ascending=False, kind="mergesort")
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
