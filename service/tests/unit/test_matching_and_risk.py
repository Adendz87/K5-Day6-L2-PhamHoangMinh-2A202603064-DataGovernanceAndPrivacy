"""Unit tests for Matching, Error Analysis, and Risk Scorer."""
from service.adapters.yolo_adapter import YOLOPrediction
from service.analysis.error_analysis import ErrorAnalyzer
from service.analysis.matching_adapter import MatchingAdapter
from service.analysis.risk_scorer import RiskScorer
from service.core.annotation_normalizer import NormalizedAnnotation


def test_matching_and_error_classification():
    matching = MatchingAdapter(match_iou=0.3, shift_gate=2.0)
    analyzer = ErrorAnalyzer()

    # Create 1 clean car annotation and 1 class-swap annotation
    anns = [
        NormalizedAnnotation("a1", "img1", "car", 0, 10.0, 10.0, 50.0, 30.0, 320, 180),
        NormalizedAnnotation("a2", "img1", "pedestrian", 1, 100.0, 50.0, 20.0, 40.0, 320, 180),
    ]

    # Predictions: car at a1, but truck at a2 (class swap!)
    preds = [
        YOLOPrediction("car", 0, 0.9, 10.0, 10.0, 50.0, 30.0),
        YOLOPrediction("truck", 2, 0.85, 100.0, 50.0, 20.0, 40.0),
    ]

    match_res = matching.match_image_data("img1", anns, preds)
    ann_evs, miss_evs = analyzer.analyze_image(match_res)

    assert len(ann_evs) == 2
    # a2 should be flagged as class_swap
    ev_a2 = next(e for e in ann_evs if e.ann_id == "a2")
    assert ev_a2.possible_error_type == "class_swap"
    assert ev_a2.risk_score > 0.5


def test_risk_scorer_hierarchy():
    scorer = RiskScorer()
    from service.analysis.error_analysis import ScoredAnnotationEvidence

    ann_evs = [
        ScoredAnnotationEvidence(
            ann_id="a1",
            image_id="img1",
            risk_score=0.1,
            risk_level="RISK_LOW",
            possible_error_type="none",
            reason_code="NONE",
            suggested_action="",
            review_required=False,
            evidence_dict={},
        ),
        ScoredAnnotationEvidence(
            ann_id="a2",
            image_id="img1",
            risk_score=0.85,
            risk_level="RISK_CRITICAL",
            possible_error_type="class_swap",
            reason_code="CLASS_SWAP",
            suggested_action="edit class:car",
            review_required=True,
            evidence_dict={},
        ),
    ]

    img_summary = scorer.compute_image_risk("img1", ann_evs, [])
    assert img_summary.image_score > 0.5
    assert img_summary.risk_level in ("RISK_HIGH", "RISK_CRITICAL")
    assert img_summary.review_required is True
