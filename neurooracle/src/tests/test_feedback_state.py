from neurooracle.src.feedback_state import FeedbackRecord, FeedbackState
from neurooracle.src.hypothesis_engine import Hypothesis


def make_hypothesis(hid: str, disease: str, region: str, feature: str) -> Hypothesis:
    return Hypothesis(
        id=hid,
        source_id=disease,
        target_id=f"{region}|{feature}",
        confidence_score=0.8,
        evidence_score=0.8,
        novelty_score=0.8,
        testability_score=0.8,
        metadata={
            "candidate_tuple": {
                "disease_id": disease,
                "region_id": region,
                "feature_id": feature,
                "feature_family": feature.split(":")[0],
                "feature_modality": "fMRI",
                "atlas_name": "AAL",
            }
        },
    )


def test_supported_exact_downweights_repeat_but_boosts_similar():
    supported = make_hypothesis("h1", "D:AD", "ROI:1", "fc:degree")
    similar = make_hypothesis("h2", "D:AD", "ROI:1", "fc:strength")
    record = FeedbackState.record_from_hypothesis(supported, "supported", reason="validated")
    state = FeedbackState([FeedbackRecord.from_dict(record)])

    exact_adj = state.score(supported)
    similar_adj = state.score(similar)

    assert exact_adj.exact_supported
    assert exact_adj.multiplier < 1.0
    assert similar_adj.supported_similarity > 0
    assert similar_adj.multiplier > 1.0


def test_contradicted_and_execution_failed_penalize_similar_hypotheses():
    contradicted = make_hypothesis("h1", "D:AD", "ROI:1", "fc:degree")
    failed = make_hypothesis("h2", "D:MDD", "ROI:2", "alff:mean")
    similar_contra = make_hypothesis("h3", "D:AD", "ROI:1", "fc:strength")
    similar_failed = make_hypothesis("h4", "D:MDD", "ROI:2", "alff:sd")
    state = FeedbackState([
        FeedbackRecord.from_dict(
            FeedbackState.record_from_hypothesis(contradicted, "contradicted", reason="not replicated")
        ),
        FeedbackRecord.from_dict(
            FeedbackState.record_from_hypothesis(failed, "execution_failed", reason="missing covariates")
        ),
    ])

    contra_adj = state.score(similar_contra)
    failed_adj = state.score(similar_failed)

    assert contra_adj.contradicted_similarity > 0
    assert contra_adj.multiplier < 1.0
    assert failed_adj.execution_failed_similarity > 0
    assert failed_adj.multiplier < 1.0
