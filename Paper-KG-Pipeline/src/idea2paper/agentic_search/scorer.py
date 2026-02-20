from __future__ import annotations


def compute_recall_quality(pattern_scores: list[float]) -> float:
    """
    Compute recall quality score from Top-10 pattern final scores.

    Weighted blend of top-3 importance and overall average:
        avg_score = (score₁×0.5 + score₂×0.3 + score₃×0.2) × 0.7
                  + (Σ score₁₋₁₀ / 10) × 0.3
    """
    if not pattern_scores:
        return 0.0

    scores = list(pattern_scores)
    while len(scores) < 10:
        scores.append(0.0)

    top3_weighted = scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2
    overall_avg = sum(scores[:10]) / 10

    return top3_weighted * 0.7 + overall_avg * 0.3


def determine_search_params(avg_score: float) -> dict:
    """
    Map recall quality to search intensity parameters.

    Returns dict with keys: search_count, year_from, year_quotas.
    year_quotas maps year -> max papers; unused quota from excluded years
    is redistributed to the most recent included year.
    """
    base_quotas = {2025: 5, 2024: 3, 2023: 2}

    if avg_score > 0.7:
        year_from = 2025
        search_count = 5
    elif avg_score >= 0.4:
        year_from = 2024
        search_count = 10
    else:
        year_from = 2023
        search_count = 20

    year_quotas: dict[int, int] = {}
    redistributed = 0
    for year, quota in sorted(base_quotas.items()):
        if year < year_from:
            redistributed += quota
        else:
            year_quotas[year] = quota

    if year_quotas:
        earliest_included = min(year_quotas)
        year_quotas[earliest_included] += redistributed

    return {
        "search_count": search_count,
        "year_from": year_from,
        "year_quotas": year_quotas,
    }
