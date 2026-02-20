from __future__ import annotations

from typing import Dict, List, Tuple


def merge_patterns(
    static_results: List[Tuple[str, Dict, float]],
    dynamic_patterns: List[Tuple[str, Dict, float]],
    boosted_static: Dict[str, float],
    max_dynamic: int = 3,
    dynamic_weight_cap: float = 0.6,
    final_top_k: int = 10,
) -> List[Tuple[str, Dict, float]]:
    """
    Merge static recall patterns with dynamically discovered patterns.

    Args:
        static_results: Original Top-10 from 3-path recall [(pattern_id, pattern_info, score)]
        dynamic_patterns: New patterns from clustering [(pattern_id, pattern_info, score)]
        boosted_static: Score boosts for existing patterns from high-similarity external papers
        max_dynamic: Max number of dynamic patterns to include
        dynamic_weight_cap: Cap dynamic pattern scores at this fraction of top static score
        final_top_k: Final number of patterns to return

    Returns:
        Merged and sorted Top-K patterns [(pattern_id, pattern_info, score)]
    """
    boosted = [
        (pid, info, score + boosted_static.get(pid, 0.0))
        for pid, info, score in static_results
    ]

    top_static_score = max((s for _, _, s in boosted), default=0.0)
    cap = dynamic_weight_cap * top_static_score

    capped_dynamic = [
        (pid, info, min(score, cap))
        for pid, info, score in dynamic_patterns[:max_dynamic]
    ]

    combined = boosted + capped_dynamic
    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:final_top_k]
