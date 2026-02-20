from __future__ import annotations

from typing import List, Set

import numpy as np

from idea2paper.agentic_search.paper_sources.base import ExternalPaper


def dedup_by_id(
    papers: List[ExternalPaper], existing_paper_ids: Set[str]
) -> List[ExternalPaper]:
    """Layer 1: Remove papers whose paper_id or DOI matches existing KG papers,
    and deduplicate among the batch itself (keep first occurrence)."""
    seen: Set[str] = set(existing_paper_ids)
    result: List[ExternalPaper] = []
    for p in papers:
        ids = {p.paper_id}
        if p.doi:
            ids.add(p.doi)
        if ids & seen:
            continue
        seen.update(ids)
        result.append(p)
    return result


def dedup_by_semantic(
    papers: List[ExternalPaper],
    embeddings: np.ndarray,
    threshold: float = 0.95,
) -> List[ExternalPaper]:
    """Layer 2: For paper pairs with cosine similarity > threshold, keep the newer one."""
    n = len(papers)
    if n != embeddings.shape[0]:
        raise ValueError(
            f"papers length ({n}) != embeddings rows ({embeddings.shape[0]})"
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    removed: set[int] = set()
    for i in range(n):
        if i in removed:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue
            if sim_matrix[i, j] > threshold:
                # Keep the newer paper; on tie keep the earlier index
                drop = i if papers[j].year > papers[i].year else j
                removed.add(drop)

    return [p for idx, p in enumerate(papers) if idx not in removed]


def filter_by_year(papers: List[ExternalPaper], year_from: int) -> List[ExternalPaper]:
    """Layer 3: Strict year filtering."""
    return [p for p in papers if p.year >= year_from]


def run_dedup_pipeline(
    papers: List[ExternalPaper],
    existing_paper_ids: Set[str],
    embeddings: np.ndarray | None,
    year_from: int,
    semantic_threshold: float = 0.95,
) -> List[ExternalPaper]:
    """Run all three dedup layers in sequence."""
    result = dedup_by_id(papers, existing_paper_ids)
    if embeddings is not None and len(result) > 1:
        result = dedup_by_semantic(result, embeddings, semantic_threshold)
    result = filter_by_year(result, year_from)
    return result
