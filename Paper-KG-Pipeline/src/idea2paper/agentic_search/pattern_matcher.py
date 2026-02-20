from __future__ import annotations

from typing import Any

import numpy as np

from idea2paper.infra.llm import call_llm, parse_json_from_llm
from idea2paper.infra.embeddings import get_embedding, get_embeddings_batch
from idea2paper.infra.run_context import get_logger

MATCH_THRESHOLD = 0.85
GRAY_ZONE_LOWER = 0.75
TOP_K_SCREEN = 20

EXTRACTION_PROMPT = """\
You are a research paper analysis assistant.
Given the following paper title and abstract, extract these fields as JSON:

Title: {title}
Abstract: {abstract}

Return ONLY a JSON object with these keys:
- "idea_summary": a one-sentence summary of the paper's core idea
- "problem_definition": what problem does this paper address
- "solution_pattern": what approach or method does it propose
- "domain": the primary research domain (e.g. "NLP", "Computer Vision")
- "sub_domains": a list of specific sub-domains (e.g. ["text classification", "transformer"])

Output JSON only, no explanation."""


def extract_paper_fields(title: str, abstract: str) -> dict | None:
    """Use LLM to extract structured fields from a paper's title + abstract."""
    prompt = EXTRACTION_PROMPT.format(title=title, abstract=abstract)
    response = call_llm(prompt, temperature=0.0, max_tokens=1024)
    if not response:
        return None
    parsed = parse_json_from_llm(response)
    if parsed is None:
        return None
    required = {"idea_summary", "problem_definition", "solution_pattern", "domain", "sub_domains"}
    if not required.issubset(parsed.keys()):
        return None
    return parsed


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.asarray(vec1, dtype=np.float64)
    b = np.asarray(vec2, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    a_lower = {s.lower().strip() for s in set_a}
    b_lower = {s.lower().strip() for s in set_b}
    intersection = a_lower & b_lower
    union = a_lower | b_lower
    return len(intersection) / len(union) if union else 0.0


def compute_pattern_similarity(
    paper_fields: dict,
    paper_abstract_emb: list[float],
    pattern: dict,
    pattern_summary_emb: list[float],
    pattern_problem_emb: list[float] | None,
    pattern_solution_emb: list[float] | None,
) -> float:
    """Compute multi-feature weighted similarity between a paper and a pattern."""
    summary_sim = _cosine_similarity(paper_abstract_emb, pattern_summary_emb)

    problem_sim = 0.0
    if pattern_problem_emb is not None:
        problem_emb = get_embedding(paper_fields.get("problem_definition", ""))
        if problem_emb is not None:
            problem_sim = _cosine_similarity(problem_emb, pattern_problem_emb)

    solution_sim = 0.0
    if pattern_solution_emb is not None:
        solution_emb = get_embedding(paper_fields.get("solution_pattern", ""))
        if solution_emb is not None:
            solution_sim = _cosine_similarity(solution_emb, pattern_solution_emb)

    paper_subs = set(paper_fields.get("sub_domains", []))
    pattern_subs = set(pattern.get("sub_domains", []))
    domain_sim = _jaccard_similarity(paper_subs, pattern_subs)

    return summary_sim * 0.4 + problem_sim * 0.25 + solution_sim * 0.25 + domain_sim * 0.1


def _get_pattern_text(pattern: dict, field: str, fallback_key: str = "") -> str:
    enhanced = pattern.get("llm_enhanced_summary", {})
    text = enhanced.get(field, "")
    if not text and fallback_key:
        text = pattern.get(fallback_key, "")
    return text


def _precompute_pattern_embeddings(
    patterns: list[dict], logger: Any = None,
) -> dict[str, dict[str, list[float] | None]]:
    """Batch-compute embeddings for all patterns: summary, base_problem, solution_pattern."""
    summary_texts = []
    problem_texts = []
    solution_texts = []

    for p in patterns:
        summary_texts.append(
            _get_pattern_text(p, "representative_ideas", "name") or p.get("name", "")
        )
        problem_texts.append(_get_pattern_text(p, "base_problem"))
        solution_texts.append(_get_pattern_text(p, "solution_pattern"))

    summary_embs = get_embeddings_batch(summary_texts, logger=logger) or [None] * len(patterns)
    problem_embs = get_embeddings_batch(
        [t if t else "N/A" for t in problem_texts], logger=logger
    )
    solution_embs = get_embeddings_batch(
        [t if t else "N/A" for t in solution_texts], logger=logger
    )

    result: dict[str, dict[str, list[float] | None]] = {}
    for i, p in enumerate(patterns):
        pid = p.get("pattern_id", str(i))
        result[pid] = {
            "summary": summary_embs[i] if summary_embs else None,
            "problem": (
                problem_embs[i] if problem_embs and problem_texts[i] else None
            ),
            "solution": (
                solution_embs[i] if solution_embs and solution_texts[i] else None
            ),
        }
    return result


def match_papers_to_patterns(
    papers: list[dict],
    patterns: list[dict],
    logger: Any = None,
) -> dict:
    """
    Match external papers against existing KG patterns.

    For each paper:
    1. Get abstract embedding
    2. Quick screen via cosine against all pattern summary embeddings -> Top-20
    3. Precise multi-feature similarity on Top-20
    4. Route by threshold:
       >= 0.85 -> boosted_static
       0.75~0.85 -> gray zone (skipped)
       < 0.75 -> pending_cluster
    """
    if logger is None:
        logger = get_logger()

    boosted_static: dict[str, float] = {}
    pending_cluster: list[dict] = []
    match_details: list[dict] = []

    if not patterns:
        return {
            "boosted_static": boosted_static,
            "pending_cluster": list(papers),
            "match_details": match_details,
        }

    pattern_embs = _precompute_pattern_embeddings(patterns, logger)
    pattern_ids = [p.get("pattern_id", str(i)) for i, p in enumerate(patterns)]

    # Build summary embedding matrix for quick screening
    summary_vecs = []
    for pid in pattern_ids:
        emb = pattern_embs[pid]["summary"]
        summary_vecs.append(emb if emb is not None else [0.0])
    max_dim = max(len(v) for v in summary_vecs)
    summary_matrix = np.zeros((len(patterns), max_dim), dtype=np.float64)
    for i, v in enumerate(summary_vecs):
        if len(v) == max_dim:
            summary_matrix[i] = v

    norms = np.linalg.norm(summary_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed_patterns = summary_matrix / norms

    for paper in papers:
        paper_id = paper.get("paper_id", "")
        fields = paper.get("fields")
        if fields is None:
            pending_cluster.append(paper)
            match_details.append({
                "paper_id": paper_id, "best_pattern": None, "similarity": 0.0,
            })
            continue

        abstract_emb = get_embedding(paper.get("abstract", ""), logger=logger)
        if abstract_emb is None:
            pending_cluster.append(paper)
            match_details.append({
                "paper_id": paper_id, "best_pattern": None, "similarity": 0.0,
            })
            continue

        # Quick screen: cosine against all pattern summaries
        paper_vec = np.asarray(abstract_emb, dtype=np.float64)
        paper_norm = np.linalg.norm(paper_vec)
        if paper_norm == 0:
            pending_cluster.append(paper)
            match_details.append({
                "paper_id": paper_id, "best_pattern": None, "similarity": 0.0,
            })
            continue

        quick_scores = normed_patterns @ (paper_vec / paper_norm)
        top_indices = np.argsort(quick_scores)[::-1][:TOP_K_SCREEN]

        # Precise multi-feature similarity on top candidates
        best_sim = 0.0
        best_pid: str | None = None
        for idx in top_indices:
            pid = pattern_ids[idx]
            embs = pattern_embs[pid]
            sim = compute_pattern_similarity(
                fields,
                abstract_emb,
                patterns[idx],
                embs["summary"] or [],
                embs["problem"],
                embs["solution"],
            )
            if sim > best_sim:
                best_sim = sim
                best_pid = pid

        match_details.append({
            "paper_id": paper_id,
            "best_pattern": best_pid,
            "similarity": round(best_sim, 4),
        })

        if best_sim >= MATCH_THRESHOLD and best_pid is not None:
            boosted_static[best_pid] = boosted_static.get(best_pid, 0.0) + best_sim
        elif best_sim < GRAY_ZONE_LOWER:
            pending_cluster.append(paper)

    return {
        "boosted_static": boosted_static,
        "pending_cluster": pending_cluster,
        "match_details": match_details,
    }
