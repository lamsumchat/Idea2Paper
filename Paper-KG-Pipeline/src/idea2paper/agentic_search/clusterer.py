from __future__ import annotations

from idea2paper.infra.llm import call_llm, parse_json_from_llm

GROUPING_PROMPT = """\
You are a research clustering assistant.
Given the following papers, group them into topical clusters.
Each cluster should contain papers that share a similar research theme.

Papers:
{paper_list}

For each cluster, provide:
- "group_name": a short descriptive name
- "paper_indices": list of paper indices (0-based) belonging to this group
- "representative_idea": a one-sentence summary of the shared idea
- "base_problem": the common problem these papers address
- "solution_pattern": the common approach or methodology

Return ONLY a JSON object:
{{"clusters": [
  {{"group_name": "...", "paper_indices": [...], "representative_idea": "...", "base_problem": "...", "solution_pattern": "..."}},
  ...
]}}"""

QUALITY_PROMPT = """\
Rate the quality of the following research pattern on a scale from 0.0 to 1.0.
Consider: specificity, coherence, and research value.

Pattern name: {name}
Representative idea: {idea}
Problem: {problem}
Solution: {solution}

Return ONLY a JSON object: {{"quality_score": <float between 0.0 and 1.0>}}"""


def _build_single_pattern(paper: dict, pattern_idx: int) -> tuple[str, dict, float]:
    fields = paper.get("fields", {})
    pattern_id = f"dyn_pattern_{pattern_idx}"
    pattern_info = {
        "pattern_id": pattern_id,
        "name": fields.get("idea_summary", "")[:100],
        "size": 1,
        "domain": fields.get("domain", ""),
        "sub_domains": fields.get("sub_domains", []),
        "llm_enhanced_summary": {
            "representative_ideas": fields.get("idea_summary", ""),
            "base_problem": fields.get("problem_definition", ""),
            "solution_pattern": fields.get("solution_pattern", ""),
        },
        "dynamic": True,
    }
    return pattern_id, pattern_info, 0.0


def _score_pattern(pattern_info: dict) -> float:
    """Ask LLM to rate a dynamic pattern's quality on 0-1 scale."""
    enhanced = pattern_info.get("llm_enhanced_summary", {})
    prompt = QUALITY_PROMPT.format(
        name=pattern_info.get("name", ""),
        idea=enhanced.get("representative_ideas", ""),
        problem=enhanced.get("base_problem", ""),
        solution=enhanced.get("solution_pattern", ""),
    )
    response = call_llm(prompt, temperature=0.0, max_tokens=128)
    if not response:
        return 0.0
    parsed = parse_json_from_llm(response)
    if parsed is None:
        return 0.0
    try:
        score = float(parsed.get("quality_score", 0.0))
        return max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        return 0.0


def _cluster_via_llm(papers: list[dict]) -> list[dict] | None:
    """Ask LLM to group papers into clusters. Returns parsed cluster list or None."""
    lines = []
    for i, p in enumerate(papers):
        fields = p.get("fields", {})
        title = p.get("title", "Untitled")
        idea = fields.get("idea_summary", "N/A")
        lines.append(f"[{i}] Title: {title}\n    Idea: {idea}")
    paper_list = "\n".join(lines)

    response = call_llm(
        GROUPING_PROMPT.format(paper_list=paper_list),
        temperature=0.0,
        max_tokens=2048,
    )
    if not response:
        return None
    parsed = parse_json_from_llm(response)
    if parsed is None or "clusters" not in parsed:
        return None
    return parsed["clusters"]


def cluster_and_extract_patterns(
    papers: list[dict],
    quality_threshold: float = 0.5,
) -> list[tuple[str, dict, float]]:
    """
    Create dynamic patterns from unmatched papers.

    <= 2 papers: each gets its own pattern via direct field extraction.
    >= 3 papers: LLM groups them, each group becomes one pattern.

    Patterns scoring below quality_threshold are discarded.
    """
    if not papers:
        return []

    results: list[tuple[str, dict, float]] = []
    pattern_counter = 0

    if len(papers) <= 2:
        for paper in papers:
            pid, info, _ = _build_single_pattern(paper, pattern_counter)
            score = _score_pattern(info)
            if score >= quality_threshold:
                results.append((pid, info, score))
            pattern_counter += 1
        return results

    clusters = _cluster_via_llm(papers)

    if clusters is None:
        for paper in papers:
            pid, info, _ = _build_single_pattern(paper, pattern_counter)
            score = _score_pattern(info)
            if score >= quality_threshold:
                results.append((pid, info, score))
            pattern_counter += 1
        return results

    for cluster in clusters:
        pattern_id = f"dyn_pattern_{pattern_counter}"
        indices = cluster.get("paper_indices", [])
        group_papers = [papers[i] for i in indices if 0 <= i < len(papers)]

        sub_domains: list[str] = []
        domain = ""
        for gp in group_papers:
            f = gp.get("fields", {})
            sub_domains.extend(f.get("sub_domains", []))
            if not domain:
                domain = f.get("domain", "")
        sub_domains = list(dict.fromkeys(sub_domains))

        pattern_info = {
            "pattern_id": pattern_id,
            "name": cluster.get("group_name", "")[:100],
            "size": len(group_papers),
            "domain": domain,
            "sub_domains": sub_domains,
            "llm_enhanced_summary": {
                "representative_ideas": cluster.get("representative_idea", ""),
                "base_problem": cluster.get("base_problem", ""),
                "solution_pattern": cluster.get("solution_pattern", ""),
            },
            "dynamic": True,
        }

        score = _score_pattern(pattern_info)
        if score >= quality_threshold:
            results.append((pattern_id, pattern_info, score))
        pattern_counter += 1

    return results
