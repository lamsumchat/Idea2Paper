"""
Agentic Search Orchestrator

Coordinates the full agentic search pipeline:
  Stage 2: Adaptive decision (quality scoring)
  Stage 3: Iterative search via DBLP + Semantic Scholar
  Stage 4: Three-layer dedup
  Stage 5: Pattern matching
  Stage 6: Clustering unmatched papers
  Stage 7: Merging static + dynamic patterns
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from idea2paper.agentic_search.scorer import compute_recall_quality, determine_search_params
from idea2paper.agentic_search.paper_sources.base import ExternalPaper
from idea2paper.agentic_search.paper_sources.dblp import DblpSource, TOP_VENUES
from idea2paper.agentic_search.paper_sources.semantic_scholar import SemanticScholarSource
from idea2paper.agentic_search.dedup import run_dedup_pipeline
from idea2paper.agentic_search.pattern_matcher import extract_paper_fields, match_papers_to_patterns
from idea2paper.agentic_search.clusterer import cluster_and_extract_patterns
from idea2paper.agentic_search.merger import merge_patterns
from idea2paper.config import PipelineConfig
from idea2paper.infra.llm import call_llm, parse_json_from_llm
from idea2paper.infra.embeddings import get_embedding, get_embeddings_batch
from idea2paper.infra.run_context import get_logger


class AgenticSearchOrchestrator:
    """Runs the full agentic search pipeline after static recall."""

    def __init__(
        self,
        user_idea: str,
        static_results: List[Tuple[str, Dict, float]],
        patterns: List[Dict],
        existing_paper_ids: Set[str],
        logger=None,
    ):
        self.user_idea = user_idea
        self.static_results = static_results
        self.patterns = patterns
        self.existing_paper_ids = existing_paper_ids
        self.logger = logger or get_logger()

        sources = PipelineConfig.AGENTIC_SEARCH_SOURCES
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",")]
        self._sources = sources

        self._dblp = DblpSource() if "dblp" in self._sources else None
        s2_key = PipelineConfig.AGENTIC_SEARCH_SEMANTIC_SCHOLAR_API_KEY
        self._s2 = SemanticScholarSource(api_key=s2_key or None) if "semantic_scholar" in self._sources else None

    def run(self) -> Dict:
        """Execute the full agentic search pipeline.

        Returns a dict with keys:
            merged_patterns: final merged pattern list
            search_meta: metadata about the search process
        """
        pattern_scores = [score for _, _, score in self.static_results]

        # Stage 2: Adaptive decision
        avg_score = compute_recall_quality(pattern_scores)
        search_params = determine_search_params(avg_score)
        print(f"\n{'='*80}")
        print(f"🔎 Agentic Search: avg_score={avg_score:.3f}, "
              f"search_count={search_params['search_count']}, "
              f"year_from={search_params['year_from']}")
        print(f"{'='*80}")

        if self.logger:
            self.logger.log_event("agentic_search_start", {
                "avg_score": avg_score,
                **search_params,
            })

        # Stage 3: Iterative search
        all_papers = self._iterative_search(search_params)
        print(f"  ✓ 搜索到 {len(all_papers)} 篇外部论文")

        if not all_papers:
            print("  ⚠️  未搜索到外部论文，跳过 agentic search")
            return {
                "merged_patterns": self.static_results,
                "search_meta": {"avg_score": avg_score, "papers_found": 0, "skipped": True},
            }

        # Enrich DBLP papers with abstracts from Semantic Scholar
        all_papers = self._enrich_abstracts(all_papers)

        # Stage 4: Dedup
        embeddings = self._compute_paper_embeddings(all_papers)
        deduped = run_dedup_pipeline(
            all_papers,
            self.existing_paper_ids,
            embeddings,
            year_from=search_params["year_from"],
        )
        print(f"  ✓ 去重后保留 {len(deduped)} 篇")

        if not deduped:
            print("  ⚠️  去重后无剩余论文")
            return {
                "merged_patterns": self.static_results,
                "search_meta": {"avg_score": avg_score, "papers_found": len(all_papers), "after_dedup": 0},
            }

        # Truncate to final_top_k by relevance
        if len(deduped) > PipelineConfig.AGENTIC_SEARCH_FINAL_TOP_K:
            deduped = self._rank_by_relevance(deduped)[:PipelineConfig.AGENTIC_SEARCH_FINAL_TOP_K]

        # Stage 5: Extract fields + match to patterns
        paper_dicts = self._prepare_paper_dicts(deduped)
        match_result = match_papers_to_patterns(paper_dicts, self.patterns, logger=self.logger)
        boosted_static = match_result["boosted_static"]
        pending_cluster = match_result["pending_cluster"]

        print(f"  ✓ {len(boosted_static)} 个 Pattern 得分增强, "
              f"{len(pending_cluster)} 篇待聚类")

        # Stage 6: Cluster unmatched → dynamic patterns
        dynamic_patterns: List[Tuple[str, Dict, float]] = []
        if pending_cluster:
            dynamic_patterns = cluster_and_extract_patterns(pending_cluster)
            print(f"  ✓ 生成 {len(dynamic_patterns)} 个动态 Pattern")

        # Stage 7: Merge
        merged = merge_patterns(
            static_results=self.static_results,
            dynamic_patterns=dynamic_patterns,
            boosted_static=boosted_static,
            max_dynamic=PipelineConfig.AGENTIC_SEARCH_MAX_DYNAMIC_PATTERNS,
            dynamic_weight_cap=PipelineConfig.AGENTIC_SEARCH_DYNAMIC_WEIGHT_CAP,
        )
        print(f"  ✓ 合并后 Top-{len(merged)} Pattern")

        search_meta = {
            "avg_score": avg_score,
            "search_params": search_params,
            "papers_found": len(all_papers),
            "after_dedup": len(deduped),
            "boosted_patterns": len(boosted_static),
            "dynamic_patterns": len(dynamic_patterns),
            "match_details": match_result.get("match_details", []),
        }
        if self.logger:
            self.logger.log_event("agentic_search_end", search_meta)

        return {"merged_patterns": merged, "search_meta": search_meta}

    def _generate_queries(self, round_num: int, prev_papers: List[ExternalPaper] | None = None) -> List[str]:
        """Use LLM to decompose the user idea into search queries."""
        if round_num == 1:
            prompt = f"""You are a research assistant. Given the following research idea, generate 3-5 diverse search queries to find relevant academic papers on DBLP.

Research idea: {self.user_idea}

Requirements:
- Each query should capture a different aspect of the idea
- Use technical terms that would appear in paper titles
- Include both specific and slightly broader queries
- Output as a JSON array of strings

Example output: ["query1", "query2", "query3"]"""
        else:
            prev_titles = [p.title for p in (prev_papers or [])[:5]]
            prompt = f"""You are a research assistant. In a previous search round, you found these papers:
{json.dumps(prev_titles, indent=2)}

But they did not sufficiently cover the research idea:
{self.user_idea}

Generate 3 refined search queries to find papers that fill the gaps. Consider:
- Broadening terms if the previous search was too narrow
- Narrowing terms if it was too broad
- Trying different conference communities

Output as a JSON array of strings."""

        response = call_llm(prompt, temperature=0.3, max_tokens=512, timeout=30)
        parsed = parse_json_from_llm(response)
        if isinstance(parsed, list):
            return [str(q) for q in parsed if isinstance(q, str)]
        if isinstance(parsed, dict) and "queries" in parsed:
            return [str(q) for q in parsed["queries"]]
        lines = [l.strip().strip('"').strip("'") for l in response.strip().split("\n") if l.strip()]
        return lines[:5] or [self.user_idea]

    def _iterative_search(self, search_params: Dict) -> List[ExternalPaper]:
        """Execute multi-round search with quality checking."""
        max_rounds = PipelineConfig.AGENTIC_SEARCH_MAX_ROUNDS
        target_count = search_params["search_count"]
        year_from = search_params["year_from"]
        year_quotas = search_params["year_quotas"]
        threshold = PipelineConfig.AGENTIC_SEARCH_RELEVANCE_THRESHOLD

        all_papers: List[ExternalPaper] = []
        seen_ids: Set[str] = set()

        for round_num in range(1, max_rounds + 1):
            print(f"\n  [Round {round_num}/{max_rounds}] 生成搜索 query...")
            queries = self._generate_queries(round_num, all_papers if round_num > 1 else None)
            print(f"    Queries: {queries}")

            round_papers: List[ExternalPaper] = []
            for query in queries:
                for year, quota in sorted(year_quotas.items(), reverse=True):
                    if not self._dblp:
                        continue
                    # Search per-venue for better precision
                    for venue in TOP_VENUES[:8]:
                        papers = self._dblp.search(
                            query,
                            max_results=quota,
                            year_from=year,
                            year_to=year,
                            venue_filter=venue,
                        )
                        for p in papers:
                            if p.paper_id not in seen_ids:
                                seen_ids.add(p.paper_id)
                                round_papers.append(p)

                    # Also try Semantic Scholar for broader coverage
                    if self._s2:
                        s2_papers = self._s2.search(
                            query,
                            max_results=quota,
                            year_from=year,
                            year_to=year,
                        )
                        for p in s2_papers:
                            if p.paper_id not in seen_ids:
                                seen_ids.add(p.paper_id)
                                round_papers.append(p)

                time.sleep(0.5)

            all_papers.extend(round_papers)
            print(f"    Round {round_num}: 找到 {len(round_papers)} 篇新论文 (累计 {len(all_papers)})")

            if len(all_papers) >= target_count:
                break

            # Quality check: if enough high-relevance papers, stop early
            if round_num < max_rounds and round_papers:
                avg_sim = self._quick_relevance_check(round_papers)
                print(f"    平均相关度: {avg_sim:.3f}")
                if avg_sim >= threshold and len(all_papers) >= target_count // 2:
                    print(f"    ✓ 质量足够，提前结束搜索")
                    break

        return all_papers[:target_count * 2]

    def _enrich_abstracts(self, papers: List[ExternalPaper]) -> List[ExternalPaper]:
        """Fetch abstracts from Semantic Scholar for papers that lack them."""
        if not self._s2:
            return papers

        enriched = []
        for p in papers:
            if p.abstract:
                enriched.append(p)
                continue
            abstract = self._s2.get_abstract_by_title(p.title)
            if abstract:
                p.abstract = abstract
            enriched.append(p)
            time.sleep(0.3)
        return enriched

    def _compute_paper_embeddings(self, papers: List[ExternalPaper]) -> np.ndarray | None:
        """Compute embeddings for dedup and ranking."""
        texts = [f"{p.title} {p.abstract or ''}" for p in papers]
        embs = get_embeddings_batch(texts, logger=self.logger, timeout=30)
        if embs is None:
            return None
        return np.array(embs, dtype=np.float32)

    def _quick_relevance_check(self, papers: List[ExternalPaper]) -> float:
        """Fast relevance check: cosine similarity between user idea and paper titles."""
        idea_emb = get_embedding(self.user_idea, logger=self.logger, timeout=10)
        if idea_emb is None:
            return 0.5

        titles = [p.title for p in papers[:10]]
        title_embs = get_embeddings_batch(titles, logger=self.logger, timeout=10)
        if title_embs is None:
            return 0.5

        idea_vec = np.array(idea_emb, dtype=np.float32)
        title_mat = np.array(title_embs, dtype=np.float32)
        idea_norm = np.linalg.norm(idea_vec)
        if idea_norm == 0:
            return 0.0
        title_norms = np.linalg.norm(title_mat, axis=1)
        title_norms[title_norms == 0] = 1.0
        scores = (title_mat @ idea_vec) / (title_norms * idea_norm)
        return float(np.mean(scores))

    def _rank_by_relevance(self, papers: List[ExternalPaper]) -> List[ExternalPaper]:
        """Rank papers by embedding similarity to user idea."""
        idea_emb = get_embedding(self.user_idea, logger=self.logger, timeout=10)
        if idea_emb is None:
            return papers

        texts = [f"{p.title} {p.abstract or ''}" for p in papers]
        embs = get_embeddings_batch(texts, logger=self.logger, timeout=15)
        if embs is None:
            return papers

        idea_vec = np.array(idea_emb, dtype=np.float32)
        mat = np.array(embs, dtype=np.float32)
        idea_norm = np.linalg.norm(idea_vec)
        if idea_norm == 0:
            return papers
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        scores = (mat @ idea_vec) / (norms * idea_norm)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [papers[i] for i, _ in indexed]

    def _prepare_paper_dicts(self, papers: List[ExternalPaper]) -> List[Dict]:
        """Convert ExternalPaper to dict format for pattern matching, with LLM field extraction."""
        result = []
        for p in papers:
            if not p.abstract:
                continue
            fields = extract_paper_fields(p.title, p.abstract)
            if fields is None:
                continue
            result.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "venue": p.venue,
                "year": p.year,
                "authors": p.authors,
                "fields": fields,
            })
        return result
