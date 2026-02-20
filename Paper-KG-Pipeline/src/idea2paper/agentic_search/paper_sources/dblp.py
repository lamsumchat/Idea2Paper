from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from idea2paper.agentic_search.paper_sources.base import ExternalPaper, PaperSource

logger = logging.getLogger(__name__)

TOP_VENUES = [
    "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV",
    "ACL", "EMNLP", "NAACL",
    "AAAI", "IJCAI",
    "KDD", "WWW", "SIGIR",
    "ICSE", "FSE",
]

_SEARCH_URL = "https://dblp.org/search/publ/api"
_TIMEOUT = 15.0


class DblpSource(PaperSource):
    def search(
        self,
        query: str,
        max_results: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        venue_filter: str | None = None,
    ) -> List[ExternalPaper]:
        q = query
        if venue_filter:
            q = f"{q} venue:{venue_filter}"

        params = {"q": q, "h": max_results, "format": "json"}

        try:
            resp = httpx.get(_SEARCH_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("DBLP search failed for query=%r: %s", query, exc)
            return []

        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not isinstance(hits, list):
            return []

        papers: List[ExternalPaper] = []
        for hit in hits:
            paper = self._parse_hit(hit)
            if paper is None:
                continue
            if year_from and paper.year < year_from:
                continue
            if year_to and paper.year > year_to:
                continue
            papers.append(paper)

        return papers

    def get_paper_details(self, paper_id: str) -> Optional[ExternalPaper]:
        params = {"q": paper_id, "h": 1, "format": "json"}

        try:
            resp = httpx.get(_SEARCH_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("DBLP detail lookup failed for id=%r: %s", paper_id, exc)
            return None

        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return None

        return self._parse_hit(hits[0])

    @staticmethod
    def _parse_hit(hit: dict) -> Optional[ExternalPaper]:
        info = hit.get("info")
        if not info:
            return None

        authors_raw = info.get("authors", {}).get("author", [])
        if isinstance(authors_raw, dict):
            authors_raw = [authors_raw]
        authors = [a.get("text", a) if isinstance(a, dict) else str(a) for a in authors_raw]

        try:
            year = int(info.get("year", 0))
        except (ValueError, TypeError):
            year = 0

        return ExternalPaper(
            paper_id=info.get("key", ""),
            title=info.get("title", "").rstrip("."),
            authors=authors,
            venue=info.get("venue", ""),
            year=year,
            doi=info.get("doi"),
            url=info.get("url"),
            source="dblp",
        )
