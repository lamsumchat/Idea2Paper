from __future__ import annotations

import logging
import time
from typing import List, Optional

import httpx

from idea2paper.agentic_search.paper_sources.base import ExternalPaper, PaperSource

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
_FIELDS = "title,abstract,authors,venue,year,externalIds"
_TIMEOUT = 15.0
_REQUEST_INTERVAL = 1.0


class SemanticScholarSource(PaperSource):
    def __init__(self, api_key: str | None = None) -> None:
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["x-api-key"] = api_key
        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _REQUEST_INTERVAL:
            time.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def search(
        self,
        query: str,
        max_results: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        venue_filter: str | None = None,
    ) -> List[ExternalPaper]:
        params: dict[str, str | int] = {
            "query": query,
            "limit": max_results,
            "fields": _FIELDS,
        }

        if year_from or year_to:
            lo = year_from or ""
            hi = year_to or ""
            params["year"] = f"{lo}-{hi}"

        if venue_filter:
            params["venue"] = venue_filter

        self._throttle()
        try:
            resp = httpx.get(
                f"{_BASE_URL}/search",
                params=params,
                headers=self._headers,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("S2 search failed for query=%r: %s", query, exc)
            return []

        papers: List[ExternalPaper] = []
        for item in data.get("data") or []:
            paper = self._parse_item(item)
            if paper:
                papers.append(paper)
        return papers

    def get_paper_details(self, paper_id: str) -> Optional[ExternalPaper]:
        self._throttle()
        try:
            resp = httpx.get(
                f"{_BASE_URL}/{paper_id}",
                params={"fields": _FIELDS},
                headers=self._headers,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("S2 detail lookup failed for id=%r: %s", paper_id, exc)
            return None

        return self._parse_item(data)

    def get_abstract_by_title(self, title: str) -> Optional[str]:
        """Look up a paper by exact title and return its abstract."""
        self._throttle()
        try:
            resp = httpx.get(
                f"{_BASE_URL}/search/match",
                params={"query": title, "fields": "title,abstract"},
                headers=self._headers,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("S2 title match failed for %r: %s", title, exc)
            return None

        return (data.get("data") or [{}])[0].get("abstract") if data.get("data") else None

    @staticmethod
    def _parse_item(item: dict) -> Optional[ExternalPaper]:
        if not item or not item.get("title"):
            return None

        authors = [a.get("name", "") for a in (item.get("authors") or [])]
        ext_ids = item.get("externalIds") or {}

        return ExternalPaper(
            paper_id=item.get("paperId", ""),
            title=item["title"],
            authors=authors,
            venue=item.get("venue", ""),
            year=item.get("year") or 0,
            abstract=item.get("abstract"),
            doi=ext_ids.get("DOI"),
            url=f"https://api.semanticscholar.org/paper/{item.get('paperId', '')}",
            source="semantic_scholar",
        )
