from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import List, Optional


@dataclasses.dataclass
class ExternalPaper:
    """Standardized representation of an externally fetched paper."""

    paper_id: str
    title: str
    authors: List[str]
    venue: str
    year: int
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: str = ""


class PaperSource(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        venue_filter: str | None = None,
    ) -> List[ExternalPaper]: ...

    @abstractmethod
    def get_paper_details(self, paper_id: str) -> Optional[ExternalPaper]: ...
