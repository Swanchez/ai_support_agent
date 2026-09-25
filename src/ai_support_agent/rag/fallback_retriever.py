"""Retrieval policy for using external reference material only as a fallback."""

from collections.abc import Callable
from dataclasses import dataclass, field

from ai_support_agent.rag.retriever import RetrievedChunk, Retriever


@dataclass
class FallbackRetriever:
    """Search primary knowledge first, then use a separate fallback collection if empty."""

    primary: Retriever
    fallback_factory: Callable[[], Retriever]
    fallback_threshold: float
    _fallback: Retriever | None = field(default=None, init=False, repr=False)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return primary evidence whenever it exists; never mix both source collections."""

        primary_arguments = {
            "top_k": top_k,
            "threshold": threshold,
            "max_chunks_per_document": max_chunks_per_document,
        }
        primary_results = self.primary.retrieve(question, **primary_arguments)
        if primary_results:
            return primary_results
        if self._fallback is None:
            self._fallback = self.fallback_factory()
        return self._fallback.retrieve(
            question,
            top_k=top_k,
            threshold=self.fallback_threshold,
            max_chunks_per_document=max_chunks_per_document,
        )
