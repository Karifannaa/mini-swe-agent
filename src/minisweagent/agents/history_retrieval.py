from typing import Any

from minisweagent import Model
from minisweagent.retrieval.bm25.index import bm25, tokenizer_tiktoken


class HistoryRetriever:
    def retrieve(self, history: list[dict[str, Any]], query: str, model: Model | None, top_k: int = 10) -> list[dict[str, Any]]:
        """Filter history using BM25 retrieval for relevant examples."""
        tokenizer = tokenizer_tiktoken
        if not history or not query:
            return history

        # Prepare non-system messages for retrieval
        retrieval_candidates = [
            msg for msg in history
            if msg.get('role') != 'system' and msg.get('content')
        ]

        if not retrieval_candidates:
            return history

        # Calculate BM25 similarity scores
        query_tokens = tokenizer(query)
        candidate_docs = [tokenizer(msg['content']) for msg in retrieval_candidates]
        relevance_scores = bm25(documents=candidate_docs, query=query_tokens)

        # Select top-k relevant documents
        top_relevant_indices = [
            idx for idx, score in enumerate(relevance_scores) if score > 0
        ]
        top_relevant_indices.sort(key=lambda idx: relevance_scores[idx], reverse=True)
        top_relevant_indices = top_relevant_indices[:top_k]

        # Map candidates to original history positions
        candidate_to_original_index = []
        for i, msg in enumerate(history):
            if msg.get('role') != 'system' and msg.get('content'):
                candidate_to_original_index.append(i)

        # Track which original messages to keep
        preserve_indices = set()
        for candidate_idx in top_relevant_indices:
            original_index = candidate_to_original_index[candidate_idx]
            preserve_indices.add(original_index)

        # Reconstruct history while preserving order and context
        result_history = []
        for i, msg in enumerate(history):
            if i in preserve_indices:
                result_history.append(msg)
            elif msg.get('role') == 'system' and self._has_context_preserved(history, i, preserve_indices):
                result_history.append(msg)

        return result_history

    def _has_context_preserved(self, history: list[dict[str, Any]], system_index: int, preserve_indices: set) -> bool:
        """Check if the next message after a system message should be preserved for context."""
        for i in range(system_index + 1, len(history)):
            if i in preserve_indices:
                return True
        return False
