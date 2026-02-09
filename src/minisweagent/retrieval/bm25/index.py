import re
import tiktoken
import heapq
from rank_bm25 import BM25Okapi



def tokenizer_simple(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_ ]", " ", text)
    tokens = text.split()
    return tokens


def tokenizer_tiktoken(text: str) -> list[str]:
    """Tokenize text using tiktoken for BM25 retrieval."""
    if not text:
        return []

    # Flatten string
    text = text.strip().lower()
    if not text:
        return []

    # Tokenize
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    tokens = [str(token_id) for token_id in token_ids]

    return tokens

def top_k_elements(xs, scores, k):
    assert len(xs) == len(scores)
    top_indices = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
    return [xs[i] for i in top_indices]

def bm25(documents: list[list[str]], query: list[str]) -> list[float]:
    bm25_index = BM25Okapi(documents)
    scores = bm25_index.get_scores(query).tolist()
    return scores