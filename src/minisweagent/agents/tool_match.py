from minisweagent.retrieval.bm25.index import tokenizer_tiktoken, bm25, top_k_elements
from minisweagent import Model, ToolDescription

class ToolMatch():
    def tool_match(self, tools: list[ToolDescription], query: str, model: Model) -> ToolDescription:
        """
        tools: list of dictionaries with 'name' and 'description'
            e.g. [{"name": "grep", "description": "Search for a pattern in files"}]
        query: LLM reasoning text about which tool to use
        """

        corpus = []
        for tool in tools:
            tokens = tokenizer_tiktoken(tool["description"])
            corpus.append(tokens)

        query_tokens = tokenizer_tiktoken(query)
        scores = bm25(corpus, query_tokens)
        return top_k_elements(tools, scores, 1)[0]