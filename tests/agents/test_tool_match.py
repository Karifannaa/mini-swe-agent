from minisweagent.agents.tool_match import ToolMatch
from minisweagent.retrieval.bm25.index import top_k_elements


def test_top_k_elements():
    xs = ['a', 'b', 'c']
    scores = [2, 1, 3]
    assert top_k_elements(xs, scores, 1)[0] == 'c'
    assert top_k_elements(xs, scores, 2)[1] == 'a'


def test_select_tool_basic_usage():
    """Test that select_tool returns the most relevant tool based on query."""

    tools = [
        {"name": "grep", "description": "Search for a specific pattern or word inside files"},
        {"name": "ls", "description": "List files and directories in the current folder"},
        {"name": "cd", "description": "Change the current working directory"},
        {"name": "read_file", "description": "Read the contents of a file"},
        {"name": "execute_command", "description": "Run a system command or script"},
        {"name": "verify", "description": "Check or validate the correctness of something"}
    ]

    test_cases = [
        ("I need to find a specific word inside files", "grep"),
        ("I want to see all files in the folder", "ls"),
        ("Move to another directory", "cd"),
        ("Open a file and read its content", "read_file"),
        ("Run a script on the system", "execute_command"),
        ("Check if the results are correct", "verify"),
    ]

    matcher = ToolMatch()
    for query, expected_tool in test_cases:
        selected_tool = matcher.tool_match(tools, query, None)
        assert selected_tool['name'] == expected_tool, f"Expected '{expected_tool}', got '{selected_tool}'"
