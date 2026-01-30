from minisweagent.agents.history_retrieval import HistoryRetriever


def test_retrieve_bash_tool_usage_with_query():
    """Test retrieving history for bash tool usage with combined step text + tool description."""

    # Real-world conversation history with bash tool usage
    history = [
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "I'm trying to debug a Python script that's failing"},
        {"role": "assistant", "content": "I'll help you debug your Python script. First, let me try to run it to see the error:\n```bash\npython my_script.py\n```"},
        {"role": "user", "content": "The error shows it's trying to import pandas but it's not installed"},
        {"role": "assistant", "content": "You can install pandas using pip:\n```bash\npip install pandas\n```"},
        {"role": "user", "content": "How do I know which Python version I'm using?"},
        {"role": "assistant", "content": "Check your Python version with:\n```bash\npython --version\n```"},
        {"role": "user", "content": "I need to find where Python packages are installed"},
        {"role": "assistant", "content": "You can see your Python site-packages location:\n```bash\npython -c \"import site; print(site.getsitepackages())\"\n```"},
    ]

    # Simulate a query that combines step text + tool description
    # Example: "debug python script" + "bash tool that runs python commands"
    query = "debug python script bash run python commands"
    retriever = HistoryRetriever()
    filtered_history = retriever.retrieve(history=history, query=query, model=None, top_k=3)

    # Should retrieve the bash-related messages about debugging
    bash_commands = [
        msg for msg in filtered_history
        if '```bash' in msg['content'] and ('python' in msg['content'])
    ]
    assert len(bash_commands) >= 1, "Should retrieve bash commands related to Python debugging"


def test_retrieve_file_reading_tool_usage():
    """Test retrieving history for file reading tool usage with realistic queries."""

    # Conversation involving file operations
    history = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "I need to examine the contents of a configuration file"},
        {"role": "assistant", "content": "Let me read the configuration file to see its contents:\n```bash\ncat config.yaml\n```"},
        {"role": "user", "content": "I see the configuration has database settings. Let me check if there's a backup"},
        {"role": "assistant", "content": "You can see the backup files in the directory:\n```bash\nls -la backup*\n```"},
        {"role": "user", "content": "I need to find specific error messages in a large log file"},
        {"role": "assistant", "content": "Let me search for error patterns in the log file:\n```bash\ngrep 'ERROR' large_log_file.log\n```"},
        {"role": "user", "content": "How can I check the line count of the file?"},
        {"role": "assistant", "content": "You can count lines using wc:\n```bash\nwc -l large_log_file.log\n```"},
    ]

    # Query that combines intent "find errors in log" + "grep tool"
    query = "find error messages log file grep search text"
    retriever = HistoryRetriever()
    filtered_history = retriever.retrieve(history=history, query=query, model=None, top_k=3)

    # Should retrieve messages with grep or file searching
    grep_commands = [
        msg for msg in filtered_history
        if 'grep' in msg['content'] or 'search' in msg['content'].lower()
    ]
    assert len(grep_commands) >= 1, "Should retrieve grep-based commands for searching files"
