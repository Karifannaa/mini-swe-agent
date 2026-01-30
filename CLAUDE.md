# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mini-swe-agent is a minimalist AI software engineering agent that solves GitHub issues and programming challenges. The design philosophy prioritizes simplicity: ~100 lines of core agent code, bash-only execution (no custom tool abstractions), and linear message history.

## Commands

```bash
# Install for development
pip install -e '.[dev]'
pre-commit install

# Run tests
pytest -n auto                          # Parallel execution
pytest tests/agents/test_default.py     # Single file
pytest -k "not slow"                    # Skip slow tests

# Lint and format
pre-commit run --all-files              # All hooks
ruff check --fix                        # Auto-fix lint issues
ruff format                             # Format code

# Run the agent
mini -t "your task"                     # Simple CLI
mini -v                                 # Textual TUI mode
```

## Architecture

**Core concept**: Agent = Model + Environment + Agent Logic

```
src/minisweagent/
├── __init__.py       # Protocols/interfaces (Agent, AgentEx, Model, Environment, ToolDescription)
├── agents/
│   ├── default.py           # Base agent (~100 lines)
│   ├── long_context.py      # LongContextAgent with BM25 retrieval
│   ├── history_retrieval.py # BM25-based history filtering
│   └── tool_match.py        # Tool selection via BM25
├── environments/     # Action execution (local, docker, singularity)
├── models/           # LM interfaces (litellm, anthropic, openrouter)
├── retrieval/
│   └── bm25/index.py # BM25 scoring with tiktoken tokenizer
├── run/              # Entry point scripts
└── config/           # YAML templates with Jinja2 rendering
```

**Control flow**: `DefaultAgent.run()` → loop of `query()` (LM call) → `parse_action()` (extract bash from markdown) → `execute_action()` (subprocess.run) → check termination → append to messages

**Key design decisions**:
- Stateless execution: each action is a separate `subprocess.run()` call
- Linear message history: no pruning, what the LM sees = what's in `self.messages`
- Protocol-based polymorphism: swap Model/Environment/Agent implementations freely
- Config-driven: YAML files with Jinja2 templates control prompts and behavior

## Agent Types

### DefaultAgent
Base agent with linear message history. Simple loop: query → parse → execute → observe.

### LongContextAgent
Extended agent for handling long contexts efficiently:
- Maintains `summary_messages` (condensed history) alongside full `messages`
- Uses BM25 retrieval to select relevant history for each query
- Automatic tool selection based on LLM reasoning text

**Key components**:
- `HistoryRetriever.retrieve()` - filters history using BM25 similarity
- `ToolMatch.tool_match()` - selects best tool from `config.tools` based on query
- `summary_messages` - truncated observation history for context efficiency

**Flow**:
```
step() → query summary_messages → extract thinking
      → tool_match() selects tool
      → retrieve() filters relevant history
      → query() with filtered context
      → update summary with truncated observation
```

## Protocols and Types

```python
# Base protocols in __init__.py
Agent          # Base agent protocol
AgentEx        # Extended protocol with summary_messages
Model          # LM interface
Environment    # Execution environment

# Tool definition
ToolDescription = TypedDict('ToolDescription', {
    'name': str,
    'description': str,
    'prompt_instruction': NotRequired[str]
})
```

## Configuration

Configs are YAML files with Jinja2 templates. Key fields in `AgentConfig`:
- `system_template`, `instance_template` - prompts
- `action_regex` - pattern to extract bash commands
- `tools` - list of `ToolDescription` for LongContextAgent

Example config: `config/default_with_tools.yaml`

## Code Style

- Python 3.10+, type annotations (`list` not `List`)
- `pathlib` over `os.path`, `Path.read_text()` over `with open()`
- `typer` for CLI, `jinja2` for templates, `pydantic` for config
- Minimal code: avoid intermediate variables, don't catch exceptions unless necessary
- Minimal comments: only for complex logic

**Test style**:
- `pytest` only, no mocking unless explicitly requested
- Inline assertions: `assert func() == expected` (not `result = func(); assert result == expected`)
- `pytest.mark.parametrize`: first arg is tuple, second is list
- Print statements in tests are OK

## Dependencies

Core: `litellm`, `jinja2`, `pydantic`, `typer`, `rich`

For LongContextAgent: `rank_bm25`, `tiktoken`
