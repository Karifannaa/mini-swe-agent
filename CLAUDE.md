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
├── __init__.py       # Protocols/interfaces for all base classes
├── agents/           # Agent control flow (DefaultAgent is ~100 lines)
├── environments/     # Action execution (local, docker, singularity)
├── models/           # LM interfaces (litellm, anthropic, openrouter)
├── run/              # Entry point scripts
└── config/           # YAML templates with Jinja2 rendering
```

**Control flow**: `DefaultAgent.run()` → loop of `query()` (LM call) → `parse_action()` (extract bash from markdown) → `execute_action()` (subprocess.run) → check termination → append to messages

**Key design decisions**:
- Stateless execution: each action is a separate `subprocess.run()` call
- Linear message history: no pruning, what the LM sees = what's in `self.messages`
- Protocol-based polymorphism: swap Model/Environment/Agent implementations freely
- Config-driven: YAML files with Jinja2 templates control prompts and behavior

## Code Style

- Python 3.10+, type annotations (`list` not `List`)
- `pathlib` over `os.path`, `Path.read_text()` over `with open()`
- `typer` for CLI, `jinja2` for templates, `dataclass` for config
- Minimal code: avoid intermediate variables, don't catch exceptions unless necessary
- Minimal comments: only for complex logic

**Test style**:
- `pytest` only, no mocking unless explicitly requested
- Inline assertions: `assert func() == expected` (not `result = func(); assert result == expected`)
- `pytest.mark.parametrize`: first arg is tuple, second is list
- Print statements in tests are OK
