"""Tests for history_summary: verify all configured tools produce usable summaries."""

import pytest
from minisweagent.agents.history_summary import (
    step_summary,
    _summarize_grep,
    _summarize_ls,
    _summarize_read_file,
    _summarize_cd,
    _summarize_find,
    _summarize_default,
)


# ── Realistic tool observations ──────────────────────────────────────────────

LS_SHORT = {
    "action": "ls",
    "output": "CLAUDE.md\nREADME.md\npyproject.toml\nsrc\ntests\n",
    "returncode": 0,
}

LS_WITH_DIR = {
    "action": "ls -la src/minisweagent/agents",
    "output": "\n".join([f"file_{i}.py" for i in range(50)]),
    "returncode": 0,
}

LS_EMPTY = {
    "action": "ls /empty",
    "output": "",
    "returncode": 0,
}

GREP_WITH_MATCHES = {
    "action": "grep -rn 'def step' src/",
    "output": "\n".join([
        "src/agents/default.py:45:    def step(self):",
        "src/agents/long_context.py:141:    def step(self) -> dict:",
        "src/agents/interactive.py:30:    def step(self):",
    ]),
    "returncode": 0,
}

GREP_MANY_MATCHES = {
    "action": "grep -rn 'import' src/",
    "output": "\n".join([f"src/file{i}.py:{i}:import something{i}" for i in range(20)]),
    "returncode": 0,
}

GREP_NO_MATCHES = {
    "action": "grep -rn 'nonexistent_pattern_xyz' src/",
    "output": "",
    "returncode": 1,
}

CD_SUCCESS = {
    "action": "cd /home/user/project/src",
    "output": "",
    "returncode": 0,
}

CD_FAIL = {
    "action": "cd /nonexistent/path",
    "output": "bash: cd: /nonexistent/path: No such file or directory",
    "returncode": 1,
}

CAT_SHORT = {
    "action": "cat pyproject.toml",
    "output": "[build-system]\nrequires = [\"setuptools\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"mini-swe-agent\"\n",
    "returncode": 0,
}

CAT_LONG = {
    "action": "cat src/minisweagent/agents/default.py",
    "output": "\n".join([f"line {i}: some python code here" for i in range(100)]),
    "returncode": 0,
}

NL_SHORT = {
    "action": "nl -ba setup.py",
    "output": "     1\tfrom setuptools import setup\n     2\tsetup()\n",
    "returncode": 0,
}

FIND_FEW = {
    "action": "find src -name '*.py' -type f",
    "output": "src/__init__.py\nsrc/agents/default.py\nsrc/agents/long_context.py\n",
    "returncode": 0,
}

FIND_MANY = {
    "action": "find . -name '*.py'",
    "output": "\n".join([f"./src/module{i}/file{j}.py" for i in range(10) for j in range(5)]),
    "returncode": 0,
}

FIND_EMPTY = {
    "action": "find . -name '*.rs'",
    "output": "",
    "returncode": 0,
}

ECHO_SHORT = {
    "action": "echo 'hello world'",
    "output": "hello world\n",
    "returncode": 0,
}

ECHO_LONG = {
    "action": "echo $PATH",
    "output": "x" * 500,
    "returncode": 0,
}

SED_SUCCESS = {
    "action": "sed -i 's/old_string/new_string/g' src/main.py",
    "output": "",
    "returncode": 0,
}

SED_WITH_OUTPUT = {
    "action": "sed -n '10,20p' src/main.py",
    "output": "\n".join([f"    line {i}" for i in range(10, 21)]),
    "returncode": 0,
}


# ── Tool definitions (matching config/default_with_tools.yaml) ───────────────

TOOLS = {
    "echo": {"name": "echo", "description": "General purpose bash command"},
    "grep": {"name": "grep", "description": "Search for a specific pattern"},
    "ls":   {"name": "ls",   "description": "List files and directories"},
    "cd":   {"name": "cd",   "description": "Change directory"},
    "cat":  {"name": "cat",  "description": "Read the content of the file"},
    "find": {"name": "find", "description": "Directory search"},
    "nl":   {"name": "nl",   "description": "View file with numbered lines"},
    "sed":  {"name": "sed",  "description": "Edit files by replace"},
}


# ── step_summary routing ────────────────────────────────────────────────────

class TestStepSummaryRouting:
    """Verify step_summary routes each tool to the correct summarizer."""

    def test_ls_routing(self):
        result = step_summary(LS_SHORT, TOOLS["ls"])
        assert result.startswith("[ls]")
        assert "CLAUDE.md" in result

    def test_grep_routing(self):
        result = step_summary(GREP_WITH_MATCHES, TOOLS["grep"])
        assert result.startswith("[grep]")
        assert "def step" in result

    def test_cd_routing(self):
        result = step_summary(CD_SUCCESS, TOOLS["cd"])
        assert result.startswith("[cd]")
        assert "Changed to" in result

    def test_cat_routing(self):
        result = step_summary(CAT_SHORT, TOOLS["cat"])
        assert result.startswith("[cat]")
        assert "pyproject.toml" in result

    def test_nl_routing(self):
        result = step_summary(NL_SHORT, TOOLS["nl"])
        assert result.startswith("[nl]")
        assert "setup.py" in result

    def test_find_routing(self):
        result = step_summary(FIND_FEW, TOOLS["find"])
        assert result.startswith("[find]")
        assert "__init__.py" in result

    def test_echo_routing(self):
        """echo → _summarize_default"""
        result = step_summary(ECHO_SHORT, TOOLS["echo"])
        assert result.startswith("[echo]")
        assert "hello world" in result

    def test_sed_routing(self):
        """sed → _summarize_default"""
        result = step_summary(SED_SUCCESS, TOOLS["sed"])
        assert result.startswith("[sed]")


# ── Individual summarizer tests ──────────────────────────────────────────────

class TestSummarizeLs:
    def test_short_listing_shows_all_files(self):
        result = _summarize_ls(LS_SHORT)
        assert "CLAUDE.md" in result
        assert "pyproject.toml" in result
        assert "tests" in result

    def test_long_listing_truncates_at_30(self):
        result = _summarize_ls(LS_WITH_DIR)
        assert "file_0.py" in result
        assert "file_29.py" in result
        assert "file_30.py" not in result
        assert "showing first 30" in result
        assert "50 entries" in result

    def test_extracts_directory(self):
        result = _summarize_ls(LS_WITH_DIR)
        assert "src/minisweagent/agents" in result

    def test_empty_ls(self):
        result = _summarize_ls(LS_EMPTY)
        assert "/empty" in result


class TestSummarizeGrep:
    def test_shows_pattern_and_matches(self):
        result = _summarize_grep(GREP_WITH_MATCHES)
        assert "def step" in result
        assert "3 matches" in result

    def test_shows_preview_lines(self):
        result = _summarize_grep(GREP_WITH_MATCHES)
        assert "default.py" in result
        assert "long_context.py" in result

    def test_no_matches(self):
        result = _summarize_grep(GREP_NO_MATCHES)
        assert "No matches" in result
        assert "nonexistent_pattern_xyz" in result

    def test_many_matches_truncates_at_5(self):
        result = _summarize_grep(GREP_MANY_MATCHES)
        assert "20 matches" in result
        assert "import something0" in result
        assert "import something4" in result
        assert "15 more matches" in result


class TestSummarizeCd:
    def test_success(self):
        result = _summarize_cd(CD_SUCCESS)
        assert "Changed to /home/user/project/src" in result

    def test_failure(self):
        result = _summarize_cd(CD_FAIL)
        assert "Failed" in result
        assert "/nonexistent/path" in result


class TestSummarizeReadFile:
    def test_short_file_shows_content(self):
        result = _summarize_read_file(CAT_SHORT)
        assert "pyproject.toml" in result
        assert "[build-system]" in result
        assert "setuptools" in result

    def test_long_file_shows_preview(self):
        result = _summarize_read_file(CAT_LONG)
        assert "100 lines" in result
        assert "line 0" in result
        assert "line 9" in result
        assert "showing first 10" in result

    def test_nl_file(self):
        result = _summarize_read_file(NL_SHORT)
        assert "setup.py" in result
        assert "setuptools" in result


class TestSummarizeFind:
    def test_few_files(self):
        result = _summarize_find(FIND_FEW)
        assert "__init__.py" in result
        assert "long_context.py" in result

    def test_many_files_truncates_at_20(self):
        result = _summarize_find(FIND_MANY)
        assert "50 files found" in result
        assert "showing first 20" in result

    def test_no_files(self):
        result = _summarize_find(FIND_EMPTY)
        assert "No files found" in result


class TestSummarizeDefault:
    def test_short_output_preserved(self):
        result = _summarize_default(ECHO_SHORT)
        assert "hello world" in result

    def test_long_output_shows_head_and_tail(self):
        result = _summarize_default(ECHO_LONG)
        assert len(result) < 500
        assert "truncated" in result

    def test_empty_output(self):
        result = _summarize_default(SED_SUCCESS)
        assert result == ""


# ── Summary usability: content is enough to understand what happened ─────────

class TestSummaryUsability:
    """Verify summaries contain enough info for LLM to make decisions."""

    @pytest.mark.parametrize("obs, tool_name, must_contain", [
        (LS_SHORT, "ls", ["CLAUDE.md", "pyproject.toml"]),
        (GREP_WITH_MATCHES, "grep", ["def step", "3 matches"]),
        (GREP_NO_MATCHES, "grep", ["No matches"]),
        (CD_SUCCESS, "cd", ["/home/user/project/src"]),
        (CAT_SHORT, "cat", ["build-system", "setuptools"]),
        (CAT_LONG, "cat", ["100 lines", "line 0"]),
        (FIND_FEW, "find", ["__init__.py", "long_context.py"]),
        (FIND_MANY, "find", ["50 files"]),
        (ECHO_SHORT, "echo", ["hello world"]),
        (SED_WITH_OUTPUT, "sed", ["line 10"]),
    ])
    def test_summary_contains_key_info(self, obs, tool_name, must_contain):
        result = step_summary(obs, TOOLS[tool_name])
        for expected in must_contain:
            assert expected in result, f"Summary for {tool_name} missing '{expected}':\n{result}"

    @pytest.mark.parametrize("obs, tool_name", [
        (LS_SHORT, "ls"),
        (GREP_WITH_MATCHES, "grep"),
        (CD_SUCCESS, "cd"),
        (CAT_SHORT, "cat"),
        (FIND_FEW, "find"),
        (ECHO_SHORT, "echo"),
        (SED_SUCCESS, "sed"),
        (NL_SHORT, "nl"),
    ])
    def test_summary_has_tool_name_and_returncode(self, obs, tool_name):
        result = step_summary(obs, TOOLS[tool_name])
        assert f"[{tool_name}]" in result
        assert f"rc={obs['returncode']}" in result
