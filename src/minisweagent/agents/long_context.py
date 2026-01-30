"""LongContextAgent with meaningful summary_messages."""

import time

from minisweagent import ToolDescription
from minisweagent.agents.default import (
    AgentConfig,
    DefaultAgent,
    FormatError,
    NonTerminatingException,
    TerminatingException,
)
from minisweagent.agents.history_retrieval import HistoryRetriever
from minisweagent.agents.tool_match import ToolMatch


class AgentLongContextConfig(AgentConfig):
    system_template_abstract: str


# Tool-specific summarizers
def _summarize_grep(obs: dict) -> str:
    action = obs.get("action", "")
    # Extract pattern from grep command - look for quoted string or first non-flag arg after grep
    pattern = ""
    import re
    # Try to find quoted pattern first
    quoted = re.search(r"['\"]([^'\"]+)['\"]", action)
    if quoted:
        pattern = quoted.group(1)
    lines = obs.get("output", "").strip().splitlines()
    if not lines:
        return f"No matches for '{pattern}'" if pattern else "No matches"
    preview = "\n".join(lines[:3])
    suffix = "..." if len(lines) > 3 else ""
    return f"{len(lines)} matches for '{pattern}':\n{preview}{suffix}" if pattern else f"{len(lines)} matches:\n{preview}{suffix}"


def _summarize_ls(obs: dict) -> str:
    action = obs.get("action", "")
    # Extract directory from ls command (last non-flag argument, or "." if none)
    parts = action.split()
    directory = "."
    for p in reversed(parts):
        if not p.startswith("-") and p != "ls":
            directory = p
            break
    lines = obs.get("output", "").strip().splitlines()
    return f"{directory}: {len(lines)} entries"


def _summarize_read_file(obs: dict) -> str:
    lines = obs.get("output", "").splitlines()
    action = obs.get("action", "")
    # Extract filename from command like "cat file.py" or "head -n 10 file.py"
    parts = action.split()
    filename = parts[-1] if parts else "file"
    return f"Read {filename}: {len(lines)} lines"


def _summarize_cd(obs: dict) -> str:
    action = obs.get("action", "")
    # Extract directory from "cd /path/to/dir"
    parts = action.split()
    directory = parts[-1] if len(parts) > 1 else "?"
    if obs.get("returncode") == 0:
        return f"Changed to {directory}"
    return f"Failed to change to {directory}"


def _summarize_execute_command(obs: dict) -> str:
    output = obs.get("output", "")
    if len(output) <= 100:
        return output
    return output[:100] + f"... [{len(output) - 100} chars]"


def _summarize_verify(obs: dict) -> str:
    rc = obs.get("returncode", 1)
    return "PASS" if rc == 0 else f"FAIL (rc={rc})"


def _summarize_default(obs: dict) -> str:
    output = obs.get("output", "")
    if len(output) <= 150:
        return output
    return output[:150] + f"... [{len(output) - 150} chars]"


_TOOL_SUMMARIZERS = {
    "grep": _summarize_grep,
    "ls": _summarize_ls,
    "read_file": _summarize_read_file,
    "cd": _summarize_cd,
    "execute_command": _summarize_execute_command,
    "verify": _summarize_verify,
}


class LongContextAgent(DefaultAgent, ToolMatch, HistoryRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, config_class=AgentLongContextConfig, **kwargs)
        self.summary_messages: list[dict] = []

    def add_summary_message(self, role: str, content: str, **kwargs):
        self.summary_messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run agent with separate summary_messages using compact system prompt."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.summary_messages = []

        # Full prompts for messages
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        # Compact system prompt for summary_messages
        self.add_summary_message("system", self.render_template(self.config.system_template_abstract))
        self.add_summary_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def update_history_summary(self, next_step_thinking: str, tool: ToolDescription, observation: dict) -> None:
        """Add meaningful summary: full reasoning + tool-specific observation summary."""
        # Assistant message: full reasoning + command
        action = observation.get("action", "")
        self.add_summary_message(role="assistant", content=f"{next_step_thinking}\n\n```bash\n{action}\n```")

        # User message: tool-specific summary
        tool_name = tool.get("name", "")
        summarizer = _TOOL_SUMMARIZERS.get(tool_name, _summarize_default)
        summary = summarizer(observation)
        returncode = observation.get("returncode", 0)
        self.add_summary_message(role="user", content=f"[{tool_name}] rc={returncode}\n{summary}")

    def step(self) -> dict:
        next_step_response = self.model.query(self.summary_messages)
        match next_step_response:
            case {"content": next_step_thinking}:
                tool: ToolDescription = self.tool_match(self.config.tools, next_step_thinking, self.model)
                only_relevant_messages = self.retrieve(
                    history=self.messages,
                    query=f'{next_step_thinking} {tool.get("description")}',
                    model=self.model
                )
                only_relevant_messages.append({"role": "user", "content": next_step_thinking})
                response = self.query(allowed_tools=[tool], messages=only_relevant_messages)
                observation = self.get_observation(response)
                self.update_history_summary(
                    next_step_thinking=next_step_thinking,
                    tool=tool,
                    observation=observation
                )
                return observation
            case _:
                raise FormatError("LLM response has no content")
