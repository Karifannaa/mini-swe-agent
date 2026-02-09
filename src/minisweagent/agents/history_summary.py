import re
from minisweagent import ToolDescription


def _summarize_grep(obs: dict) -> str:
    action = obs.get("action", "")
    pattern = ""
    # Try to find quoted pattern first
    quoted = re.search(r"['\"]([^'\"]+)['\"]", action)
    if quoted:
        pattern = quoted.group(1)
    lines = obs.get("output", "").strip().splitlines()
    if not lines:
        return f"No matches for '{pattern}'" if pattern else "No matches"
    preview = "\n".join(lines[:5])
    suffix = f"\n[{len(lines) - 5} more matches]" if len(lines) > 5 else ""
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
    if len(lines) <= 30:
        return f"{directory}:\n" + "\n".join(lines)
    preview = "\n".join(lines[:30])
    return f"{directory} ({len(lines)} entries, showing first 30):\n{preview}"


def _summarize_read_file(obs: dict) -> str:
    output = obs.get("output", "")
    lines = output.splitlines()
    action = obs.get("action", "")
    parts = action.split()
    filename = parts[-1] if parts else "file"
    if len(lines) <= 20:
        return f"Read {filename} ({len(lines)} lines):\n{output}"
    preview = "\n".join(lines[:10])
    return f"Read {filename} ({len(lines)} lines, showing first 10):\n{preview}"


def _summarize_cd(obs: dict) -> str:
    action = obs.get("action", "")
    # Extract directory from "cd /path/to/dir"
    parts = action.split()
    directory = parts[-1] if len(parts) > 1 else "?"
    if obs.get("returncode") == 0:
        return f"Changed to {directory}"
    return f"Failed to change to {directory}"



def _summarize_find(obs: dict) -> str:
    lines = obs.get("output", "").strip().splitlines()
    if not lines:
        return "No files found"
    if len(lines) <= 20:
        return "\n".join(lines)
    preview = "\n".join(lines[:20])
    return f"{len(lines)} files found (showing first 20):\n{preview}"


def _summarize_default(obs: dict) -> str:
    output = obs.get("output", "")
    if len(output) <= 300:
        return output
    return output[:200] + f"\n... [{len(output) - 300} chars truncated] ...\n" + output[-100:]


def step_summary(observation: dict, tool: ToolDescription) -> str:
    """ tool-specific summarization """

    returncode = observation.get("returncode", 0)
    tool_name = tool.get("name")
    summary: str | None = None
    match tool_name:
        case name if name.strip().lower().startswith("ls"):
            summary = _summarize_ls(observation)
        case name if name.strip().lower().startswith("cd"):
            summary = _summarize_cd(observation)
        case name if name.strip().lower().startswith("grep"):
            summary = _summarize_grep(observation)
        case name if name.strip().lower() in ("cat", "nl"):
            summary = _summarize_read_file(observation)
        case name if name.strip().lower().startswith("find"):
            summary = _summarize_find(observation)
        case _:
            summary = _summarize_default(observation)

    return f"[{tool_name}] rc={returncode}\n{summary}"