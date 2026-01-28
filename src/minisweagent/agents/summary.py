"""Agent that maintains a condensed summary history alongside the full message log."""

import time

from minisweagent.agents.default import DefaultAgent

_MAX_OBSERVATION_CHARS = 200


class SummaryAgent(DefaultAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_messages: list[dict] = []

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        self.summary_messages = []
        return super().run(task, **kwargs)

    def add_message(self, role: str, content: str, **kwargs):
        super().add_message(role, content, **kwargs)
        summary_content = content
        # Truncate observation messages (user messages after system + instance)
        if role == "user" and len(self.messages) > 2:
            summary_content = content[:_MAX_OBSERVATION_CHARS]
            if len(content) > _MAX_OBSERVATION_CHARS:
                summary_content += f"\n... [{len(content) - _MAX_OBSERVATION_CHARS} chars truncated]"
        self.summary_messages.append({"role": role, "content": summary_content, "timestamp": time.time()})
