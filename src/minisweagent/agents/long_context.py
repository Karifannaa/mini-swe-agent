"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import time

from minisweagent.agents.default import AgentConfig, DefaultAgent, FormatError, ToolDescription
from minisweagent.agents.history_retrieval import HistoryRetriever
from minisweagent.agents.tool_match import ToolMatch


class AgentLongContextConfig(AgentConfig):
    system_template_abstract: str


class LongContextAgent(DefaultAgent, ToolMatch, HistoryRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_messages: list[dict] = []

    def add_summary_message(self, role: str, content: str, **kwargs):
        self.summary_messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def update_history_summary(self, next_step_thinking: str, selected_tool_description: str, observation: dict) -> None:
        _MAX_OBSERVATION_CHARS = 200
        content = observation["content"]
        summary_content = content[: _MAX_OBSERVATION_CHARS]
        if len(content) > _MAX_OBSERVATION_CHARS:
            summary_content += f"\n... [{len(content) - _MAX_OBSERVATION_CHARS} chars truncated]"
        self.add_summary_message(role="user", content=summary_content, timestamp=time.time())

    def step(self) -> dict:
        next_step_response = self.model.query(self.summary_messages)
        match next_step_response:
            case {"content": next_step_thinking, **_}:
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
                    selected_tool_description=tool.get("description"),
                    observation=observation
                )
                return observation
            case _:
                raise FormatError("LLM response has no content")

        raise ValueError("Error")



