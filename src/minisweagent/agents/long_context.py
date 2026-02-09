"""agent class with long context support."""

import re
import time

from minisweagent import AbstractMessage
from minisweagent.agents.default import AgentConfig, DefaultAgent, FormatError, NonTerminatingException, ToolDescription
from minisweagent.agents.history_retrieval import HistoryRetriever
from minisweagent.agents.history_summary import step_summary
from minisweagent.agents.tool_match import ToolMatch
from minisweagent.utils.log import logger


class AgentLongContextConfig(AgentConfig):
    system_template_abstract: str
    instance_template_abstract: str
    tool_template: str
    next_direction_template: str


class LongContextAgent(DefaultAgent, ToolMatch, HistoryRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_messages: list[AbstractMessage] = []

    def add_summary_message(self, role: str, content: str, original_message: None | dict=None, **kwargs):
        self.summary_messages.append(
            AbstractMessage(
                role=role,
                content=content,
                original_message=original_message,
                **(kwargs | {"timestamp": time.time()})
            )
        )
        # self.summary_messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def update_history_summary(self, next_step_thinking: str, tool: ToolDescription, observation: dict) -> None:
        action = observation.get("action", "")
        self.add_summary_message(
            role="assistant",
            content=f"{next_step_thinking}\n\n```bash\n{action}\n```",
            original_message=self.messages[-2],
        )
        self.add_summary_message(
            role="user",
            content=step_summary(observation, tool),
            original_message=self.messages[-1],
            tool=tool,
        )


    def run(self, task: str, **kwargs) -> tuple[str, str]:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.summary_messages = []
        # self.add_summary_message("system", self.render_template(self.config.system_template_abstract))
        # self.add_summary_message("user", self.render_template(self.config.instance_template_abstract, tools=self.config.tools))
        return super().run(task, **kwargs)


    def parse_direction(self, response: dict, regexp: str, allowed_tools: list[ToolDescription] | None = None) -> str:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(regexp, response["content"], re.DOTALL)
        if len(actions) == 1:
            if allowed_tools and actions[0].strip().split()[0] not in [tool['name'] for tool in allowed_tools]:
                raise FormatError("I can't execute this command because it is not from the allowed list.")
            return actions[0].strip()
        raise FormatError("I can not parse your response. Please follow the response format instruction.")


    def predict_direction(self) -> str:
        """ using abstract history predicts next direction (tool name) and provide text explaining the choice """

        user_content = ""
        messages = [
            {"role": "system", "content": self.render_template(self.config.system_template_abstract)},
            # {"role": "user",   "content": self.render_template("The task is: {{task}}")},
        ]
        user_content += self.render_template("The task is: {{task}}")

        user_content += self.render_template(self.config.next_direction_template, tools=self.config.tools)
        # messages += [
        #     {"role": "user", "content": self.render_template(self.config.next_direction_template, tools=self.config.tools)}
        # ]

        if len(self.summary_messages) == 0:
            user_content += "It's the project beginning. Delegate the first task."
            # messages.append({"role": "user", "content": "It's the project beginning and you have to make the first step."})
        else:
            user_content += "The project history report:"
            # messages.append({"role": "user",   "content": "The project history report:"})
            for id, msg in enumerate(self.summary_messages, start=1):
                user_content += f'{id}) {msg.content}\n'
                # messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user",   "content": user_content})
        logger.debug("Payload:\n %s", messages)
        try:
            response = self.model.query(messages=messages)
            logger.debug("Direction Response:\n %s", response['content'])
            # tool_name = self.parse_direction(response, "<tool>\\s*\n(.*?)\\s*</tool>", self.config.tools)
            # print("TOOL NAME: ", tool_name)
            # tool = [tool for tool in self.config.tools if tool['name'] == tool_name][0]
            explanation = self.parse_direction(response, "<thinking>\\s*(.*?)\\s*</thinking>")
        except NonTerminatingException as e:
            raise
        return explanation

    def get_relevant_context(self, explanation: str) -> list[dict]:
        """ Return historically ordered list of messages relevant to explanation"""
        return [
            msg.original_message for msg in self.retrieve(self.summary_messages[1: ], explanation, None)
            if msg.original_message is not None
        ]

    def predict_action(self, context: list[dict], explantion: str) -> dict:
        """
        Predict next action given context and selected tool (direction) for this action

        :param messages: Message history excluding system prompt and task description
        :type messages: list[dict]
        :param explantion: Text explanation we d
        :type explantion: str
        """
        messages = [
            {"role": "system", "content": self.render_template(self.config.system_template)},
            {"role": "user",   "content": self.render_template(self.config.instance_template)}
        ]
        messages += context
        messages += [{
            "role": "user",
            "content": self.render_template(
                template=self.config.tool_template,
                tools=self.config.tools,
                explantion=explantion
            )}]

        response = self.query(messages=messages)
        return self.parse_action(response=response)


    def step(self) -> dict:
        logger.info("STEP")
        explanation = self.predict_direction()
        logger.info("NEXT STEP EXPLANATION: %s", explanation)
        messages = self.get_relevant_context(explanation=explanation)
        action: dict = self.predict_action(messages, explanation)
        tool = [tool for tool in self.config.tools if action['action'].startswith(tool['name'])][0]
        observation = self.execute_action(action=action)
        rendered_obs = self.render_template(self.config.action_observation_template, output=observation)
        self.add_message("user", rendered_obs)
        self.update_history_summary(
            next_step_thinking=explanation,
            tool=tool,
            observation=observation
        )
        return observation




