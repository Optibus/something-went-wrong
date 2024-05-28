from typing import List
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage

from classes.agent import Agent


class BasicAgent(Agent):
    def __init__(self, name: str, llm: ChatAnthropic, system_prompt: str):
        self.build_graph()
        super().__init__(name, llm, [], system_prompt, self.runnable)

    def invoke_model(self, state: List[BaseMessage]):
        res = self.llm_with_tools.invoke(state)
        return res

    def build_graph(self):
        graph = MessageGraph()
        graph.add_node("invoke_model", self.invoke_model)
        graph.add_edge("invoke_model", END)
        graph.set_entry_point("invoke_model")
        self.runnable = graph.compile()

    def invoke(self, input: str):
        try:
            res = self.runnable.invoke(
                [
                    SystemMessage(self.system_prompt),
                    HumanMessage(input),
                ],
                {"callbacks": [self.langfuse]},
            )
            return res[-1].content
        except Exception as e:
            return "Something went wrong. Please try again."
