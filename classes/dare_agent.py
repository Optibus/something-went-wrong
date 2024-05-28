from typing import List
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage

from classes.agent import Agent


class DareAgent(Agent):
    def __init__(self, name: str, llm: ChatAnthropic, vision: str, mission: str):
        self.build_graph()
        self.vision = vision
        self.mission = mission
        super().__init__(name, llm, [], "", self.runnable)

    def invoke_model(self, state: List[BaseMessage]):
        res = self.llm_with_tools.invoke(state)
        return res

    def build_graph(self):
        graph = MessageGraph()
        graph.add_node("invoke_model", self.invoke_model)
        graph.add_edge("invoke_model", END)
        graph.set_entry_point("invoke_model")
        self.runnable = graph.compile()

    def invoke(self, input: str, context: str):
        try:
            # Query from the vector database
            print(system_prompt)
            res = self.runnable.invoke(
                [
                    SystemMessage(system_prompt),
                    HumanMessage(input),
                ],
                {"callbacks": [self.langfuse]},
            )
            return res[-1].content
        except Exception as e:
            print(e)
            return "Something went wrong. Please try again."
