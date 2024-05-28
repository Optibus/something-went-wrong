from typing import List
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage, BaseMessage


from classes.agent import Agent
from tools.python_utils import migrate_to_py3


class Python2To3Converter(Agent):
    def __init__(self, name: str, llm: ChatAnthropic, system_prompt: str):
        self.build_graph()
        super().__init__(name, llm, [migrate_to_py3], system_prompt, self.runnable)

    def invoke_model(self, state: List[BaseMessage]):
        return self.llm_with_tools.invoke(state)

    def invoke_tool(self, state: List[BaseMessage]):
        tool = state[-1].content[-1]
        is_tool_use = tool["type"] == "tool_use"

        if is_tool_use:
            if tool["name"] == "migrate_to_py3":
                res = migrate_to_py3.invoke(tool["input"])
                return HumanMessage(
                    [
                        {
                            "type": "tool_result",
                            "content": res,
                            "tool_use_id": tool["id"],
                        }
                    ]
                )
        raise Exception("No adder input found.")

    def router(self, state: List[BaseMessage]):
        is_tool_use = (
            isinstance(state[-1].content, List)
            and state[-1].content[-1]["type"] == "tool_use"
        )
        if is_tool_use:
            return state[-1].content[-1]["name"]
        else:
            return "end"

    def build_graph(self):
        graph = MessageGraph()
        graph.add_node("invoke_model", self.invoke_model)
        graph.add_node("migrate_to_py3", self.invoke_tool)

        graph.add_edge("migrate_to_py3", END)

        graph.set_entry_point("invoke_model")
        graph.add_conditional_edges(
            "invoke_model",
            self.router,
            {
                "migrate_to_py3": "migrate_to_py3",
                "end": END,
            },
        )
        self.runnable = graph.compile()
