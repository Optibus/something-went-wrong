from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from tools.langfuse import create_langfuse


class Agent:
    name: str
    llm: ChatAnthropic
    tools: list
    system_prompt: str
    llm_with_tools: ChatAnthropic
    runnable: any

    def __init__(
        self,
        name: str,
        llm: ChatAnthropic,
        tools: list,
        system_prompt: str,
        runnable: any,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.runnable = runnable
        self.langfuse = create_langfuse(self.name)

    def get_name(self):
        return self.name

    def get_llm(self):
        return self.llm_with_tools

    def get_tools(self):
        return self.tools

    def export_graph(self, file_path: str):
        self.runnable.get_graph().draw_png(file_path)

    def invoke(self, input: str):
        return self.runnable.invoke(
            [
                SystemMessage(self.system_prompt),
                HumanMessage(input),
            ],
            {"callbacks": [self.langfuse]},
        )
