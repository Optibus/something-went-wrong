from typing import Dict, List
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
import xml.etree.ElementTree as ET


from classes.agent import Agent
from classes.basic_prompt_agent import BasicAgent


class SuperVisorAgent(Agent):
    def __init__(
        self, name: str, llm: ChatAnthropic, members: Dict[str, Dict[str, any]]
    ):
        self.name = name
        self.members = members
        self.llm = llm
        self.build_graph(members)
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {members.keys()}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH."
            "\n\n"
            "Agent descriptions:\n"
        )
        for [member, member_node] in members.items():
            system_prompt += f"{member}: {member_node['description']}\n"
        super().__init__(name, llm, [], system_prompt, self.runnable)

    def invoke_model(self, state: List[BaseMessage]):
        options = [k for k in self.members.keys()]
        state.append(
            HumanMessage(
                (
                    "Given the conversation above, who should act next?"
                    f" Or should we FINISH? Select one of: {options}"
                    "If there is no worker needed to act next, please respond with FINISH"
                    "\nyour message should be in the following format and exacly this format don't add any extra text or whitespaces:\n"
                    "<root>\n"
                    "   <agent>worker_name</agent>\n"
                    "   <message>\nThe message to give to the agent\n"
                    "       {Any additional details}\n  </message>\n</root>\n\n"
                    "Make sure that the message describes exacly what the worker should do and specific for his task only\n"
                    "Keep in mind that the Agent doesn't have history of the chat so you will need to provide him with all the details that he needs to complete the task\n"
                    "In the extra details you can add any additional information that the worker might need to know for example the code that he might need to migrate/analyze\n"
                    "If you choose FINISH as the next worker, you will need to provide details of the task that you are trying to do, and make sure to include all the agent calls that happend and each agent response!\n"
                    "Make sure to provide the output agent with each agent task and their response!\n"
                    "Also in case you choose finish use the following format\n"
                    "<root>\n"
                    "   <agent>FINISH</agent>\n"
                    "   <message>\n"
                    "       the requested task details\n\n"
                    "       first agent name:\n"
                    "       first agent response\n"
                    "       second agent name\n"
                    "       second agent response\n"
                    "   </message>\n"
                    "</root>\n"
                )
            )
        )
        return self.llm_with_tools.invoke(state)

    def extract_worker_name(self, text):
        root = ET.fromstring(text)

        message_element = root.find("agent")

        if message_element is not None:
            return message_element.text.strip()

        return "END"

    def extract_worker_input(self, text):
        root = ET.fromstring(text)

        message_element = root.find("message")

        if message_element is not None:
            print("EXTRACTING WORKER INPUT FROM", message_element.text.strip())
            return message_element.text.strip()

        return "END"

    def extract_final_output(self, text):
        print("EXTRACTING FINAL OUTPUT FROM", text)
        root = ET.fromstring(text)

        message_element = root.find("response")

        if message_element is not None:
            return message_element.text.strip()

        return "No reply"

    def router(self, state: List[BaseMessage]):
        worker_name = self.extract_worker_name(state[-1].content)
        return worker_name

    def get_worker_response_content(self, message):
        print("AGENT RESPONSE", message)
        if type(message) == str:
            return message
        if type(message[-1]) == str:
            return message[-1]

        if len(message[-1].content) > 0:
            if type(message[-1].content[-1]) == str:
                return message[-1].content[-1]

        return message[-1].content[-1]["content"]

    def create_worker(self, agent: Agent):
        return lambda state: HumanMessage(
            f"<root>{agent.get_name()} <response>{self.get_worker_response_content(agent.invoke(self.extract_worker_input(state[-1].content)))}</response></root>"
        )

    def create_ouput_agent(self):
        return BasicAgent(
            "Agent - Output Agent",
            self.llm,
            (
                "You will be providing a summary of the output of the code migration and analysis.\n"
                "Make sure to provide a detailed summary of the output\n"
                "The output should be written specifically for everyone to understand\n"
                "Write it in a formal way and not in a casual way like a chat\n"
                "Make sure to not miss any details from any of the agents\n"
                "If the output contains code or tests make sure to include all of them in the output\n"
                "Make sure that the code and tests output is written fully in your response"
            ),
        )

    def build_graph(self, members: Dict[str, Dict[str, any]]):
        graph = MessageGraph()

        graph.add_node(self.name, self.invoke_model)
        # graph.add_node("ouput_agent", self.create_worker(self.create_ouput_agent()))
        # graph.add_edge("ouput_agent", END)

        for [member, member_node] in members.items():
            print("ADDING MEMBER", member, member_node)
            graph.add_node(member, self.create_worker(member_node["agent"]))
            # We want our workers to ALWAYS "report back" to the supervisor when done
            graph.add_edge(member, self.name)

        conditionalMap = {k: k for k in members.keys()}
        conditionalMap["FINISH"] = END
        conditionalMap["END"] = END

        graph.add_conditional_edges(
            self.name,
            self.router,
            conditionalMap,
        )

        graph.set_entry_point(self.name)

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
            return self.extract_final_output(res[-1].content)
        except Exception as e:
            print(e)
            return "Something went wrong. Please try again."
