from flask import Flask, request
from flask_cors import CORS
import os
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
from langchain_core.tools import tool
import requests
import base64
from langchain_anthropic import ChatAnthropic
from typing import List
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph import END, MessageGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import json
from json import JSONDecodeError
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException


from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)


from langchain_core.outputs import ChatGeneration, Generation

from langchain.agents.agent import AgentOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from classes.basic_prompt_agent import BasicAgent
from classes.python_converter_agent import Python2To3Converter
from classes.supervisor_agent import SuperVisorAgent


load_dotenv()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Github
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")

# Langfure
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

langfuse_handler = None
llm = None
llm_with_tools = None
runnable = None
graph = MessageGraph()

app = Flask(__name__)
CORS(app, support_credentials=True)

if __name__ == "__main__":
    llm = ChatAnthropic(
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0,
        # model="claude-3-haiku-20240307",
        model="claude-3-opus-20240229",
        max_tokens=4096,
    )
    llm_haiku = ChatAnthropic(
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0,
        model="claude-3-haiku-20240307",
        max_tokens=4096,
    )
    code = (
        "break_plus_taxis_and_split_taxis = get_break_plus_taxis_and_split_taxis(duty)\n"
        "max_unpaid_time = safeget(max_unpaid_time_dic, duty.pref_group.id)\n"
        "unpaid_break_time =  min(break_plus_taxis_and_split_taxis, max_unpaid_time) if max_unpaid_time else break_plus_taxis_and_split_taxis\n"
        "unpaid_events = get_unpaid_events_time(duty)\n"
        "unpaid_time = unpaid_break_time + unpaid_events\n"
        "guarantee_time = safeget(guarantee_dic, duty.pref_group.id)\n"
        "paid_time = max(original_paid_time_func.calculate_time(duty) - unpaid_time, guarantee_time)\n"
        "return round_paid_time(paid_time)\n"
    )
    content = (
        "Given the following code snippet, migrate it from Python 2 to Python 3\n"
        "```python\n"
        f"{code}\n"
        "```\n"
        "If the code is already valid Python 3 code, do not migrate it.\n"
        "After migrating the code, analyze it for potential issues that might work in Python 2 and not in Python 3.\n"
        "Make sure to update the code with the help of the analyze details.\n"
        "Finally, generate unit tests for the migrated code.\n"
        "Make sure to provide back the final code and the generated unit tests.\n"
        "Also make sure to provide the code to every agent you use\n"
        "Don't use \{ or \} in the input that you give to the agents\n"
    )

    # Agent Creation
    python_agent = Python2To3Converter(
        "Agent - Python2 To Python3",
        llm_haiku,
        "You are an expert in migrating code from python2 to python3. You will be given a code snippet to migrate.\nalways try to migrate the code even though it's already in python3",
    )
    python_agent.export_graph("code-agent.png")

    code_analyzer_agent = BasicAgent(
        "Agent - Python Code Analyzer",
        llm_haiku,
        "You are an expert in analyzing migrated python2 code to python3 and founding all the possible edge cases that might occur.\nalways try to analyze the code and try to figure out what the inner functions might be doing and what issues can be there.\nExample of issues that might occur: exception message: '>' not supported between instances of 'NoneType' and 'int' in python2 this might work but in python3 it will throw an exception.",
    )
    code_analyzer_agent.export_graph("code-analyzer-agent.png")

    test_generator_agent = BasicAgent(
        "Agent - Python Unit Test Generator",
        llm_haiku,
        "You are an expert in generating unit tests for the migrated python3 code.\nalways try to generate the unit tests for the provided code.",
    )
    test_generator_agent.export_graph("test-generator-agent.png")

    # members = ["Code Analyzer", "Python2 to Python3 Converter", "Code Updater"]
    agents = {}

    agents[python_agent.get_name()] = {
        "agent": python_agent,
        "description": (
            "This agent is an expert in migrating code from python2 to python3. it only knows to migrate code and always require sending him the code in the following format:\n"
            "```code\n"
            "   the code\n"
            "```"
        ),
    }
    agents[code_analyzer_agent.get_name()] = {
        "agent": code_analyzer_agent,
        "description": (
            "This agent is an expert in analyzing migrated python2 code to python3 and founding all the possible issues.\n"
            "it only knows to analyze the code and always require sending him the code in the following format:\n"
            "```code\n"
            "   the code\n"
            "```"
        ),
    }
    agents[test_generator_agent.get_name()] = {
        "agent": test_generator_agent,
        "description": (
            "This agent is an expert in generating unit tests for the migrated python3 code.\n"
            "it only knows to generate the unit tests and always require sending him the code in the following format:\n"
            "```code\n"
            "   the code\n"
            "```"
        ),
    }
    # agents["END"] = END

    supervisor_agent = SuperVisorAgent("Agent - Supervisor", llm_haiku, agents)
    supervisor_agent.export_graph("supervisor-agent.png")

    res = supervisor_agent.invoke(content)

    print("FINAL FINAL OUTPUT")
    print("====================================")
    print(res)
