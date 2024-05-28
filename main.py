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

from langchain_core.runnables import Runnable, RunnablePassthrough

from classes.dare_agent import DareAgent


load_dotenv()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0,
    model="claude-3-opus-20240229",
    max_tokens=4096,
)

llm_haiku = ChatAnthropic(
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0,
    model="claude-3-haiku-20240307",
    max_tokens=4096,
)

agent = DareAgent(
    name="Agent - Output Agent",
    llm=llm_haiku,
    vision="To provide accurate and helpful information to the user, while ensuring that the information is relevant to the user's query",
    mission="To provide a solution to the user's problem",
)

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route("/ping")
def ping():
    return "pong"


@app.route("/invoke", methods=["POST"])
def invoke():
    data = request.json
    input = data["input"]
    res = agent.invoke(input=input, context="")
    return res
