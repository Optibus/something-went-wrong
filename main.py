from flask import Flask, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

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
