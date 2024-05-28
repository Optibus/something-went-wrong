import json
from flask import Flask, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import requests
from requests.auth import HTTPBasicAuth
from classes.dare_agent import DareAgent
from bs4 import BeautifulSoup


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


@app.route("/dataset")
def dataset():
    # Replace these with your own values
    username = "sami.kadry@optibus.com"
    api_token = (
        "ATATT3xFfGF0rOh8mvqroz5VKONxd02fx2xz8LW2xnKHTdZsphtVDjJ-XDi82mkavc5zUasFP-I99zL37drdDJwwn3cq1YW8TOEPCB5AhI6tcv"
        "_q1T8dc4NwFBsVub_FwMSMhsmE_il_3giAZ_bmON7a5gmqDgJIoVKJCcSWi6qQfIam7alfqVE=A3FF4A23"
    )
    base_url = "https://optibus.atlassian.net"
    page_id = "3904602187"  # The ID of the page you want to fetch

    url = f"{base_url}/wiki/rest/api/content/{page_id}"

    # Make the request
    response = requests.get(
        url,
        auth=HTTPBasicAuth(username, api_token),
        params={"expand": "body.storage"}
    )

    # Check for successful response
    if response.status_code == 200:
        page_content = response.json()
        html_content = page_content["body"]["storage"]["value"]
    else:
        raise Exception(f"Failed to fetch page content: {response.status_code}")

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the table in the HTML
    table = soup.find("table")

    # Extract table rows
    headers = None
    rows = []
    for idx, tr in enumerate(table.find_all("tr")):
        cells = tr.find_all("td")
        if idx == 0:
            headers = [cell.text.strip() for cell in cells]
        else:
            row = [cell.text.strip() for cell in cells]
            rows.append(row)

    # Convert to list of dictionaries
    table_objects = []
    for row in rows:
        row_dict = {headers[idx]: cell for idx, cell in enumerate(row)}
        table_objects.append(row_dict)

    with open("./resources/data.json", "w") as json_file:
        json.dump(table_objects, json_file, indent=4)

    return table_objects
