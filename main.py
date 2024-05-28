import json
from flask import Flask, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import requests
from requests.auth import HTTPBasicAuth
from classes.dare_agent import DareAgent
from classes.basic_prompt_agent import BasicAgent
from bs4 import BeautifulSoup

from db import get_context_from_db

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

classification_agent = BasicAgent(
    name="Agent - Classification Agent",
    llm=llm,
    system_prompt=(
        "Persona: You are a Support Engineer for Optibus Company, your goal is to suggest solutions to errors that comes out of the platform with as much details as possible especially names that can quide the user to find the issue.\n\n"
        "You will be classifying error messages into categories to help guide users on how to fix issues they encounter.\n\n"
        "The possible categories to classify the error into are:\n"
        "- Data input\n"
        "- Configuration\n"
        "- Connection\n"
        "- User action is not allowed\n"
        "- System error\n\n"
        "First, carefully analyze the error message and think through what category it best fits into. Write out your reasoning inside <reasoning> tags.\n"
        "Then, output the final category you selected inside <category> tags.\n"
        "Include the message that should be returned to the user inside <user_output> tags.\n"
        "Your response should be structured as a chat message to the user, providing a clear explanation with as much details as possible of the error and how to resolve it.\n"
        "The customer can't reply to your message so make sure you tell the user if he need any further help to contact the support team using this link 'https://optibus.my.site.com/Support/s/'\n"
        "You don't need to do any kind of greeting or goodbyes you can always finish by telling him to reach out to support\n"
        "Remember, the goal is to accurately categorize the error so that users can be guided on how to resolve the issue, or to contact support if it is a system error they cannot resolve themselves. Think carefully about the root cause of the error based on the details provided in the message.\n"
        "Keep in mind that the user that is getting the error message might not be technical so try to keep the explanation as simple as possible\n"
    ),
)

examples = [
    {
        "input": "Timeplan couldnt be imported\nThe file could not be imported because one or more rows contain unsupported values. To resolve the issue go to the file and edit the affected rows.Then import again",
        "output": "Data input",
    },
    {
        "input": "{\n     'recording_id': 'CreateRosterReports/20220913/fa23cee4337511ed955dbebf55008eb8',\n     'category': 'uncategorized_error',\n     'short_description': 'Something went wrong. If you try again and still need help, contact support@optibus.com.',\n     'parameters': {},\n     'sub_category': 'xlsxwriter.exceptions.FileCreateError',\n     'explanation': null,\n     'detailed_description': null,\n     'debug': '&lt;class 'xlsxwriter.exceptions.FileCreateError'&gt;\n    FileCreateError(IOError(2, 'No such file or directory'),)',\n     'severity': 4,\n     'event_type': 'unknown',\n     'stack_trace': '  File \\'/euclid/euclid/http/rostering.py\\', line 285, in _post\n        ignore_stats=ignore_stats)\n      File \\'/euclid/euclid/rostering/reports.py\\', line 137, in create_all_roster_reports\n        to_xlsx(report, file_name, force_string_cell_value=force_string_cell_value)\n      File \\'/envp/site-packages/optibus/utils/io_utils.py\\', line 329, in to_xlsx\n        workbook.close()\n      File \\'/envp/site-packages/xlsxwriter/workbook.py\\', line 322, in close\n        raise FileCreateError(e)\n    ',\n     'request_id': 'IkUyYnGImxCC2Wxd',\n     'task_id': null,\n     'externalError': true,\n     'main_schedule_id': null,\n     'user': 'marco.semeraro@optibus.com',\n     'customer': 'sg-east',\n     'account_name': 'Stagecoach UK Bus',\n     'account_region': 'Europe',\n     'country': 'United Kingdom',\n     'requestId': 'IkUyYnGImxCC2Wxd',\n     'operationType': 'exportRoster',\n     'client_url': 'https://sg-east.optibus.co/project/r3f3npmkr/rosterSchedules/rv5drNdI1Y/roster-calendar',\n     'project_id': null,\n     'schedules': [\n          'fijQGNSGP',\n          'fcTCLTdoaD',\n          'fEb8vpEROe'\n     ],\n     'scheduleSets': [\n          '7HfUlL198'\n     ],\n     'datasets': [\n          'DCEaZkwYN'\n     ],\n     'data set props': [\n          '6ZbAmkNp7'\n     ],\n     'deadheadCatalogs': [\n          'vpAbOrlel'\n     ],\n     'taxiCatalogs': [\n          'YoAYPHSoW',\n          'MSoPy5Z5D',\n          'xcbpaEUW3'\n     ]\n}",
        "output": "Data input",
    },
    {
        "input": "Export could not be completed\nThe action could not be completed because there is a schema issue with the Public Transportation 1 preference. Please contact support@optibus.com for assistance",
        "output": "Configuration",
    },
]


def build_context():
    # Build context as string with the following format
    # "Examples: \n\n"
    # "Input: <input>\n"
    # "Output: <output>\n\n"
    # "Input: <input>\n"
    # "Output: <output>\n\n"
    return """Suggested Solution: This error is usually thrown when a Time Definition is deleted from the schedule's preferences when it is still being used by another preference, such as a Global Constraint, a different Time Definition, or one of the Preference Designers. The user should review the preferences in their schedule to find the existing preference which references this Time Definition, and either delete the preference or change it so it references a different, not-deleted, time definition.

Time Definitions are preferences that allows users to define various time-related parameters of duties. Those parameters are then used in optimization, stats, and various linked preferences.	Issue: On analyse, the error message tells us something is missing but not which preference is causing the error, so the user doesn’t know how to fix it. Currently the expected flow only caters for users that want to re-add the time definition, but doesn’t help users that want to remove or replace a time definition."""
    return "\n\n".join(
        [
            f"Input: {example['input']}\nOutput: {example['output']}"
            for example in examples
        ]
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
    # res = agent.invoke(input=input, context="")
    res = classification_agent.invoke(
        input=(
            "Here is the context for the task:\n"
            "<context>\n"
            f"{build_context()}\n"
            "</context>\n"
            "Here is the error message to classify:'n"
            "<input>\n"
            f"{input}\n"
            "</input>"
        )
    )
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
        url, auth=HTTPBasicAuth(username, api_token), params={"expand": "body.storage"}
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
