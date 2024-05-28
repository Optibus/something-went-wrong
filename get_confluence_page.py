import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup


def pretty_print_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator="\n")
def get_confluence_page_content(base_url, page_id, username, api_token):
    url = f"{base_url}/rest/api/content/{page_id}?expand=body.storage"
    auth = HTTPBasicAuth(username, api_token)
    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, auth=auth, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve page data: {response.status_code}")



def main():
    base_url = "https://optibus.atlassian.net/wiki"  # Your Confluence base URL
    page_id = ""  # Replace with the actual page ID from the Confluence page URL, PAGEID is -
    # https://your-domain.atlassian.net/wiki/spaces/SPACEKEY/pages/PAGEID/Page+Title
    api_token = "your-api-token"  # Replace with your actual API token generated from Confluence
    username = "your-email@example.com"  # Replace with your actual Atlassian account email (username)

    try:
        page_data = get_confluence_page_content(base_url, page_id, username, api_token)
        # Extract and print the content of the page
        page_content = page_data['body']['storage']['value']
        pretty_content = pretty_print_html_content(page_content)
        print(pretty_content)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()