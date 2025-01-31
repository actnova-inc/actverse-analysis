import json
from urllib.parse import urlparse

import requests


def is_url(x: any):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def load_json(path: str) -> dict:
    # check path is url or local file
    if is_url(path):
        response = requests.get(path)
        return response.json()
    else:
        with open(path, "r") as f:
            return json.load(f)
