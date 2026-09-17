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
        _record_load(path, status_code=response.status_code,
                     content_type=response.headers.get("Content-Type"),
                     size=len(response.content))
        return response.json()
    else:
        with open(path, "r") as f:
            data = json.load(f)
        _record_load(path, status_code=None, content_type="local", size=None)
        return data


def _record_load(path, **info):
    # ACT-5939: 어떤 경로에서 무엇을 받았는지 요약만 남긴다 (본문은 보내지 않음)
    try:
        from actverse.telemetry import record_call

        shown = path if not is_url(path) else path.split("?")[0]
        record_call("load_json", path=shown[:300], is_url=is_url(path), **info)
    except Exception:
        pass
