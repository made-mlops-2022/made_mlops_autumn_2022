import requests
import json


def get_response(path_to_json):
    url = "http://127.0.0.1:8000/predict"

    with open(path_to_json, encoding="utf-8") as f:
        json_list = json.load(f)

    res = requests.post(url, json=json_list[0])

    if res:
        print("Response OK")
    else:
        print("Response Failed")


if __name__ == '__main__':
    get_response("tests.json")