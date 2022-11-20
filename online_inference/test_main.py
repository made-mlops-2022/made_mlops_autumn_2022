from fastapi.testclient import TestClient
from server import app
import json


client = TestClient(app)
    

def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_ok():
    with open('tests.json', encoding='utf-8') as f:
        json_list = json.load(f)
    
    for json_example in json_list:
        response = client.post(
            "/predict",
            json=json_example
        )
        assert response.status_code == 200
        assert response.json() == json_example['condition']


def test_predict_wrong_values():
    good_json={
        "age": 150,
        "sex": 1,
        "cp": 3,
        "trestbps": 0,
        "chol": 0,
        "fbs": 1,
        "restecg": 2,
        "thalach": 0,
        "exang": 1,
        "oldpeak": 0,
        "slope": 2,
        "ca": 3,
        "thal": 2
    }
    
    big_values={
        "age": 151,
        "sex": 2,
        "cp": 4,
        "fbs": 2,
        "restecg": 3,
        "exang": 2,
        "slope": 3,
        "ca": 4,
        "thal": 3
    }
    
    for key, val in big_values.items():
        good_val = good_json[key]
        good_json[key] = val
        response = client.post(
            "/predict",
            json=good_json
        )
        good_json[key] = good_val
        assert response.status_code == 422
    
    for key, val in big_values.items():
        good_val = good_json[key]
        good_json[key] = -1
        response = client.post(
            "/predict",
            json=good_json
        )
        good_json[key] = good_val
        assert response.status_code == 422


def test_predict_null_values():
    good_json={
        "age": 150,
        "sex": 1,
        "cp": 3,
        "trestbps": 0,
        "chol": 0,
        "fbs": 1,
        "restecg": 2,
        "thalach": 0,
        "exang": 1,
        "oldpeak": 0,
        "slope": 2,
        "ca": 3,
        "thal": 2
    }

    for key, val in good_json.items():
        good_json[key] = None
        response = client.post(
            "/predict",
            json=good_json
        )
        good_json[key] = val
        assert response.status_code == 422