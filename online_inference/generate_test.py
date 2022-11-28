from faker import Faker
from made_mlops_autumn_2022.online_inference.model import Model
import json
import numpy as np


def generate_json():
    fake = Faker()
    json = {}
    
    json['age'] = fake.pyint(0, 150)
    json['sex'] = fake.pyint(0, 1)
    json['cp'] = fake.pyint(0, 3)
    json['trestbps'] = fake.pyint(0, 400)
    json['chol'] = fake.pyint(0, 600)
    json['fbs'] = fake.pyint(0, 1)
    json['restecg'] = fake.pyint(0, 2)
    json['thalach'] = fake.pyint(0, 300)
    json['exang'] = fake.pyint(0, 1)
    json['oldpeak'] = fake.pyfloat(min_value=0, max_value=7)
    json['slope'] = fake.pyint(0, 2)
    json['ca'] = fake.pyint(0, 3)
    json['thal'] = fake.pyint(0, 2)
    
    arr = np.array([json['age'], json['sex'], json['cp'], json['trestbps'],
           json['chol'], json['fbs'], json['restecg'], json['thalach'],
           json['exang'], json['oldpeak'], json['slope'], json['ca'], json['thal']])
    
    return json, arr


def generator(model):
    json_dict, arr = generate_json()
    pred = model.predict(arr)
    json_dict['condition'] = pred
    return json_dict


if __name__ == '__main__':
    data = []
    model = Model()
    for i in range(20):
        example = generator(model)
        data.append(example)
    
    data = json.dumps(data, indent=4)
    
    with open("tests.json", "w") as outfile:
        outfile.write(data)
    