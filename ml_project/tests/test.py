import unittest
from faker import Faker
import pandas as pd
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, r'/home/code/made_mlops_autumn_2022/')

from src.predict import predict
from src.train import run_train
from src.train_test_split import run_split_csv


class TestModules(unittest.TestCase):
    def test_synthetyc_dataset(self):
        count = 100
        df = self.get_dataset(count)      
        df.to_csv('../data/test_dataset.csv', index=False)  
        
        run_split_csv()
        run_train()
        predict('../data/test.csv', "../models/log_regression.joblib", "../data/predictions.csv")
        self.test_predict()
        
    def get_dataset(self, count):
        Faker.seed(42)
        fake = Faker()
        
        age = [fake.pyint(20, 90) for i in range(count)]
        sex = [fake.pyint(0, 1) for i in range(count)]
        cp = [fake.pyint(1, 3) for i in range(count)]
        trestbps = [fake.pyint(90, 200) for i in range(count)]
        chol = [fake.pyint(100, 600) for i in range(count)]
        fbs = [fake.pyint(0, 1) for i in range(count)]
        restecg = [fake.pyint(0, 2) for i in range(count)]
        thalach = [fake.pyint(50, 250) for i in range(count)]
        exang = [fake.pyint(0, 1) for i in range(count)]
        oldpeak = [fake.pyfloat(0, 7) for i in range(count)]
        slope = [fake.pyint(0, 2) for i in range(count)]
        ca = [fake.pyint(0, 3) for i in range(count)]
        thal = [fake.pyint(0, 2) for i in range(count)]
        condition = [fake.pyint(0, 1) for i in range(count)]

        lst = list(zip(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal, condition))
        
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                          'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']
        
        df = pd.DataFrame(lst, columns=columns)
        return df
    
    def test_predict(self):
        y_true = pd.read_csv('/home/code/made_mlops_autumn_2022/data/y_true.csv')
        y_pred = pd.read_csv('/home/code/made_mlops_autumn_2022/data/predictions.csv')
        
        accuracy = accuracy_score(y_true, y_pred)
        self.assertLess(0.7, accuracy)


if __name__ == "__main__":
    unittest.main()