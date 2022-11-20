from joblib import load
import numpy as np
from sklearn import preprocessing
from fastapi import HTTPException, status


class Model:
    def __init__(self):
        self.model = None
        path_to_model="weights/log_regression.joblib"
        try:
            self.model = load(path_to_model)
        except:
            self.is_ready()
    
    
    def predict(self, X_array):
        self.is_ready()
        
        X_array = np.reshape(X_array, (1, len(X_array)))        
        X_array = preprocessing.normalize(X_array)
        y_pred = self.model.predict(X_array)
        
        return y_pred.tolist()


    def is_ready(self):
        if self.model == None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'model dont ready',
            )
        return True