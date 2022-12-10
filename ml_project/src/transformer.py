from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


class CustomTransformer(BaseEstimator, TransformerMixin):
    def  __init__(self):
       ...
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = preprocessing.normalize(X_)
        return X_


pipeline = make_pipeline(CustomTransformer(), LogisticRegression())
pipeline.fit()
pipeline.predict()