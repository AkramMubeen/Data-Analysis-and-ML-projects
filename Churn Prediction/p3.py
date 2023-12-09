import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import pickle

dv_path = 'dv.bin'
model_path = 'model1.bin'

with open(dv_path, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

customer = {"reports": 0, "share": 0.001694,
            "expenditure": 0.12, "owner": "yes"}


X = dv.transform([customer])

y_pred = model.predict_proba(X)[0,1]
print(f"input: {customer}")
print(f"churn probability: {y_pred}")
