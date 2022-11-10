import pandas as pd
import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

columns_to_encode = ['Sex','ChestPainType','FastingBS','RestingECG','ExerciseAngina','ST_Slope']
columns_to_scale  = ['Age', 'RestingBP','Cholesterol','MaxHR','Oldpeak']


class Patient(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

model_ref = bentoml.xgboost.get('heart_failure_prediction:latest')

dv = model_ref.custom_objects['dictVectorizer']

scaler = model_ref.custom_objects['minmaxscaler']

model_runner = model_ref.to_runner()

svc = bentoml.Service('heart_failure_classifier', runners=[model_runner])

@svc.api(input=JSON(pydantic_model=Patient), output=JSON())
def classify(patient):
    application_data = patient.dict()
    data = pd.DataFrame(data=application_data,index=[0])
    scaled_columns  = scaler.transform(data[columns_to_scale])
    encode_dict = data[columns_to_encode].to_dict(orient='records')
    encoded_columns = dv.transform(encode_dict)
    vector = np.concatenate([scaled_columns, encoded_columns], axis=1)
    prediction = model_runner.predict.run(vector)
    result = prediction[0]
    print('Prediction:', result)
    if result == 1:
        return {'Status': 'Its bad news, you have a high chance of heart failure.'}
    else:
        return {'Status': 'Its good news, you do not need to worry.'}

