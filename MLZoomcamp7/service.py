import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class UserProfile(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int

model_ref = bentoml.xgboost.get('credit_risk_model:latest')

dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])

@svc.api(input=JSON(pydantic_model=UserProfile), output=JSON())
def classify(user_profile):
    application_data = user_profile.dict()
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    result = prediction[0]
    print('Prediction:', result)
    if result > 0.5:
        return {'Status': 'Sorry your application has been declined'}
    elif result > 0.3:
        return {'Status': 'Your application for loan has been accepted but you have to wait'}
    else:
        return {'Status': 'Congratulations, your loan application has been accepted.'}
