import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray


model_ref = bentoml.sklearn.get('mlzoomcamp_homework:qtzdz3slg6mwwdu5')
model_runner = model_ref.to_runner()

svc = bentoml.Service('mlzoomcamp_homework', runners=[model_runner])

@svc.api(input=NumpyNdarray(shape=(-1,4),enforce_shape=True), output=NumpyNdarray())
def classify(vector):
    prediction = model_runner.predict.run(vector)
    result = prediction[0]
    print('Prediction:', result)
    return result
