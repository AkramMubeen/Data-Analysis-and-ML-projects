FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy
COPY ["app.py","preprocess.py","vectorizer.pkl","LRmodel.pkl","./"]

EXPOSE 9696

ENTRYPOINT [ "waitress-serve" ,"--listen=0.0.0.0:9696","app:app" ]