import requests
from time import sleep

url = "http://localhost:8080/predict"

client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
#response = requests.post(url, json=client).json()

#print(response)
while True:
    sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)
