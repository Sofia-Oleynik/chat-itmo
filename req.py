import requests

url = "http://127.0.0.1:5000/api/request"
data = {
    "query": "Tell about ITMO",
    "id": 1
}

response = requests.post(url, json=data)


print(response)
