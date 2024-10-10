import requests

#resp = requests.post("https://getprediction-cqgendxtea-et.a.run.app", files={'file': open('eight.png', 'rb')})

resp = requests.post("http://localhost:5000/", files={'file': open('mrcnnima.jpg', 'rb')})
print(resp.json())
