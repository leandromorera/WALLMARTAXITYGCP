import requests

#resp = requests.post("https://getprediction-cqgendxtea-et.a.run.app", files={'file': open('eight.png', 'rb')})

resp = requests.post("https://mrcnnfast-cqgendxtea-uc.a.run.app", files={'file': open('mrcnnima.jpg', 'rb')})
print(resp.json())
