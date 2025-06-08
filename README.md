### 1. Write App (Flask, TensorFlow)
- The code to build, train, and save the model is in the `test` folder.
- Implement the app in `main.py`
### 2. Setup Google Cloud 
- Create new project
- Activate Cloud Run API and Cloud Build API

### 3. Install and init Google Cloud SDK
- https://cloud.google.com/sdk/docs/install

### 4. Dockerfile, requirements.txt, .dockerignore
- https://cloud.google.com/run/docs/quickstarts/build-and-deploy#containerizing

### 5. Cloud build & deploy
```
gcloud builds submit --tag gcr.io/positive-tempo-302918/index
gcloud run deploy --image gcr.io/positive-tempo-302918/index --platform managed
```
### 6. Local environment
```
install docker
go to the directory where is located the docker
docker build -t my-flask-app .
docker images
take the field:  IMAGE ID  0bda0ee1faec
docker run -e PORT=5000 -p 5000:5000 0bda0ee1faec
to listen in the host IP (tested)
docker run -e PORT=5000 -p <HOST_PUBLIC_IP>:5000:5000 0bda0ee1faec
```

### Test
- Test the code with `test/test.py`

### For another reference and practical examples watch the video tutorial
- How To Deploy ML Models With Google Cloud Run

[![Alt text](https://img.youtube.com/vi/vieoHqt7pxo/hqdefault.jpg)](https://youtu.be/vieoHqt7pxo)
