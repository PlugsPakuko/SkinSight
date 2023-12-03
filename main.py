# model
from model import predict as model_predict
# Web Server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# create app
app = FastAPI()
# alow cross origin all origin

# index route
@app.get('/')
def index():
    return {'message': 'This is a SkinSight model API.'}

# predict route
@app.get('/predict')
def predict(URL: str, MODEL: str):
    return {'text': model_predict(URL, MODEL)}

# start server
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)