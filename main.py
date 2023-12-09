# model
from model import predict as model_predict
# Web Server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO

# create app
app = FastAPI()
# alow cross origin all origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# index route
@app.get('/')
def index():
    return {'message': 'This is a SkinSight model API.'}

# predict route
@app.get('/predict')
def predict(URL: str, MODEL: str):
    if URL is None or MODEL is None:
        return {'error': 'Both URL and MODEL parameters are required.'}
    
    pred = model_predict(URL, MODEL)
    return {'Level': pred[1],
            'Confidence': pred[0]
            }

# start server
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)