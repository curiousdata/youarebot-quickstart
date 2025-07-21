from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

CLASSIFIER_URL = "http://classifier:8000/predict"
LLM_URL = "http://llama-server:11434/v1/chat/completions"


class PredictRequest(BaseModel):
    text: str


class GetMessageRequest(BaseModel):
    dialog_id: str
    last_msg_text: str
    last_message_id: str


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        response = requests.post(CLASSIFIER_URL, json=request.dict())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_message")
async def get_message(request: GetMessageRequest):
    try:
        response = requests.post(LLM_URL, json=request.dict())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))