import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load the "champion" model from MLflow
MODEL_NAME = "msg_cls"
MODEL_ALIAS = "champion"
MLFLOW_TRACKING_URI = "http://mlflow:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        prediction = model.predict([request.text])
        return {"is_bot_probability": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))