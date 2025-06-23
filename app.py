from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("svd_model.pkl")

class Request(BaseModel):
    user_id: str
    item_id: str

@app.post("/predict")
def predict(req: Request):
    prediction = model.predict(req.user_id, req.item_id)
    return {
        "user_id": prediction.uid,
        "item_id": prediction.iid,
        "estimated_rating": prediction.est
    }
