from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    reponse:int 

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status" : "healthy"}      

@app.post("/predict", response_model=IrisResponse)
async def predict_irsi(iris_feature: IrisData):
    try:
        features = [[iris_feature.sepal_length, iris_feature.sepal_width,iris_feature.petal_length,iris_feature.petal_width]]
        prediction = predict_data(features)
        return IrisResponse(reponse=int(prediction[0]))   
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))