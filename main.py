import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import warnings
#N	P	K	temperature	humidity	ph	rainfall	
warnings.filterwarnings("ignore")

app=FastAPI()

class Dist(BaseModel):
    nitrogen:float
    phosporus:float
    potasium:float
    temperature:float
    humidity:float
    ph:float
    rainfall:float
model=joblib.load("sivabasha_crop1")


@app.get("/")
def home():
    return {"this is your home page"}

@app.post("/predict")
def predict_output(data:Dist):
    recieved=data.dict()
    nitrogen=recieved['nitrogen']
    phosporus=recieved["phosporus"]
    potasium=recieved["potasium"]
    temperature=recieved["temperature"]
    humidity=recieved["humidity"]
    ph=recieved["ph"]
    rainfall=recieved["rainfall"]
    result=model.predict([[nitrogen,phosporus,potasium,temperature,humidity,ph,rainfall]])
    result1=result[0]
    return {"prediction ":result1}



if __name__=="__main__":
    uvicorn.run(app)
