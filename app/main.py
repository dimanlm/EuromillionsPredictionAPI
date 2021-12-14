from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import model
import os
import pandas as pd

class dataEuro(BaseModel):
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    e1: int
    e2: int

    def getlist(self):
        return [self.n1,self.n2,self.n3,self.n4,self.n5,self.e1,self.e2]

class newDataEuro(BaseModel):
    Date: str
    N1: int
    N2: int
    N3: int
    N4: int
    N5: int
    E1: int
    E2: int
    Winner: int
    Gain: int


app = FastAPI()

@app.post("/api/predict/")
async def getPrediction(donnees: dataEuro):
    list=donnees.getlist()
    if os.path.exists('model.joblib') and os.path.exists('clustering.joblib'):
        m, c= model.chargement() # m_foret, c_cluster
    else:
        return{"Train the model first please."}
    p= model.prediction(m,c, list)
    return {"p": p}

@app.post("/api/train/")
async def trainModel():
    if os.path.exists('model.joblib') and os.path.exists('clustering.joblib'):
        m_foret, c = model.entrainement()
        return {"model retrained"}
    else:
        m, c= model.entrainement()
        return {"model trained for the first time"}

@app.get("/api/predict/")
async def getMyPredict():
    if os.path.exists('model.joblib') and os.path.exists('clustering.joblib'):
        m, c= model.chargement()
        return{"pred": model.generationChiffres(m,c)}

    return{"error": "You need to train your model first"}


@app.get("/api/model")
async def getModelDetails():
    if os.path.exists('model.joblib') and os.path.exists('clustering.joblib'):
        m, c= model.chargement()
        return{"pred": model.description(m,c)}

    return{"error": "You need to train your model first"}


@app.put("/api/createdata/{newdataId}")
async def createNewData(data: newDataEuro):
    print(model.ajoutEtEntrainement(data.dict()))
    model.entrainement()
    return {"model trained with the new data"}