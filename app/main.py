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


app = FastAPI()

@app.post("/predict/")
async def getPrediction(donnees: dataEuro):
    list=donnees.getlist()
    if (os.path.exists('model.joblib')):
        m= model.chargement()
        _, c = model.entrainement()
        msg = {"model.joblib loaded"}
    else:
        m, c= model.entrainement()
        msg = "model.joblib not found. The model has been trained"

    p= model.prediction(m,c, list)
    return {"p": p, "msg": msg }


@app.post("/train/")
async def trainModel():
    m, c = model.entrainement()
    return {"m": 'model.joblib'}