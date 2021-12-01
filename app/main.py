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
        return [self.n1,self.n2,self.n3,self.n4,self.n5,self.e1,self.e2, 1]


app = FastAPI()

@app.post("/saisie/")
async def saisie_donnees(donnees: dataEuro):
    list=donnees.getlist()
    if (os.path.exists('mon_model.joblib')):
        m= model.chargement()
    else:
        m= model.entrainement()
    p= model.prediction(m, list)

    return {"p": p}