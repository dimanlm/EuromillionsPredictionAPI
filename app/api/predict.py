from datetime import date
from sys import prefix
from fastapi import Depends, APIRouter
import learnmodel
import os
import pandas as pd
from pydantic import BaseModel
from varfile import *

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


router = APIRouter(
    prefix="/api/predict",
)


@router.post("/")
async def predictTheResultOfInputData(donnees: dataEuro):
    list=donnees.getlist()
    # checking if the model is already trained
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        m, c= learnmodel.chargement() # m => forest, c => cluster
    else:
        return{'error': TRAIN_THE_MODEL_MSG}
    p= learnmodel.prediction(m,c, list)
    return {"Proba_perte": f"{p[0][0]*100}%",
            "Proba_gain" : f"{p[0][1]*100}%"}

@router.get("/")
async def getMyWinningCombo():
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        m, c= learnmodel.chargement()
        return{"winning_combo": learnmodel.generationChiffres(m,c)}

    return{"error": TRAIN_THE_MODEL_MSG}

