from datetime import date
from fastapi import Depends, APIRouter
import learnmodel
import os
import pandas as pd
from pydantic import BaseModel
from varfile import *


class newDataEuro(BaseModel):
    Date: date 
    N1: int
    N2: int
    N3: int
    N4: int
    N5: int
    E1: int
    E2: int
    Winner: int
    Gain: int

    '''
    Checks if the input data is valid. N in range [0,50] and E in [0,12]
    '''
    def isValidSuites(self):
        suiteN = [self.N1, self.N2, self.N3, self.N4, self.N5]
        suiteE = [self.E1, self.E2]
        for i in range(len(suiteN)):
            if (0>suiteN[i] or suiteN[i]>50):
                raise ValueError("Suite value must be [0,50]")
        for j in range(len(suiteE)):
            if (0>suiteE[j] or suiteE[j]>12):
                raise ValueError("Etoile value must be [0,12]")



router = APIRouter(
    prefix="/api/model",
)

@router.post("/retrain")
async def trainModel():
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        learnmodel.entrainement()
        return {"model retrained"}
    else:
        learnmodel.entrainement()
        return {"model trained for the first time"}


@router.get("/")
async def getModelDetails():
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        m, c= learnmodel.chargement()
        return{"model_details": learnmodel.description(m,c)}

    return{"error": "You need to train your model first"}


@router.put("/{train_model_choise}")
async def createNewData(data: newDataEuro, train_model_choise: bool):
    # check if the input is correct: N = [0,50] and E = [0,12]
    try:
        data.isValidSuites()
    except ValueError:
        msg = "Invalid data. N values must be [0,50] and E must be [0,12]"
        return{"msg": msg}
    
    learnmodel.ajoutDonnees(data.dict())
    
    if train_model_choise:
        learnmodel.entrainement()
        msg = "The model has been retrained with the new data"
    else:
        msg = "New data has been added. You can use '/api/model/retrain/' to update the model"
    
    return {
        'train?' : train_model_choise,
        'msg': msg,
    }
