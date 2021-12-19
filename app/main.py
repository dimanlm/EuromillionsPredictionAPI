from datetime import date
from fastapi import FastAPI
from pydantic import BaseModel
import model
import os
import pandas as pd
from varfile import *
# varfile contains the paths to different files
# such as .csv data file, or .joblib files

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

    
    def isValidSuites(self):
        suiteN = [self.N1, self.N2, self.N3, self.N4, self.N5]
        suiteE = [self.E1, self.E2]
        for i in range(len(suiteN)):
            if (0>suiteN[i] or suiteN[i]>50):
                raise ValueError("Suite value must be [0,50]")
        for j in range(len(suiteE)):
            if (0>suiteE[j] or suiteE[j]>12):
                raise ValueError("Etoile value must be [0,12]")


app = FastAPI()

@app.post("/api/predict/")
async def predictTheResultOfInputData(donnees: dataEuro):
    list=donnees.getlist()
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        m, c= model.chargement() # m => forest, c => cluster
    else:
        return{"Train the model first please."}
    p= model.prediction(m,c, list)
    return {"Proba_perte": f"{p[0][0]*100}%",
            "Proba_gain" : f"{p[0][1]*100}%"}


@app.post("/api/model/retrain/")
async def trainModel():
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        model.entrainement()
        return {"model retrained"}
    else:
        model.entrainement()
        return {"model trained for the first time"}


@app.get("/api/predict/")
async def getMyWinningCombo():
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        m, c= model.chargement()
        return{"winning_combo": model.generationChiffres(m,c)}

    return{"error": "You need to train your model first"}


@app.get("/api/model")
async def getModelDetails():
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        m, c= model.chargement()
        return{"model_details": model.description(m,c)}

    return{"error": "You need to train your model first"}


@app.put("/api/createdata/{train_model_choise}")
async def createNewData(data: newDataEuro, train_model_choise: bool):
    # check if the input is correct: N = [0,50] and E = [0,12]
    try:
        data.isValidSuites()
    except ValueError:
        msg = "Invalid data. N values must be [0,50] and E must be [0,12]"
        return{"msg": msg}
    
    model.ajoutDonnees(data.dict())
    
    if train_model_choise:
        model.entrainement()
        msg = "The model has been retrained with the new data"
    else:
        msg = "New data has been added. You can use '/api/model/retrain/' to update the model"
    
    return {
        'train?' : train_model_choise,
        'msg': msg,
        }
