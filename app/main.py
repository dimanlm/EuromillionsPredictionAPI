from fastapi import FastAPI, Depends
from pydantic import BaseModel
import learnmodel
import os
import pandas as pd
from varfile import *
from api import predict, model
# varfile contains the paths to different files
# such as .csv data file, or .joblib files

app = FastAPI()
app.include_router(predict.router)
app.include_router(model.router)

@app.get('/')
async def main():
    return {'Hi! Visit /docs to try all the available requests :)'}