from fastapi import Depends, FastAPI
from datetime import date
from pydantic import BaseModel
import learnmodel
import os
import pandas as pd
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
