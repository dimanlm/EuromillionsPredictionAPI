import sys
sys.path.insert(1,'../app')
import os
import learnmodel
from fastapi.testclient import TestClient
from main import app
from varfile import *

client = TestClient(app)

def test_main():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == ['Hi! Visit /docs to try all the available requests :)']


def test_get_prediction_combo():
    response = client.get('/api/predict/')
    assert response.status_code == 200
    try:
        os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER)
    except:
        assert response.json() == {'error': TRAIN_THE_MODEL_MSG}


def test_prediction_input():
    response = client.post('/api/predict/',
        json={"n1": 0, "n2": -1,  "n3": -5,  "n4": 0,  "n5": 5,  "e1": 13,  "e2": 0},
    )
    assert response.status_code == 200
    if os.path.exists(GENERATED_MODEL) and os.path.exists(GENERATED_CLUSTER):
        assert response.json() == {'error': INVALID_DATA_MSG}
    else:
        assert response.json() == {'error': TRAIN_THE_MODEL_MSG}

    # another test
    response = client.post('/api/predict/',
        json={"n1": 'a', "n2": 'b',  "n3": 'c',  "n4": 'd',  "n5": 'e',  "e1": 13,  "e2": 0},
    )
    assert response.status_code == 422
    


def test_create_new_data():
    response = client.post('/api/model/{train_model_choise}',
        json={
            "Date": "2021-12-19",
            "N1": 0,
            "N2": 0,
            "N3": 0,
            "N4": 0,
            "N5": 0,
            "E1": 0,
            "E2": 0,
            "Winner": 0,
            "Gain": 0
            },
    )
    assert response.status_code == 405


def test_create_new_data():
    train_model_choise = True
    response = client.put(f'/api/model/{train_model_choise}', 
        json={
            "Date": "2021-12-19",
            "N1": 25,
            "N2": 100,
            "N3": 52,
            "N4": -5,
            "N5": 0,
            "E1": 12,
            "E2": 0,
            "Winner": 0,
            "Gain": 0
            }
    )
    assert response.status_code == 200
    assert response.json() == {'msg' : INVALID_DATA_MSG}

    ## another test
    train_model_choise = False
    response = client.put(f'/api/model/{train_model_choise}', 
        json={
            "Date": "2021-12-19",
            "N1": 'a',
            "N2": 'b',
            "N3": 'c',
            "N4": 'd',
            "N5": 'e',
            "E1": 'f',
            "E2": 'g',
            "Winner": 'h',
            "Gain": 'i'
            }
    )
    assert response.status_code == 422

def test_create_new_data_2():
    train_model_choise = False
    response = client.put(f'/api/model/{train_model_choise}', 
        json={
            "Date": "20-21-12",
            "N1": 0,
            "N2": 0,
            "N3": 0,
            "N4": 0,
            "N5": 0,
            "E1": 0,
            "E2": 0,
            "Winner": 0,
            "Gain": 0
            }
    )
    assert response.status_code == 422
