# TP Euromillions

Euro Millions prediction
API applied to AI

Architecture Microservice

Group:
Dmytro Matsepa (ICC)
Pierre Virgaux (IA)

## Installation


```console
$ git clone https://github.com/dimanlm/TPEuromil.git
$ sudo apt install python3-pip
$ sudo apt install uvicorn
$ cd TPEuromil
$ sudo pip3 install -r requirements.txt
```

## Usage

```console
$ cd app
$ uvicorn main:app --reload
```
Once you execute the **uvicorn** command above in the **/TPEuromil/app** folder, you can go to **http://127.0.0.1:8000/docs** on your browser and try the different requests.

![Logo](https://i.imgur.com/C99ZVzq.png)

## Requests
* **GET /api/predict**: it will generate a combination of numbers with a high probability of winning. The prediction should be a sequence of numbers, predicted to win, from the model.

* **POST /api/predict**: it allows you to make a prediction based on an input draw proposal. The prediction must be probabilistic (Prob. win: X%, Prob. loss: 1-X%).

* **POST /api/model/retrain**: it allows the model to be re-trained. It must take into account the data added afterwards.

* **GET /api/model**: it provides the technical information of the model:
  * Performance metrics
  * Name of the algorithm
  * Training parameters

* **PUT /api/model**: it is used to enrich the model with additional data. Additional data must have the same format as the rest of the data.
  * The {train_model_choise} is a boolean that allows you to choose if you either want or not to retrain the model with the new data.
  * N values must be in range [0,50] and the E values in [0,12]. An error message will be displayed returned.


## Pytests

```console
$ cd test
$ pytest
```
Once you execute the **uvicorn** command above in the **/TPEuromil/app** folder, you can go to **http://127.0.0.1:8000/docs** on your browser and try the different requests.
