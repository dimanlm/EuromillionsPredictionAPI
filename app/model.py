import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/EuroMillions_numbers.csv', sep = ";", parse_dates = True)
data = data.drop(['Winner',	'Gain', 'Date'], axis = 1)
data['Winner'] = 1


def generation_data_perdante(nb):
    
    ligne_data = []
    for i in range(nb):

        a = np.random.choice(range(1,50),5, replace = False)
        b = np.random.choice(range(1,11),2, replace = False)

        ligne_data.append(list(a) + list(b))
    df = pd.DataFrame(ligne_data, columns=["N1","N2","N3","N4","N5","E1","E2"])
    df['Winner'] = 0
    return df

data_perdu = generation_data_perdante(8 * data.shape[0])
data_complete = pd.concat([data, data_perdu])
X = data_complete.drop('Winner', axis = 1)
y = data_complete['Winner']


def entrainement(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, stratify=y)
    foret = RandomForestClassifier(oob_score=True).fit(X_train, y_train)
    return foret

def prediction(model, chiffres):
    return model.predict_proba([chiffres])