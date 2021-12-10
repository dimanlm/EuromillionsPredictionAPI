import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

chemin_fichier = "../data/EuroMillions_numbers.csv"

def generation_data_perdante(nb):
    '''
    Permet la génération de donnée perdante au loto pour un future entraînement.Le tirage est aléatoire et sans remise

    nb: Nombre de lignes à générer 
    '''    
    ligne_data = []
    for i in range(nb):

        a = np.random.choice(range(1,50),5, replace = False)
        b = np.random.choice(range(1,11),2, replace = False)

        ligne_data.append(list(a) + list(b))
    df = pd.DataFrame(ligne_data, columns=["N1","N2","N3","N4","N5","E1","E2"])
    df['Winner'] = 0

    return df


def creation_data():
    "Permet la création du tableau de données"

    data = pd.read_csv(chemin_fichier, sep = ";")
    data = data.drop(['Winner',	'Gain', 'Date'], axis = 1)
    data['Winner'] = 1

    data_perdu = generation_data_perdante(8 * data.shape[0])
    data_complete = pd.concat([data, data_perdu])
    X = data_complete.drop('Winner', axis = 1)
    y = data_complete['Winner']

    return (X, y)


def feature_engineering():
    X,y = creation_data()
    kmeans = KMeans(15).fit(X)
    X['Cluster'] = kmeans.labels_

    return X, y, kmeans


def entrainement():
    X, y, clustering  = feature_engineering()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, stratify=y)
    foret = RandomForestClassifier(oob_score=True).fit(X_train, y_train)
    dump(foret, 'model.joblib')

    return foret, clustering


def prediction(foret, clustering, chiffres):
    cluster = clustering.predict([chiffres])[0]
    foret.predict_proba([chiffres+ [cluster]])
    return foret.predict_proba([chiffres+ [cluster]])


def chargement():
    if(os.path.exists('model.joblib')):
        return load('model.joblib')
