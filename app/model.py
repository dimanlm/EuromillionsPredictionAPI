import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PATH_TO_DATA_FILE = "../data/EuroMillions_numbers.csv"

def generationDataPerdante(nb):
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


def creationData(original_modif = False):
    '''
    Permet la création du tableau de données et la mise en place d'un échantillonage introduisant des joueurs perdants
    '''

    if original_modif :
        return pd.read_csv(PATH_TO_DATA_FILE, sep = ";")

    data = pd.read_csv(PATH_TO_DATA_FILE, sep = ";")
    data = data.drop(['Winner',	'Gain', 'Date'], axis = 1)
    data['Winner'] = 1

    data_perdu = generationDataPerdante(8 * data.shape[0])
    data_complete = pd.concat([data, data_perdu])
    X = data_complete.drop('Winner', axis = 1)
    y = data_complete['Winner']

    return (X, y)


def featureEngineering():
    '''
    Permet la mise en place de clustering 
    '''
    X,y = creationData()
    kmeans = KMeans(15).fit(X)
    X['Cluster'] = kmeans.labels_

    return X, y, kmeans


def entrainement():
    X, y, clustering  = featureEngineering()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, stratify=y)
    foret = RandomForestClassifier(oob_score=True).fit(X_train, y_train)
    dump(foret, 'model.joblib')
    dump(clustering, 'clustering.joblib')
    return foret, clustering


def prediction(foret, clustering, chiffres):

    cluster = clustering.predict([chiffres])[0]
    foret.predict_proba([chiffres+ [cluster]])
    return foret.predict_proba([chiffres+ [cluster]])


def chargement():

    '''
    Permet le chargement des 2 modèles permettant la prédiction
    '''
    if os.path.exists('model.joblib') and os.path.exists('clustering.joblib'):
        return (load('model.joblib'), load('clustering.joblib'))


def description(foret, clustering):

    '''
    Décrit les 2 modèles permettant les prédictions
    '''
    dico_foret = foret.get_params()
    dico_foret['metric accuracy'] = 'Accuracy standard'
    dico_foret['description'] = 'Ceci est une description des paramètres de la foret aléatoire.'
    dico_clustering = clustering.get_params()
    dico_clustering['description'] = 'Ceci est une description des paramètres de l\'algorithme des kmeans.'

    return (foret.get_params(), clustering.get_params())



def generationChiffres(foret, clustering):

    '''
    Permet de générer une combinaison qui a plus de chance de gagner que les autres.
    '''

    combinaisons = generationDataPerdante(10)[["N1","N2","N3","N4","N5","E1","E2"]]
    combinaisons['Cluster'] = clustering.predict(combinaisons)
    probas = foret.predict_proba(combinaisons)
    return combinaisons.loc[np.argmax(probas[:,1])].to_dict()


def ajoutEtEntrainement(new_data):
    ligne_ajout = pd.DataFrame.from_dict(new_data, orient='index').T
    original = creationData(original_modif = True)
    
    pd.concat([original, ligne_ajout]).to_csv(PATH_TO_DATA_FILE,index = False)

    return "Donnée ajoutée"