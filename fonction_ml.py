import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore')
drogues_col=st.session_state['drogues_col']


def prediction_training(features,target,model):
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Autres métriques de performance
    report = classification_report(y_test, y_pred)
    print(report)
    plt.clf()
    sns.distplot(y_pred,hist=False,color='r',label = 'Predicted Values')
    sns.distplot(y_test,hist=False,color='b',label = 'Actual Values')
    plt.title('Actual vs predicted values',fontsize =16)
    plt.xlabel('Values',fontsize=12)
    plt.ylabel('Frequency',fontsize =12)
    plt.legend(loc='upper left',fontsize=13)
    
    return plt,y_pred,report,accuracy


def grid_search_ml(features,target,param_grid):
    '''
    param_grid = {
        'C': [0.1, 1, 10, 100],             # Paramètre de régularisation
        'gamma': [1, 0.1, 0.01, 0.001],     # Coefficient du noyau pour 'rbf', 'poly' et 'sigmoid'
        'kernel': ['rbf', 'poly', 'sigmoid'] # Type de noyau
    }
    '''

    # Créer un modèle de base
    dt = SVC()

    # Instancier GridSearchCV
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    # Exécuter la recherche
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres
    print("Meilleurs paramètres : ", grid_search.best_params_)

    # Évaluer le meilleur modèle trouvé
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    plt.clf()
    sns.distplot(y_pred,hist=False,color='r',label = 'Predicted Values')
    sns.distplot(y_test,hist=False,color='b',label = 'Actual Values')
    plt.title('Actual vs predicted values',fontsize =16)
    plt.xlabel('Values',fontsize=12)
    plt.ylabel('Frequency',fontsize =12)
    plt.legend(loc='upper left',fontsize=13)
    return plt,y_pred,accuracy

def prepare_dataset_for_drug_prediction(dataframe, drug_name):
    
    # Copying the dataframe to avoid modifying the original data
    df = dataframe.copy()

    # Check if the drug name is in the dataframe
    if drug_name not in df.columns:
        raise ValueError(f"The drug '{drug_name}' is not found in the dataframe.")


    df['Target'] = df[drug_name].apply(lambda x: 1 if x > 0 else 0)

    # Removing the original drug column
    df.drop(drug_name, axis=1, inplace=True)
    return df

def prediction_training_f(data,data_final,model,drug):
    data=prepare_dataset_for_drug_prediction(data,drug)
    X = data_cannabis.drop(["Target","Consommation de Semeron"], axis=1)
    Y = data_cannabis['Target']
    
    model.fit(X, Y)
    y_pred = model.predict(data_final)
    return y_pred
    
def process_user_input(data,age, genre, education, neuroticisme, extraversion, exp, amicalite, conscience, impulsivite, recherche, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18):

    drogues_col = ['Consommation d\'alcool', 'Consommation d\'amphétamines', 'Consommation d\'amyl', 'Consommation de benzodiazepine', 'Consommation de café', 'Consommation de cannabis', 'Consommation de chocolat', 'Consommation de cocaïne', 'Consommation de crack', 'Consommation d\'ecstasy', 'Consommation d\'héroïne', 'Consommation de ketamine', 'Consommation de drogues légales', 'Consommation de LSD', 'Consommation de meth', 'Consommation de champignons magiques', 'Consommation de nicotine', 'Consommation de Semeron', 'Consommation de VSA']
    
    # Créez un DataFrame utilisateur
    user_data = pd.DataFrame(data=[[0] * len(data.columns)], columns=data.columns)
    
    # Remplacez les valeurs dans le DataFrame utilisateur
    user_data.loc[0] = [age, genre, education, 0.96082, -0.31685, neuroticisme, extraversion, exp, amicalite, conscience,
                        impulsivite, recherche] + [0] * len(drogues_col)

    # Colonnes de caractéristiques
    caract_col = ['Névrotique', 'Extraverti', 'Ouvert à l\'expérience',
                'Amical', 'Consciencieux']

    st.write("Colonnes disponibles dans user_data :", user_data.columns)
    user_data[caract_col] = user_data[caract_col].astype(float).apply(lambda x: x * 6)

    neuro_mapping = {
        0: -3.46436, 6: -3.46436, 12: -3.46436, 13: -3.15735, 14: -2.75696, 15: -2.52197, 16: -2.42317,
        17: -2.34360, 18: -2.21844, 19: -2.05048, 20: -1.86962, 21: -1.69163,
        22: -1.55078, 23: -1.43907, 24: -1.32828, 25: -1.19430, 26: -1.05308,
        27: -0.92104, 28: -0.79151, 29: -0.67825, 30: -0.58016, 31: -0.46725,
        32: -0.34799, 33: -0.24649, 34: -0.14882, 35: 0.04257, 36: 0.13606,
        37: 0.22393, 38: 0.31287, 39: 0.41667, 40: 0.52135, 41: 0.62967,
        42: 0.73545, 43: 0.82562, 44: 0.91093, 45: 1.02119, 46: 1.13281,
        47: 1.23461, 48: 1.37297, 49: 1.49158, 50: 1.60383, 51: 1.72012,
        52: 1.83990, 53: 1.98437, 54: 2.12700, 55: 2.28554, 56: 2.46262,
        57: 2.61139, 58: 2.82196, 59: 3.27393, 60: 3.27393
    }

    user_data['Neuroticisme'] = [neuro_mapping[i] for i in user_data['Neuroticisme']]

    extraversion_mapping = {
        0: -3.27393, 6: -3.27393, 12: -3.27393, 16: -3.27393, 18: -3.00537, 19: -2.72827, 20: -2.53830, 21: -2.44904,
        22: -2.32338, 23: -2.21069, 24: -2.11437, 25: -2.03972, 26: -1.92173,
        27: -1.76250, 28: -1.63340, 29: -1.50796, 30: -1.37639, 31: -1.23177,
        32: -1.09207, 33: -0.94779, 34: -0.80615, 35: -0.69509, 36: -0.57545,
        37: -0.43999, 38: -0.30033, 39: -0.15487, 40: 0.00332, 41: 0.16767,
        42: 0.32197, 43: 0.47617, 44: 0.63779, 45: 0.80523, 46: 0.96248,
        47: 1.11406, 48: 1.28610, 49: 1.45421, 50: 1.58487, 51: 1.74091,
        52: 1.93886, 53: 2.12700, 54: 2.32338, 55: 2.57309, 56: 2.85950,
        58: 3.00537, 59: 3.27393, 60: 3.27393
    }

    user_data['Extraversion'] = [extraversion_mapping[i] for i in user_data['Extraversion']]

    ouverture_mapping = {
        0: -3.27393, 6: -3.27393, 12: -3.27393, 24: -3.27393, 26: -2.85950, 28: -2.63199, 29: -2.39883, 30: -2.21069,
        31: -2.09015, 32: -1.97495, 33: -1.82919, 34: -1.68062, 35: -1.55521,
        36: -1.42424, 37: -1.27553, 38: -1.11902, 39: -0.97631, 40: -0.84732,
        41: -0.71727, 42: -0.58331, 43: -0.45174, 44: -0.31776, 45: -0.17779,
        46: -0.01928, 47: 0.14143, 48: 0.29338, 49: 0.44585, 50: 0.58331,
        51: 0.72330, 52: 0.88309, 53: 1.06238, 54: 1.24033, 55: 1.43533,
        56: 1.65653, 57: 1.88511, 58: 2.15324, 59: 2.44904, 60: 2.90161,
    }

    user_data['Ouverture à l\'expérience'] = [ouverture_mapping[i] for i in user_data['Ouverture à l\'expérience']]

    amicalite_mapping = {
        0: -3.46436, 6: -3.46436, 12: -3.46436, 16: -3.15735, 18: -3.00537, 23: -2.90161, 24: -2.78793,
        25: -2.70172, 26: -2.53830, 27: -2.35413, 28: -2.21844, 29: -2.07848,
        30: -1.92595, 31: -1.77200, 32: -1.62090, 33: -1.47955, 34: -1.34289,
        35: -1.21213, 36: -1.07533, 37: -0.91699, 38: -0.76096, 39: -0.60633,
        40: -0.45321, 41: -0.30172, 42: -0.15487, 43: -0.01729, 44: 0.13136,
        45: 0.28783, 46: 0.43852, 47: 0.59042, 48: 0.76096, 49: 0.94156,
        50: 1.11406, 51: 1.28610, 52: 1.45039, 53: 1.61108, 54: 1.81866,
        55: 2.03972, 56: 2.23427, 57: 2.46262, 58: 2.75696, 59: 3.15735,
        60: 3.46436,
    }

    user_data['Amicalité'] = [amicalite_mapping[i] for i in user_data['Amicalité']]

    conscience_mapping = {
        0: -3.46436, 6: -3.46436, 12: -3.46436, 18: -3.46436, 17: -3.46436, 19: -3.15735, 20: -2.90161, 21: -2.72827, 22: -2.57309,
        23: -2.42317, 24: -2.30408, 25: -2.18109, 26: -2.04506, 27: -1.92173,
        28: -1.78169, 29: -1.64101, 30: -1.51840, 31: -1.38502, 32: -1.25773,
        33: -1.13788, 34: -1.01450, 35: -0.89891, 36: -0.78155, 37: -0.65253,
        38: -0.52745, 39: -0.40581, 40: -0.27607, 41: -0.14277, 42: -0.00665,
        43: 0.12331, 44: 0.25953, 45: 0.41594, 46: 0.58489, 47: 0.7583,
        48: 0.93949, 49: 1.13407, 50: 1.30612, 51: 1.46191, 52: 1.63088,
        53: 1.81175, 54: 2.04506, 55: 2.33337, 56: 2.63199, 57: 3.00537,
        59: 3.46436, 60: 3.46436
    }

    user_data['Conscience'] = [conscience_mapping[i] for i in user_data['Conscience']]

    sensations = {
        -2.07848: 0,
        -1.54858: 1,
        -1.18084: 2,
        -0.84637: 3,
        -0.52593: 4,
        -0.21575: 5,
        0.07987: 6,
        0.40148: 7,
        0.76540: 8,
        1.22470: 9,
        1.92173: 10
    }

    # Mapping inverse pour Recherche de sensations
    inverse_sensations = {v: k for k, v in sensations.items()}
    user_data['Recherche de sensations'] = user_data['Recherche de sensations'].map(inverse_sensations)

    # Mapping inverse pour Impulsivité
    impulsivite_mapping = {
        -2.55524: 0, -1.37983: 1, -0.71126: 2, -0.21712: 3,
        0.19268: 4, 0.52975: 5, 0.88113: 6, 1.29221: 7,
        1.86203: 8, 2.90161: 9
    }
    inverse_impulsivite = {v: k for k, v in impulsivite_mapping.items()}
    user_data['Impulsivité'] = user_data['Impulsivité'].map(inverse_impulsivite)

    # Mapping pour 'Age'
    age_mapping = {
        '18-24': -0.95197,
        '25-34': -0.07854,
        '35-44': 0.49788,
        '45-54': 1.09449,
        '55-64': 1.82213,
        '65+': 2.59171
    }
    user_data['Age'] = user_data['Age'].map(age_mapping)

    # Mapping pour 'Education'
    education_mapping = {
        "A quitté l'école avant 16 ans": -2.43591,
        "A quitté l'école à 16 ans": -1.73790,
        "A quitté l'école à 17 ans": -1.43719,
        "A quitté l'école à 18 ans": -1.22751,
        "Universitaire, sans diplôme": -0.61113,
        "Certificat / diplôme professionnel": -0.05921,
        "Diplôme universitaire": 0.45468,
        "Master": 1.16365,
        "Doctorat": 1.98437
    }
    user_data['Education'] = user_data['Education'].map(education_mapping)

    # Assigner toutes les valeurs d'Ethnie à -0.31685
    user_data['Ethnie'] = -0.31685

    # Assigner toutes les valeurs de Pays à 0.96082
    user_data['Pays'] = 0.96082

    # Mapping pour 'Genre'
    genre_mapping = {
        'Femme': 0.48246,
        'Homme': -0.48246
    }
    user_data['Genre'] = user_data['Genre'].map(genre_mapping)

    droguescol = [
        'Consommation d\'alcool',
        'Consommation d\'amphétamines',
        'Consommation d\'amyl',
        'Consommation de benzodiazepine',
        'Consommation de café',
        'Consommation de cannabis',
        'Consommation de chocolat',
        'Consommation de cocaïne',
        'Consommation de crack',
        'Consommation d\'ecstasy',
        'Consommation d\'héroïne',
        'Consommation de ketamine',
        'Consommation de drogues légales',
        'Consommation de LSD',
        'Consommation de meth',
        'Consommation de champignons magiques',
        'Consommation de nicotine',
        'Consommation de Semeron',
        'Consommation de VSA'
    ]
    len(droguescol)
    liste1 = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, 0, d18]
    for i in range (19):
        user_data[droguescol[i]] = liste1[i]

    return user_data
