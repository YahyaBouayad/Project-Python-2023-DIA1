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
