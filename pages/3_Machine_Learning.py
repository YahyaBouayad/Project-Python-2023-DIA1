import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fonction_project import *
from fonction_ml import *

import warnings
warnings.filterwarnings('ignore')
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


data=st.session_state["data"]
drogues_col=st.session_state['drogues_col'] 

models = { ' Support Vector Machines': SVC(),
            ' Decision Tree Classifier': DecisionTreeClassifier(),
            'Random Forest Classifier': RandomForestClassifier(),}

st.title("Machine Learning work:")
st.markdown("The objectif of our Machin learning training is to determine if we can find is someone use or used one of the drugs using her Information and his caracter. ")
st.divider()

name_features=['Age','Genre', 'Education','Névrotique', 'Extraverti', 'Ouvert à l\'expérience', 'Amical', 'Consciencieux', 'Imuplsif', 'En recherche de sensations']
name_target=st.selectbox('Choose a drug to train', drogues_col)
features = data[name_features]  # Ajoutez ici d'autres caractéristiques pertinentes
target = data[name_target]

st.markdown("The data features are :")
"""
'Age','Genre', 'Education','Névrotique', 'Extraverti', 'Ouvert à l\'expérience', 'Amical', 'Consciencieux', 'Imuplsif', 'En recherche de sensations'
"""
st.markdown("The data Target is :")
st.write(name_target)




tab1, tab2, tab3 = st.tabs(["Support Vector Machines", "Decision Tree Classifier","Random Forest Classifier"])

with tab1:
    plt_1,y_pred_1,report_1,accuracy_1=prediction_training(features,target,SVC())
    st.write(f'Accuracy: {accuracy_1 * 100:.2f}%')
    st.pyplot(plt_1)

    
with tab2:
    plt_2,y_pred_2,report_2,accuracy_2=prediction_training(features,target,DecisionTreeClassifier())
    st.write(f'Accuracy: {accuracy_2 * 100:.2f}%')
    st.pyplot(plt_2)
    
with tab3:
    plt_3,y_pred_3,report_3,accuracy_3=prediction_training(features,target,RandomForestClassifier())
    st.write(f'Accuracy: {accuracy_3 * 100:.2f}%')
    st.pyplot(plt_3)
####################################################################################################################
st.divider()
st.markdown("Observation a inserer + passage au grid_search")

on_grid = st.toggle('Activate Automatisation search')
if on_grid :
    param_grid = {
        'C': [0.1, 1, 10, 100],             # Paramètre de régularisation
        'gamma': [1, 0.1, 0.01, 0.001],     # Coefficient du noyau pour 'rbf', 'poly' et 'sigmoid'
        'kernel': ['rbf', 'poly', 'sigmoid'] # Type de noyau
    }
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Regularisation parametre :")
        c_param = st.radio(
        "What is your C parameter",
        [0.1, 1, 10,100])
    
    with col2:
        st.write("Coefficient du noyau :")
        g_param = st.radio(
        "What is your Gamma parameter",
        [1, 0.1, 0.01, 0.001])
    with col3:
        st.write("Type du noyau :")
        k_param = st.radio(
        "What is your kernel",
        ['rbf', 'poly', 'sigmoid'])
    param_grid = {
        'C':[c_param] ,             # Paramètre de régularisation
        'gamma':[g_param] ,     # Coefficient du noyau pour 'rbf', 'poly' et 'sigmoid'
        'kernel': [k_param] # Type de noyau
    }
plt_g,y_pred_g,accuracy_g=grid_search_ml(features,target,param_grid)
st.write(f'Accuracy: {accuracy_g * 100:.2f}%')
st.pyplot(plt_g)
####################################################################

st.divider()
st.markdown("Nouveau nettoyage des donnes :")
name_target2=st.selectbox('Choose a drug to train', drogues_col)

data_cannabis = prepare_dataset_for_drug_prediction(data, name_target2)
st.write(data_cannabis.head(10))

features_cannabis = data_cannabis.drop(["Target","Consommation de Semeron"], axis=1)
target_cannabis = data_cannabis['Target']

tab4, tab5, tab6 = st.tabs(["Support Vector Machines", "Decision Tree Classifier","Random Forest Classifier"])

with tab4:
    plt_4,y_pred_4,report_4,accuracy_4=prediction_training(features_cannabis,target_cannabis,SVC())
    st.write(f'Accuracy: {accuracy_4 * 100:.2f}%')
    st.pyplot(plt_4)

    
with tab5:
    plt_2,y_pred_5,report_5,accuracy_5=prediction_training(features_cannabis,target_cannabis,DecisionTreeClassifier())
    st.write(f'Accuracy: {accuracy_5 * 100:.2f}%')
    st.pyplot(plt_5)
    
with tab6:
    plt_6,y_pred_6,report_6,accuracy_6=prediction_training(features_cannabis,target_cannabis,RandomForestClassifier())
    st.write(f'Accuracy: {accuracy_6 * 100:.2f}%')
    st.pyplot(plt_6)
