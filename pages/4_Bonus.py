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
st.title("Bonus  :")
st.markdown("For the bonus we created a little game, where you can put your data and see if you taken a drug or not.")
st.divider()

age = st.radio(
    "What's your age",
    ["18-24", "25-34", "35-44","45-54","55-64","65+"],
    index=None,
    horizontal=True
)

genre = st.radio(
    "What's your gender",
    ["Homme", "Femme"],
    index=None,
)
education = st.radio(
    "What's your Education",
    ["A quitté l\'école avant 16 ans", "A quitté l\'école à 16 ans","A quitté l\'école à 17 ans","A quitté l\'école à 18 ans","Universitaire, sans diplôme","Certificat / diplôme professionnel","Diplôme universitaire","Master","Doctorat"],
    index=None,
    horizontal=True
)
neuroticisme = st.slider('Neuroticisme', 0, 10, 1)
extraversion = st.slider('Extraversion', 0, 10, 1)
exp = st.slider('Ouverture à l\'expérience', 0, 10, 1)
amicalite = st.slider('Amicalité', 0, 10, 1)
conscience = st.slider('Conscience', 0, 10, 1)
impulsivite = st.slider('Impulsivité', 0, 10, 1)
recherche = st.slider('Recherche de sensations', 0, 10, 1)
st.markdown("drogue")
st.divider()

option_choisie = st.selectbox("Choisissez une drogue que vous voullez tester:", drogues_col)
options_restantes = [option for option in drogues_col if option != option_choisie]

valeurs_sliders = {}

for option in options_restantes:
    valeur_slider = st.slider(f"Slider pour {option}", 0, 6, 1)
    valeurs_sliders[option] = valeur_slider
valeurs_sliders[option_choisie]=0


if st.button('Lancement du code :'):
    data_final=process_user_input(data,age, genre, education, neuroticisme, extraversion, exp, amicalite, conscience, impulsivite, recherche,valeurs_sliders['Consommation d\'alcool'],valeurs_sliders['Consommation d\'amphétamines'],valeurs_sliders['Consommation d\'amyl'],valeurs_sliders['Consommation de benzodiazepine'],valeurs_sliders['Consommation de café'],valeurs_sliders['Consommation de cannabis'],valeurs_sliders['Consommation de chocolat'],valeurs_sliders['Consommation de cocaïne'],valeurs_sliders['Consommation de crack'],valeurs_sliders['Consommation d\'ecstasy'],valeurs_sliders['Consommation d\'héroïne'],valeurs_sliders['Consommation de ketamine'],valeurs_sliders['Consommation de drogues légales'],valeurs_sliders['Consommation de LSD'],valeurs_sliders['Consommation de meth'],valeurs_sliders['Consommation de champignons magiques'],valeurs_sliders['Consommation de nicotine'],valeurs_sliders['Consommation de VSA'])
    data_cannabis=prepare_dataset_for_drug_prediction(data, option_choisie )
    features_cannabis = data_cannabis.drop(["Target","Consommation de Semeron"], axis=1)
    target_cannabis = data_cannabis['Target']
    st.write(data_final)
    st.write(features_cannabis.head(5))
    model_train=prediction_training_f(features_cannabis,target_cannabis,SVC())
    st.write(model_train.predict(data_final))
    
    
