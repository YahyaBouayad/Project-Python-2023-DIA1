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

option_choisie = st.selectbox("Choisissez une option", drogues_col)
options_restantes = [option for option in drogues_col if option != option_choisie]

for option in options_restantes:
    st.slider(f"Slider pour {option}", 0, 6, 1)
