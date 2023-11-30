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


info_col = [
    'Age', 
    'Genre', 
    'Education', 
    'Pays',
    'Ethnie',
]

caract_col = [
    'Neuroticisme',
    'Extraversion',
    'Ouverture à l\'expérience',
    'Amicalité',
    'Conscience',
    'Impulsivité',
    'Recherche de sensations'
]

personne_col = info_col + caract_col

drogues_col = [
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

drogues_autorisées = ['Consommation d\'alcool', 'Consommation de café', 'Consommation de chocolat', 'Consommation de nicotine']
drogues_illégales = [i for i in drogues_col if i not in drogues_autorisées]
col = personne_col + drogues_col

data=lecture_data()
pers_data=pre_data(data,col,drogues_col,drogues_illégales)
st.write(data)
