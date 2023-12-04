import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fonction_project import *

import warnings
warnings.filterwarnings('ignore')
#%% Traitement fichier 
info_col = [
    'Age', 
    'Genre', 
    'Education', 
    'Pays',
    'Ethnie',
]

caract_col = [
    'Névrotique',
    'Extraverti',
    'Ouvert à l\'expérience',
    'Amical',
    'Consciencieux',
    'Imuplsif',
    'En recherche de sensations'
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

if 'data' not in st.session_state:
    st.session_state['data'] = data
if 'pers_data' not in st.session_state:
    st.session_state['pers_data'] = pers_data
if 'info_col' not in st.session_state:
    st.session_state['info_col'] = info_col
if 'caract_col' not in st.session_state:
    st.session_state['caract_col'] = caract_col
if 'drogues_col' not in st.session_state:
    st.session_state['drogues_col'] = drogues_col
if 'drogues_autorisées' not in st.session_state:
    st.session_state['drogues_autorisées'] = drogues_autorisées
if 'drogues_illégales' not in st.session_state:
    st.session_state['drogues_illégales'] = drogues_illégales
    
from fonction_ml import *
#%% Debut Page 

# TITRE : 
st.title("Data Analysis for the drug data ")
st.markdown("Project made by \n Yahya BOUAYAD,\n Hamza HALINE\n et Joshua BORNET")
st.divider()

"""

Presentation of the raw data:
We chose the dataset on the study done on the drug available on this link:

"""
st.link_button("Go to link ", "https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified")
if st.checkbox("Show raw data"):
    st.subheader('Addiction Raw data used')
    st.write(data)
    st.markdown(
        """
        The dataset here is made up of 31 columns, the first represents some global characteristics about the person such as his age, education and others. Then comes some psychological aspects of the person, and the last part of the dataset is about what kind of addiction the person has, from chocolate to cocaine.
We have 1800 individuals, which gives us a good dataset to study.

        """
        )
