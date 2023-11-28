import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
#%% Traitement fichier 
fichier_data = 'drug_consumption.data'
data = pd.read_csv(fichier_data, delimiter=',', header=0)
noms_colonnes = ['ID', 'Age', 'Genre', 'Education', 'Pays', 'Ethnie', 'Neuroticisme', 'Extraversion', 'Ouverture à l\'expérience', 'Amicalité', 'Conscience', 'Impulsivité', 'Recherche de sensations', 'Consommation d\'alcool', 'Consommation d\'amphétamines', 'Consommation d\'amyl', 'Consommation de benzodiazepine', 'Consommation de café', 'Consommation de cannabis', 'Consommation de chocolat', 'Consommation de cocaïne', 'Consommation de crack', 'Consommation d\'ecstasy', 'Consommation d\'héroïne', 'Consommation de ketamine', 'Consommation de drogues légales', 'Consommation de LSD', 'Consommation de meth', 'Consommation de champignons magiques', 'Consommation de nicotine', 'Consommation de Semeron', 'Consommation de VSA']
data.columns = noms_colonnes
data = data.set_index('ID')

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
for i in drogues_col:
    data[i] = data[i].map({'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6})
semerons = data[data['Consommation de Semeron'] != 0]
drogues_col.remove('Consommation de Semeron')
drogues_illégales.remove('Consommation de Semeron')
data.drop(columns = 'Consommation de Semeron')
pers_data = data.copy()
age = ['18-24' if i <= -0.9 else 
       '25-34' if i >= -0.5 and i < 0 else 
       '35-44' if i > 0 and i < 1 else 
       '45-54' if i > 1 and i < 1.5 else 
       '55-64' if i > 1.5 and i < 2 else 
       '65+' 
       for i in pers_data['Age']]

genre = ['Femme' if i > 0 else "Homme" for i in pers_data['Genre']]

education = ['A quitté l\'école avant 16 ans' if i <-2 else 
             'A quitté l\'école à 16 ans' if i > -2 and i < -1.5 else 
             'A quitté l\'école à 17 ans' if i > -1.5 and i < -1.4 else 
             'A quitté l\'école à 18 ans' if i > -1.4 and i < -1 else 
             'Universitaire, sans diplôme' if i > -1 and i < -0.5 else 
             'Certificat / diplôme professionnel' if i > -0.5 and i < 0 else 
             'Diplôme universitaire' if i > 0 and i < 0.5 else 
             'Master' if i > 0.5 and i < 1.5 else 
             'Doctorat' 
             for i in pers_data['Education']]

pays = ['USA' if i < -0.5 else 
           'New Zealand' if i > -0.5 and i < -0.4 else 
           'Other' if i > -0.4 and i < -0.2 else 
           'Australia' if i > -0.2 and i < 0 else 
           'Ireland' if i > 0 and i < 0.23 else 
           'Canada' if i > 0.23 and i < 0.9 else 
           'UK' 
           for i in pers_data['Pays']]

ethnie = ['Black' if i < -1 else 
             'Asian' if i > -1 and i < -0.4 else 
             'White' if i > -0.4 and i < -0.25 else 
             'Mixed-White/Black' if i >= -0.25 and i < 0.11 else 
             'Mixed-White/Asian' if i > 0.12 and i < 1 else 
             'Mixed-Black/Asian' if i > 1.9 else 
             'Other' 
             for i in pers_data['Ethnie']]


pers_data['Age'] = age
pers_data['Genre'] = genre
pers_data['Education'] = education
pers_data['Pays'] = pays
pers_data['Ethnie'] = ethnie

#%% Debut Page 

# TITRE : 
st.title("Data Analyse for the drig data ")
