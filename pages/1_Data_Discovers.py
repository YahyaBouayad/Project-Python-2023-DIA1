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

#%% Debut Page 

# TITRE : 
st.title("Data Discover :")

st.divider()
if st.checkbox("Show raw data"):
    st.subheader('Addiction Raw data used')
    st.write(data)
    st.markdown(
        """
        The dataset here is made up of 31 columns, the first represents some global characteristics about the person such as his age, education and others. Then comes some psychological aspects of the person, and the last part of the dataset is about what kind of addiction the person has, from chocolate to cocaine.
We have 1800 individuals, which gives us a good dataset to study.

        """
        )
st.markdown("We read the documentation and thus, we were able to rename the columns and modify them to make their content readable by everybody.")
if st.checkbox("Show cleaned data"):
    st.subheader('Cleaned data used')
    st.write(pers_data)
    st.markdown("Ther was no null data, so we had nothing to remove. However, we managed to remove all of the users that tried to trick the study by saying they were Semerons consumers, whereas Semeron is a fake drug. In total, they were barely ten.")


st.title("Heat map of the data")
st.markdown("We plot this heat map so we can visualize the correlation between data, it can give us an idea of which kind of graphics we could make.")
st.pyplot(heat_map_data(data))
st.divider()
#################################################################
st.title("General analysis of the surveyed people : ")
st.markdown("So let's take a look at the general information we talked about just before.")
st.divider()
####################################################################

selected_column = st.selectbox('Choose a column to plot', info_col)
tab1, tab2 = st.tabs(["Bar exemple", "Pie chart Exemple"])
with tab1:
    fig = plot_proportion_bar(pers_data, selected_column)
    st.pyplot(fig)
with tab2:
    st.plotly_chart(plot_pie_chart(pers_data,selected_column),theme="streamlit", use_container_width=True)
   
    
st.divider()
##############################################################################

st.markdown("2-columns graph")
selected_column_a = st.selectbox('Choose the first column',drogues_col )
selected_column_b = st.selectbox('Choose the second column',info_col )
# Affichage du graphique à barres en fonction des colonnes sélectionnées
if st.button('Show graph'):
    fig2 = plot_proportion_bar2(pers_data, selected_column_a, selected_column_b)
    st.pyplot(fig2)

st.divider()
#######################################################################################
st.markdown("User count for differents addictions")
st.pyplot(plot_user_counts_per_drug_combined(data,drogues_col))
st.markdown("And for more detail :")
selected_column_details = st.selectbox('Choose the column',drogues_col )
on = st.toggle('Show graph')

if on:
    st.pyplot(plot_proportion_bar_drug(data,selected_column_details))
st.divider()
#######################################################################################
mon_dictionnaire = {
    "Individu": info_col,
    "Caractere": caract_col,
    "Drogues": drogues_col
}

st.markdown("Correlation matrix analysis based on a group of column")
cle_choisie = st.selectbox("Choose a key :", list(mon_dictionnaire.keys()))
st.pyplot(plot_correlation_matrix(data,mon_dictionnaire[cle_choisie]))
st.divider()


