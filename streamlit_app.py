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



#%% Debut Page 

# TITRE : 
st.title("Data Analyse for the drug data ")
st.markdown("Project made by \n Yahya BOUAYAD,\n Hamza HALINE\n et Joshua BORNET")

if st.checkbox("Show raw data"):
    st.subheader('Drug Raw data used')
    st.write(data)
    st.markdown(
        """
        The dataset here is made up of 31 columns, the first represents some global characteristics about the person such as his age, education and others. Then comes some psychological aspects of the person, and the last part of the dataset is about what kind of addiction the person has, from chocolate to cocaine.
We have 1800 individuals, which gives us a good data set to study.

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
#################################################################
st.title("General analysis of the surveyed people : ")
st.markdown("So let's take a look at the general information we talked about just before.")
####################################################################

selected_column = st.selectbox('Choose a column to plot', info_col)
tab1, tab2 = st.tabs(["Bar exemple", "Pie chart Exemple"])
with tab1:
    fig = plot_proportion_bar(pers_data, selected_column)
    st.pyplot(fig)
with tab2:
    st.plotly_chart(plot_pie_chart(pers_data,selected_column),theme="streamlit", use_container_width=True)
   
    

##############################################################################

st.markdown("2-columns graph")
selected_column_a = st.selectbox('Choose the first column',drogues_col )
selected_column_b = st.selectbox('Choose the second column',info_col )
# Affichage du graphique à barres en fonction des colonnes sélectionnées
if st.button('Show graph'):
    fig2 = plot_proportion_bar2(pers_data, selected_column_a, selected_column_b)
    st.pyplot(fig2)

#######################################################################################
st.markdown("User count for differents drugs")
st.pyplot(plot_user_counts_per_drug_combined(data,drogues_col))
st.markdown("And for more detail :")
selected_column_details = st.selectbox('Choose the second column',drogues_col )
on = st.toggle('Show graph')

if on:
    st.pyplot(plot_proportion_bar_drug(data,selected_column_details))

#######################################################################################
mon_dictionnaire = {
    "Individu": info_col,
    "Caractere": caract_col,
    "Drogues": drogues_col
}

st.markdown("Correlation matrix analysis based on a group of column")
cle_choisie = st.selectbox("Choose a key :", list(mon_dictionnaire.keys()))
st.pyplot(plot_correlation_matrix(data,mon_dictionnaire[cle_choisie]))
################################################################################


#######################################################################################
st.title("Analyse approfondie :")
st.markdown("Analyse des caracteres d'une personne moyenne qui consomme une drogue")
selected_column_drogue = st.selectbox('Choisissez une colonne pour l\'analyse', drogues_col )

col1, col2 = st.columns(2)
# Afficher les graphiques dans les colonnes correspondantes
with col1:
    st.pyplot(radar_chart_consommation_drogue(pers_data,selected_column_drogue,caract_col,True))
with col2:
    st.pyplot(radar_chart_consommation_drogue(pers_data,selected_column_drogue,caract_col,False))

st.markdown("Pour une vue d'ensemble:")
st.plotly_chart(profil_drogue_radar(pers_data,drogues_col,caract_col),theme="streamlit", use_container_width=True)

###########################################################################################################################

st.subheader("Tendance de consomation selon la tranche d'age :sunglasses:")
col1, col2 = st.columns([1, 2])
with col1:
    selected_column_drug = st.selectbox('Choose a drug :', drogues_col )
with col2:
    st.plotly_chart(plot_drug_use_trends_by_age_pers_data(pers_data,selected_column_drug),theme="streamlit", use_container_width=True)

###################################################
st.markdown("Repartiton du type de drogue par rapport au etude effectuer")
options_disponibles = drogues_col
multi_options_choisies = st.multiselect("Choisissez vos options :", options_disponibles)
st.plotly_chart(plot_education_level_sunburst(pers_data,multi_options_choisies),theme="streamlit",use_container_width=True)

#######################################################
all_dat_combinaision=pre_combinaison(pers_data,drogues_illégales)
st.markdown("Combinaison de drogue : ")
st.write(all_dat_combinaision)
st.plotly_chart(frequence_combinaison(all_dat_combinaision),theme="streamlit",use_container_width=True)
#######################################################

