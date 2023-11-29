import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fonction_project import *
#%% Traitement fichier 
data=lecture_data()
pers_data=pre_data(data)



#%% Debut Page 

# TITRE : 
st.title("Data Analyse for the drug data ")
st.markdown("Project made by \r Yahya \r Hamza \r Joshua ")

if st.checkbox("Show raw data"):
    st.subheader('Drug Raw data used')
    st.write(data)
    st.markdown(
        """
        The data here is made up of 31 columns, the first represents characteristics about the person such as age, education and others. Then there are the psychological characteristics of the person, and finally if he has consumed certain types of drugs ranging from chocolate to cocaine.
We have 1800 individuals, which gives us a good data set to study.

        """
        )
st.markdown("We cleaned the dataset and we were able to assign this representation to each value, for this we used the explanations in the description of the dataset.")
if st.checkbox("Show cleaned data"):
    st.subheader('Cleaned data used')
    st.write(pers_data)
    st.markdown("Ther was no null data, so we had nothing to remove, we just removed the line where someone said that they have used Semeron. Wich is a fake drug used to get raid of the liar")


st.title("Heat map of the data")
st.markdown("We plot this heat map so we can visualise the correlation between data, it can give us idea of graphic we can make.")
st.pyplot(heat_map_data(data))

st.title("... plot for analyse the simple data : ")
st.markdown("Proportion of data for every Personnel information")

selected_column = st.selectbox('Choisissez une colonne pour l\'analyse', info_col)
if st.button('Afficher le graphique'):
    fig = plot_proportion_bar(pers_data, selected_column)
    st.pyplot(fig)
