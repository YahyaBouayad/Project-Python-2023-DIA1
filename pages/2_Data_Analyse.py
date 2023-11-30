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

data=st.session_state["data"]
drogues_col=st.session_state['drogues_col'] 
pers_data=st.session_state["pers_data"]
caract_col=st.session_state["caract_col"]
st.title("Let's get in the details :")
st.markdown("Radar graph that highlights the characteristics of the average person that consumes :")
selected_column_drogue = st.selectbox('Choose a column', drogues_col )

col1, col2 = st.columns(2)
# Afficher les graphiques dans les colonnes correspondantes
with col1:
    st.pyplot(radar_chart_consommation_drogue(pers_data,selected_column_drogue,caract_col,True))
with col2:
    st.pyplot(radar_chart_consommation_drogue(pers_data,selected_column_drogue,caract_col,False))

st.markdown("Global view :")
st.plotly_chart(profil_drogue_radar(pers_data,drogues_col,caract_col),theme="streamlit", use_container_width=True)
st.divider()
###########################################################################################################################

st.subheader("Consumption trend by age group :")
col1, col2 = st.columns([1, 2])
with col1:
    selected_column_drug = st.selectbox('Choose a column :', drogues_col )
with col2:
    st.plotly_chart(plot_drug_use_trends_by_age_pers_data(pers_data,selected_column_drug),theme="streamlit", use_container_width=True)
st.divider()
###################################################
st.markdown("Consumption trend by study level :")
options_disponibles = drogues_col
multi_options_choisies = st.multiselect("Choose your degree :", options_disponibles)
st.plotly_chart(plot_education_level_sunburst(pers_data,multi_options_choisies),theme="streamlit",use_container_width=True)
st.divider()
#######################################################
all_dat_combinaision=pre_combinaison(pers_data,drogues_illégales)
st.markdown("Addiction combination : ")
st.write(all_dat_combinaision)
st.plotly_chart(frequence_combinaison(all_dat_combinaision),theme="streamlit",use_container_width=True)
#######################################################
