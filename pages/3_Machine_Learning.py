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
    st.write(f'Accuracy:* {accuracy_1 * 100:.2f}%*')
    st.pyplot(plt_1)

    
with tab2:
    plt_2,y_pred_2,report_2,accuracy_2=prediction_training(features,target,DecisionTreeClassifier())
    st.pyplot(plt_2)
    
with tab3:
    plt_3,y_pred_3,report_3,accuracy_3=prediction_training(features,target,RandomForestClassifier())
    st.pyplot(plt_3)
