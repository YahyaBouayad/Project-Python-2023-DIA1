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

st.write(data)

features = data[['Age','Genre', 'Education','Névrotique', 'Extraverti', 'Ouvert à l\'expérience', 'Amical', 'Consciencieux', 'Imuplsif', 'En recherche de sensations']]  # Ajoutez ici d'autres caractéristiques pertinentes
target = data['Consommation de cannabis']
prediction_training(features,target,SVC())

st.title("Machine Learning work:")
st.markdown("The objectif of our Machin learning training is to determine if we can find is someone use or used one of the drugs using her Information and his caracter. ")

