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
    ["18-24", "25-34", "35-44","45-54","55-64","64+"],
    index=None,
)
st.write(age)
