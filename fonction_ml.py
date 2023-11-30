import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
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
import warnings
warnings.filterwarnings('ignore')

def prediction_training(features,target,model):
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Autres métriques de performance
    report = classification_report(y_test, y_pred)
    print(report)
    plt.clf()
    sns.distplot(y_pred,hist=False,color='r',label = 'Predicted Values')
    sns.distplot(y_test,hist=False,color='b',label = 'Actual Values')
    plt.title('Actual vs predicted values',fontsize =16)
    plt.xlabel('Values',fontsize=12)
    plt.ylabel('Frequency',fontsize =12)
    plt.legend(loc='upper left',fontsize=13)
    
    return plt,y_pred,report,accuracy

