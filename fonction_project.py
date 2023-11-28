import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def heat_map_data(data):
    corr = data.corr()
    plt.figure(figsize=(20,10))
    sns.heatmap(corr, annot=True, vmin=-1)
    return plt
