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

def plot_proportion_bar(dataset, column):
    # Calculer la proportion de la population par tranche d'âge
    proportion_par_tranche = dataset[column].value_counts(normalize=True).sort_index()

    # Définir des couleurs pour chaque tranche d'âge
    couleurs = ['skyblue', 'lightcoral', 'gold', 'lightgreen', 'lightsalmon', 'mediumpurple']

    # Créer le graphique à barres avec des couleurs différentes pour chaque barre
    plt.figure(figsize=(10, 6))
    proportion_par_tranche.plot(kind='bar', color=couleurs)
    plt.title(f'Proportion de la population par {column}', fontsize=16)
    plt.xlabel(f'{column}', fontsize=14)
    plt.ylabel('Proportion de la population', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    return plt

def plot_proportion_bar2(dataset, column_a, column_b):
    # Calculer la proportion par colonne_b
    proportion_par_colonne = dataset.groupby(column_b)[column_a].mean().sort_index()

    # Créer le graphique à barres
    plt.figure(figsize=(10, 6))
    proportion_par_colonne.plot(kind='bar', color='skyblue')
    plt.title(f'Moyenne de {column_a} par {column_b}', fontsize=16)
    plt.xlabel(f'{column_b}', fontsize=14)
    plt.ylabel(f'Moyenne de {column_a}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    return plt

def plot_pie_chart(dataset, column):
    # Calculer les pourcentages pour chaque catégorie
    pourcentages = dataset[column].value_counts(normalize=True) * 100

    # Créer le diagramme en camembert
    fig = px.pie(pourcentages, values=pourcentages, names=pourcentages.index, 
                 title=f'Répartition de la population par {column}',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Afficher le diagramme en camembert
    return fig

def plot_correlation_matrix(dataset, columns):
    # Sélectionner les colonnes spécifiées dans la liste
    selected_columns = dataset[columns]

    # Calculer la matrice de corrélation
    correlation_matrix = selected_columns.corr()
    
    # Créer la heatmap avec Seaborn
    plt.figure(figsize=(10, 8))
    
    # Utiliser des chiffres avec moins de décimales et ajuster la taille des chiffres
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f", annot_kws={"size": 8})
    
    plt.title('Matrice de Corrélation', fontsize=16)
    return plt

