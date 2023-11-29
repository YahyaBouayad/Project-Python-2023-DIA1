import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

info_col = [
    'Age', 
    'Genre', 
    'Education', 
    'Pays',
    'Ethnie',
]

caract_col = [
    'Neuroticisme',
    'Extraversion',
    'Ouverture à l\'expérience',
    'Amicalité',
    'Conscience',
    'Impulsivité',
    'Recherche de sensations'
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

def lecture_data():
    fichier_data = 'drug_consumption.data'
    data = pd.read_csv(fichier_data, delimiter=',', header=0)
    noms_colonnes = ['ID', 'Age', 'Genre', 'Education', 'Pays', 'Ethnie', 'Neuroticisme', 'Extraversion', 'Ouverture à l\'expérience', 'Amicalité', 'Conscience', 'Impulsivité', 'Recherche de sensations', 'Consommation d\'alcool', 'Consommation d\'amphétamines', 'Consommation d\'amyl', 'Consommation de benzodiazepine', 'Consommation de café', 'Consommation de cannabis', 'Consommation de chocolat', 'Consommation de cocaïne', 'Consommation de crack', 'Consommation d\'ecstasy', 'Consommation d\'héroïne', 'Consommation de ketamine', 'Consommation de drogues légales', 'Consommation de LSD', 'Consommation de meth', 'Consommation de champignons magiques', 'Consommation de nicotine', 'Consommation de Semeron', 'Consommation de VSA']
    data.columns = noms_colonnes
    data = data.set_index('ID')
    return data

def pre_data(data):
    for i in drogues_col:
        data[i] = data[i].map({'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6})
    semerons = data[data['Consommation de Semeron'] != 0]
    drogues_col.remove('Consommation de Semeron')
    drogues_illégales.remove('Consommation de Semeron')
    data.drop(columns = 'Consommation de Semeron')
    pers_data = data.copy()
    age = ['18-24' if i <= -0.9 else 
           '25-34' if i >= -0.5 and i < 0 else 
           '35-44' if i > 0 and i < 1 else 
           '45-54' if i > 1 and i < 1.5 else 
           '55-64' if i > 1.5 and i < 2 else 
           '65+' 
           for i in pers_data['Age']]

    genre = ['Femme' if i > 0 else "Homme" for i in pers_data['Genre']]

    education = ['A quitté l\'école avant 16 ans' if i <-2 else 
                 'A quitté l\'école à 16 ans' if i > -2 and i < -1.5 else 
                 'A quitté l\'école à 17 ans' if i > -1.5 and i < -1.4 else 
                 'A quitté l\'école à 18 ans' if i > -1.4 and i < -1 else 
                 'Universitaire, sans diplôme' if i > -1 and i < -0.5 else 
                 'Certificat / diplôme professionnel' if i > -0.5 and i < 0 else 
                 'Diplôme universitaire' if i > 0 and i < 0.5 else 
                 'Master' if i > 0.5 and i < 1.5 else 
                 'Doctorat' 
                 for i in pers_data['Education']]

    pays = ['USA' if i < -0.5 else 
               'New Zealand' if i > -0.5 and i < -0.4 else 
               'Other' if i > -0.4 and i < -0.2 else 
               'Australia' if i > -0.2 and i < 0 else 
               'Ireland' if i > 0 and i < 0.23 else 
               'Canada' if i > 0.23 and i < 0.9 else 
               'UK' 
               for i in pers_data['Pays']]

    ethnie = ['Black' if i < -1 else 
                 'Asian' if i > -1 and i < -0.4 else 
                 'White' if i > -0.4 and i < -0.25 else 
                 'Mixed-White/Black' if i >= -0.25 and i < 0.11 else 
                 'Mixed-White/Asian' if i > 0.12 and i < 1 else 
                 'Mixed-Black/Asian' if i > 1.9 else 
                 'Other' 
                 for i in pers_data['Ethnie']]


    pers_data['Age'] = age
    pers_data['Genre'] = genre
    pers_data['Education'] = education
    pers_data['Pays'] = pays
    pers_data['Ethnie'] = ethnie
    return pers_data

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

