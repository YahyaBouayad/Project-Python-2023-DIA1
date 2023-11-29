import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def lecture_data():
    fichier_data = 'drug_consumption.data'
    data = pd.read_csv(fichier_data, delimiter=',', header=0)
    noms_colonnes = ['ID', 'Age', 'Genre', 'Education', 'Pays', 'Ethnie', 'Neuroticisme', 'Extraversion', 'Ouverture à l\'expérience', 'Amicalité', 'Conscience', 'Impulsivité', 'Recherche de sensations', 'Consommation d\'alcool', 'Consommation d\'amphétamines', 'Consommation d\'amyl', 'Consommation de benzodiazepine', 'Consommation de café', 'Consommation de cannabis', 'Consommation de chocolat', 'Consommation de cocaïne', 'Consommation de crack', 'Consommation d\'ecstasy', 'Consommation d\'héroïne', 'Consommation de ketamine', 'Consommation de drogues légales', 'Consommation de LSD', 'Consommation de meth', 'Consommation de champignons magiques', 'Consommation de nicotine', 'Consommation de Semeron', 'Consommation de VSA']
    data.columns = noms_colonnes
    data = data.set_index('ID')
    return data

def pre_data(data,col,drogues_col,drogues_illégales):
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

def plot_user_counts_per_drug_combined(data, drogues_col):
    # Create a copy of the DataFrame to avoid modifying the original
    copy_df = data.copy()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Loop through each drug and create a countplot
    for drogue_col in drogues_col:
        # Create a new binary column
        copy_df['User_' + drogue_col.replace(" ", "_")] = (copy_df[drogue_col] > 0).astype(int)

    # Combine all the user count columns into a single DataFrame
    user_counts_df = copy_df[[col for col in copy_df.columns if 'User_' in col]]

    # Calculate total count for each user type (user and non-user)
    total_counts = user_counts_df.sum()

    # Calculate percentage for each user type
    percentages = total_counts / len(copy_df) * 100

    # Plot grouped bar chart
    user_counts_df.sum().plot(kind='bar', color=['blue', 'orange'], ax=ax, position=0.5, width=0.4, label='Users')
    (1 - user_counts_df).sum().plot(kind='bar', color=['orange', 'blue'], ax=ax, position=-0.5, width=0.4, label='Non-Users')

    # Annotate with percentages (rotated 90 degrees)
    for i, count in enumerate(total_counts):
        ax.text(i, count + 1, f"{count}\n({percentages[i]:.2f}%)", ha='center', va='bottom', rotation=90)

    # Set labels and title
    ax.set_ylabel('Count')
    ax.set_xlabel('Drug')
    ax.set_title('User Counts for Different Drugs')

    # Add legend
    ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlapping
    return plt

def plot_user_counts_for_drug(data, drug_name):
    # Create a copy of the DataFrame to avoid modifying the original
    copy_df = data.copy()

    # Create a new binary column
    col_name = 'User_' + drug_name.replace(" ", "_")
    copy_df[col_name] = (copy_df[drug_name] > 0).astype(int)

    # Create a countplot for the specified drug
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=col_name, data=copy_df)

    # Customize the plot to remove y-axis label and x-axis ticks
    ax.set(yticklabels=[])
    ax.set_xticks([])

    plt.title(f'Users vs Non-Users - {drug_name}')
    return plt

def radar_chart_consommation_drogue(data, drogue,caract_col, consommateurs=True):
    
    # Filtrer les lignes en fonction de la consommation
    if consommateurs:
        data_filtre = data[data[drogue] > 1]
    else:
        data_filtre = data[data[drogue] < 1]

    # Calculer la moyenne pour chaque caractéristique
    moyenne_caracteristiques = data_filtre[caract_col].mean()

    # Normaliser les valeurs pour les utiliser dans la radar chart
    moyenne_normalisee = (moyenne_caracteristiques - moyenne_caracteristiques.min()) / (moyenne_caracteristiques.max() - moyenne_caracteristiques.min())

    # Créer un tableau de valeurs pour chaque angle de la radar chart
    angles = np.linspace(0, 2 * np.pi, len(caract_col), endpoint=False)

    # Ajouter la première valeur à la fin pour fermer le cercle
    moyenne_normalisee = np.concatenate((moyenne_normalisee, [moyenne_normalisee[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # Créer le graphique en radar avec une zone de couleur rouge
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, moyenne_normalisee, color='red', alpha=0.5)

    # Ajouter des étiquettes pour chaque caractéristique
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(caract_col)

    # Ajouter un titre en fonction de la consommation
    titre = f'Radar Chart - {drogue}' if consommateurs else f'Radar Chart - Non {drogue}'
    plt.title(titre, size=16, y=1.1)

    # Afficher le graphique
    return plt

def profil_drogue_radar(pers_data,drug_columns,personality_traits):
    average_profiles = pd.DataFrame()

    # Boucle pour calculer les moyennes pour chaque drogue
    for drug in drug_columns:
        # Filtrer les consommateurs réguliers
        regular_consumers = pers_data[pers_data[drug] >= 4]

        # Calculer la moyenne des traits de personnalité
        if not regular_consumers.empty:
            average_profile = regular_consumers[personality_traits].mean()
            average_profile['Drug'] = drug  # Ajouter le nom de la drogue
            average_profiles = pd.concat([average_profiles, average_profile.to_frame().T], ignore_index=True)


    # Créer un graphique radar pour chaque drogue
    fig = go.Figure()

    # Ajouter une ligne pour chaque drogue
    for drug in average_profiles['Drug'].unique():
        drug_data = average_profiles[average_profiles['Drug'] == drug]
        fig.add_trace(go.Scatterpolar(
            r=drug_data[personality_traits].values[0],
            theta=personality_traits,
            fill='toself',
            name=drug
        ))

    # Améliorer la mise en page
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Assurez-vous que cette plage correspond à votre échelle de données
            )),
        showlegend=True,
        title='Comparaison des profils de personnalité moyens des consommateurs réguliers de différentes drogues'
    )
    return fig
