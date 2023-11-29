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
    noms_colonnes = ['ID', 'Age', 'Genre', 'Education', 'Pays', 'Ethnie', 'Névrotique', 'Extraverti', 'Ouvert à l\'expérience', 'Amical', 'Consciencieux', 'Imuplsif', 'En recherche de sensations', 'Consommation d\'alcool', 'Consommation d\'amphétamines', 'Consommation d\'amyl', 'Consommation de benzodiazepine', 'Consommation de café', 'Consommation de cannabis', 'Consommation de chocolat', 'Consommation de cocaïne', 'Consommation de crack', 'Consommation d\'ecstasy', 'Consommation d\'héroïne', 'Consommation de ketamine', 'Consommation de drogues légales', 'Consommation de LSD', 'Consommation de meth', 'Consommation de champignons magiques', 'Consommation de nicotine', 'Consommation de Semeron', 'Consommation de VSA']
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

def plot_proportion_bar_drug(dataset, column):
    consommateurs = dataset[dataset[column] >= 0]
    
    proportion_consommateurs = consommateurs[column].value_counts(normalize=True).sort_index()

    couleurs = ['skyblue', 'lightcoral', 'gold', 'lightgreen', 'lightsalmon', 'mediumpurple']

    plt.figure(figsize=(12, 8))

    bars = plt.bar(proportion_consommateurs.index, proportion_consommateurs, color='darkorange', label='Consommateurs', alpha=0.7)

    for i, value in enumerate(proportion_consommateurs):
        plt.text(i, value + 0.01, f'{value:.2%}', ha='center', va='bottom', fontsize=10, color='darkorange')

    plt.title(f'Proportion de la population selon leur {column}', fontsize=16)
    plt.xlabel('Fréquence de consommation', fontsize=14)
    plt.ylabel('Proportion de la population', fontsize=14)
    
    labels = ['Jamais consommé', 'Pas ces 10 dernières années', 'Une fois en 10 ans', 'Une fois par an', 'Une fois par mois', 'Une fois par semaine', 'Tous les jours']
    
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=12)

    plt.yticks(fontsize=12)
    plt.legend()
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

def plot_drug_use_trends_by_age_pers_data(dataset, drug_name):
    
    if drug_name not in dataset.columns:
        raise ValueError("Le nom de la drogue spécifié n'est pas valide.")
    grouped_data = dataset.groupby("Age")[drug_name].value_counts().unstack().fillna(0)

    # Création d'un graphique en ligne
    fig = px.line(
        grouped_data, 
        title=f"Tendances de {drug_name} par catégories d'âge"
    )
    
    fig.update_xaxes(title_text='Catégorie d\'âge')
    fig.update_yaxes(title_text=f'Nombre de consommateurs de {drug_name}')
    
    return fig

def plot_education_level_sunburst(dataset, drugs_list):

    # Filtrage des données pour inclure uniquement les colonnes nécessaires
    cols_to_use = ['Education'] + drugs_list
    filtered_data = dataset[cols_to_use]

    # Préparation des données pour le graphique sunburst
    sunburst_data = []
    for drug in drugs_list:
        for level, count in filtered_data['Education'][filtered_data[drug] > 0].value_counts().items():
            sunburst_data.append([drug, level, count])

    # Conversion en DataFrame
    sunburst_df = pd.DataFrame(sunburst_data, columns=['Drug', 'Education Level', 'Count'])
    fig = px.sunburst(
        sunburst_df, 
        path=['Drug', 'Education Level'], 
        values='Count',
        color='Count',
        title="Comparaison des niveaux d'éducation entre les consommateurs de différentes drogues"
    )

    return fig

def pre_combinaison(dataset,drug_columns):
    def get_drug_combinations(dataset,drug_columns):
        def get_drug_combinations_at_level(dataset, level, drug_columns):
    
            # Filtrage du dataset pour inclure uniquement les individus avec une consommation de niveau spécifié
            filtered_data = dataset.copy()
            for drug in drug_columns:
                filtered_data[drug] = filtered_data[drug].apply(lambda x: 1 if x == level else 0)
    
            # Identification des combinaisons pour chaque individu
            filtered_data['Combinations'] = filtered_data[drug_columns].apply(
                lambda row: '-'.join(row.index[row == 1]) if row.sum() >= 2 else '', axis=1
            )
            filtered_data["Taux d'addiction"] = level
    
            # Filtrage pour exclure les lignes avec des combinaisons vides ou contenant une seule drogue
            filtered_data = filtered_data[filtered_data['Combinations'] != '']
    
            return filtered_data[['Combinations', "Taux d'addiction"]]
    
    
    
        all_levels_combinations = pd.DataFrame()
    
        # Boucle sur les niveaux de consommation de 1 à 6
        for level in range(2, 7):
            level_combinations = get_drug_combinations_at_level(dataset, level, drug_columns)
            all_levels_combinations = pd.concat([all_levels_combinations, level_combinations], ignore_index=True)
    
        # Filtrer pour ne conserver que les combinaisons présentes au moins deux fois
        all_levels_combinations = all_levels_combinations.groupby('Combinations').filter(lambda x: len(x) >= 2)
    
        return all_levels_combinations
    
    # Exemple d'utilisation
    all_levels_combinations = get_drug_combinations(dataset, drug_columns)


    def shorten_drug_names(combination, mapping):
        
        # Séparer la combinaison en drogues individuelles
        drugs = combination.split('-')
        
        # Remplacer chaque nom long par son équivalent court
        short_names = [mapping[drug] if drug in mapping else drug for drug in drugs]
        return '-'.join(short_names)
    
    name_mapping = {
        "Consommation d'alcool": "alcool",
        "Consommation d'amphétamines": "amphétamines",
        "Consommation d'amyl": "amyl",
        "Consommation de benzodiazepine": "benzodiazepine",
        "Consommation de café": "café",
        "Consommation de cannabis": "cannabis",
        "Consommation de chocolat": "chocolat",
        "Consommation de cocaïne": "cocaïne",
        "Consommation de crack": "crack",
        "Consommation d'ecstasy": "ecstasy",
        "Consommation d'héroïne": "héroïne",
        "Consommation de ketamine": "ketamine",
        "Consommation de drogues légales": "drogues légales",
        "Consommation de LSD": "LSD",
        "Consommation de meth": "meth",
        "Consommation de champignons magiques": "champignons magiques",
        "Consommation de nicotine": "nicotine",
        "Consommation de Semeron": "Semeron",
        "Consommation de VSA": "VSA"
    }

    all_levels_combinations['Combinations'] = all_levels_combinations['Combinations'].apply(lambda x: shorten_drug_names(x, name_mapping))
    return all_levels_combinations

def frequence_combinaison(all_levels_combinations):
    freq_combinations = all_levels_combinations.groupby(['Combinations', 'Taux d\'addiction']).size().reset_index(name='Frequency')

    # Trier les combinaisons par fréquence
    sorted_combinations = freq_combinations.sort_values(by='Frequency', ascending=False)
    sorted_combinations=sorted_combinations.head(20)
    
    # Création du graphique en barres
    fig = px.bar(
        sorted_combinations, 
        x='Combinations', 
        y='Frequency', 
        color='Taux d\'addiction', 
        title='Fréquence des Combinaisons les Plus Présentes par Niveau d\'Addiction'
    )
    fig.update_layout(xaxis_title="Combinaisons", yaxis_title="Fréquence", xaxis={'categoryorder':'total descending'})
    return fig
