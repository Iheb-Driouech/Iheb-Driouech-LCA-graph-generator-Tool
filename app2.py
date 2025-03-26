# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch


def analyze_excel_and_generate_tables(file_path, sheet_name=0):
    try:
        # Lecture du fichier Excel
        data = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
        print(f"Fichier Excel chargé avec succès depuis la feuille : {sheet_name} !")

        # Étape 1 : Détection des scénarios
        scenario_row = data.iloc[0, 1:]
        scenario_names = scenario_row.unique()
        scenario_names_cleaned = [name[name.find("(") + 1 : name.find(")")].strip().lower() for name in scenario_names]
        print(f"Scénarios détectés ({len(scenario_names_cleaned)}) : {scenario_names_cleaned}")

        # Étape 2 : Détection des contributions
        contribution_row = data.iloc[1, 1:]
        print(f"Contributions détectées : {list(contribution_row.unique())}")

        # Étape 3 : Structuration des données
        categories = data.iloc[2:, 0]
        values = data.iloc[2:, 1:]
        values.columns = pd.MultiIndex.from_arrays([scenario_row, contribution_row])
        values.index = categories

        # Tableau initial
        initial_table = values.copy()

        # Totaux d’impact
        total_impact_table = values.abs().T.groupby(level=0).sum().T

        # Remplacer les noms des colonnes par les noms simplifiés des scénarios
        total_impact_table.columns = [name[name.find("(") + 1 : name.find(")")].strip() for name in total_impact_table.columns]


        # Pourcentages relatifs
        relative_percentage_table = total_impact_table.copy()
        for category in relative_percentage_table.index:
            max_value = relative_percentage_table.loc[category].max()
            relative_percentage_table.loc[category] = (
                (relative_percentage_table.loc[category] / max_value) * 100
            )
        relative_percentage_table = relative_percentage_table.add_suffix(" (%)")

        # Fusion des deux tableaux
        combined_table = pd.concat([total_impact_table, relative_percentage_table], axis=1)
        
        return initial_table, combined_table, total_impact_table, scenario_names_cleaned
    except Exception as e:
        print(f"Erreur lors de l'analyse du fichier Excel : {e}")
        return None, None, None, None

def generate_percentage_table(initial_table, total_impact_table):
    """
    Génère un tableau identique à `initial_table`, mais avec les valeurs en pourcentage.

    :param initial_table: DataFrame initial contenant les valeurs d'impact.
    :param total_impact_table: DataFrame contenant les totaux d'impact pour chaque catégorie.
    :return: DataFrame avec les valeurs en pourcentage.
    """
    try:
        # Copier la structure du tableau initial
        percentage_table = initial_table.copy()
        
        # Calculer les pourcentages
        for category in initial_table.index:
            # Récupérer le maximum dans `total_impact_table` pour cette catégorie
            max_total_impact = total_impact_table.loc[category].max()

            # Diviser chaque valeur par ce maximum et multiplier par 100
            percentage_table.loc[category] = (initial_table.loc[category] / max_total_impact) * 100

        return percentage_table
    except Exception as e:
        print(f"Erreur lors de la génération du tableau en pourcentage : {e}")
        return None

    

def generate_tables_by_scenario(initial_table, scenario_names_cleaned):
    """
    Génère un tableau formaté par scénario à partir du tableau initial,
    avec une colonne pour le total d’impact et les pourcentages de contribution.

    :param initial_table: DataFrame initial contenant les catégories d'impact et les contributions.
    :param scenario_names_cleaned: Liste des noms des scénarios simplifiés.
    :return: Dictionnaire contenant un tableau par scénario.
    """
    try:
        scenario_tables = {}
        for scenario_clean in scenario_names_cleaned:
            matching_scenario = [
                col for col in initial_table.columns.levels[0] if scenario_clean in col.lower()
            ]
            if not matching_scenario:
                raise KeyError(f"Le scénario simplifié '{scenario_clean}' n'a pas de correspondance.")
            matching_scenario = matching_scenario[0]

            # Filtrer les données pour ce scénario
            scenario_data = initial_table.xs(key=matching_scenario, axis=1, level=0)

            # Créer un tableau formaté
            scenario_table = pd.DataFrame()
            scenario_table["Catégorie d'impact"] = initial_table.index
            for contribution in scenario_data.columns:
                scenario_table[contribution] = scenario_data[contribution].values

            # Calcul du total d’impact pour chaque catégorie
            # Calcul du total d’impact en utilisant la valeur absolue des impacts
            scenario_table["Total d'impact"] = scenario_table.iloc[:, 1:].abs().sum(axis=1)


            # Calcul des pourcentages de contribution
            for contribution in scenario_data.columns:
                scenario_table[f"% {contribution}"] = (
                    scenario_table[contribution] / scenario_table["Total d'impact"] * 100
                )

            # Ajouter le tableau au dictionnaire
            scenario_tables[scenario_clean] = scenario_table

            # Afficher un aperçu du tableau
            print(f"\nTableau généré pour {scenario_clean} :")
            print(scenario_table.head())

        return scenario_tables
    except Exception as e:
        print(f"Erreur lors de la génération des tableaux par scénario : {e}")
        return None

def plot_comparison_bar_chart(total_impact_table):
    try:
        global scenario_colors
        figs = []  # Liste pour stocker les figures générées
        
        # Normalisation des données
        normalized_data = total_impact_table.copy()
        for category in normalized_data.index:
            max_value = normalized_data.loc[category].max()
            normalized_data.loc[category] = (normalized_data.loc[category] / max_value) * 100
        
        categories = normalized_data.index
        scenarios = normalized_data.columns
        num_scenarios = len(scenarios)
        bar_width = 0.9 / num_scenarios
        x_positions = np.arange(len(categories))

        # Gestion des couleurs
        if scenario_colors is not None and not scenario_colors.empty:
            scenario_color_map = {
                row["Scenario"].strip().lower(): row["Hex Code"]
                for _, row in scenario_colors.iterrows()
            }
        else:
            scenario_color_map = None
            generic_colors = plt.get_cmap("tab10").colors

        # --------------------------------------------------
        # Graphique principal (sans totaux)
        # --------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        bars = []
        labels = []

        for i, scenario in enumerate(scenarios):
            color = (
                scenario_color_map.get(scenario.strip().lower(), "#000000")
                if scenario_color_map
                else generic_colors[i % len(generic_colors)]
            )
            bar = ax1.bar(x_positions + i * bar_width, normalized_data[scenario], bar_width, color=color)
            bars.append(bar[0])
            labels.append(scenario)

        ax1.set_title("Comparison of scenarios", fontsize=16, pad=30, fontweight="bold")
        ax1.set_ylabel("(%)", fontsize=14)
        ax1.set_ylim(0, 100)
        ax1.set_xlim(-0.5, len(categories))
        ax1.set_xticks(x_positions + (len(scenarios) - 1) * bar_width / 2)
        ax1.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        figs.append(fig1)
        plt.close(fig1)

        # --------------------------------------------------
        # Légende séparée
        # --------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.axis("off")
        ax2.legend(bars, labels, title="Scenarios", fontsize=10, loc="center")
        plt.tight_layout()
        figs.append(fig2)
        plt.close(fig2)

        # --------------------------------------------------
        # Version avec totaux
        # --------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        
        for i, scenario in enumerate(scenarios):
            color = (
                scenario_color_map.get(scenario.strip().lower(), "#000000")
                if scenario_color_map
                else generic_colors[i % len(generic_colors)]
            )
            ax3.bar(x_positions + i * bar_width, normalized_data[scenario], bar_width, color=color)

        # Ajout des totaux
        for i, scenario in enumerate(scenarios):
            for j, value in enumerate(normalized_data[scenario]):
                total_value = total_impact_table.loc[categories[j], scenario]
                formatted_value = (
                    f"{total_value:.2f}" 
                    if 0.01 <= abs(total_value) <= 1000 
                    else f"{total_value:.2E}"
                )
                ax3.text(
                    x_positions[j] + i * bar_width,
                    value + 2,
                    formatted_value,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="black",
                    fontweight="bold",
                    rotation=90 if abs(total_value) < 0.01 else 0
                )

        ax3.set_title("Comparison of scenarios (with totals)", fontsize=16, pad=80, fontweight="bold")
        ax3.set_ylabel("(%)", fontsize=14)
        ax3.set_ylim(0, 100)
        ax3.set_xlim(-0.5, len(categories))
        ax3.set_xticks(x_positions + (len(scenarios) - 1) * bar_width / 2)
        ax3.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        figs.append(fig3)
        plt.close(fig3)

        return figs

    except Exception as e:
        st.error(f"Erreur de génération du graphique: {str(e)}")
        return []


def plot_relative_contribution_by_scenario(scenario_tables):
    """
    Génère et retourne une liste de tuples (figure principale, légende) pour chaque scénario
    """
    try:
        global contributions_colors
        figures = []

        # Identifier toutes les contributions
        all_contributions = set()
        for table in scenario_tables.values():
            contribution_columns = [col for col in table.columns 
                                  if col.startswith("%") and col != "%Total d'impact"]
            all_contributions.update(contribution_columns)

        # Gestion des couleurs
        contribution_colors = {}
        if contributions_colors is not None and not contributions_colors.empty:
            for contribution in sorted(all_contributions):
                contrib_name = contribution.replace("%", "").strip()
                match = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_name.lower()]
                if not match.empty:
                    contribution_colors[contribution] = match["Hex Code"].iloc[0]
                else:
                    contribution_colors[contribution] = "#000000"
        else:
            generic_colors = plt.get_cmap("tab10").colors
            for i, contribution in enumerate(sorted(all_contributions)):
                contribution_colors[contribution] = generic_colors[i % len(generic_colors)]

        # Génération des graphiques pour chaque scénario
        for scenario_name, table in scenario_tables.items():
            # Création des figures
            main_fig, main_ax = plt.subplots(figsize=(16, 8))
            legend_fig, legend_ax = plt.subplots(figsize=(5, 3))
            
            # Paramètres communs
            categories = table["Catégorie d'impact"]
            contribution_columns = [col for col in table.columns 
                                  if col.startswith("%") and col != "%Total d'impact"]
            percentage_data = table[contribution_columns].astype(float)
            total_impact = table["Total d'impact"].astype(float)
            bar_width = 0.6
            bars = []
            labels = []

            # Empilement des barres
            bottom_pos = np.zeros(len(categories))
            bottom_neg = np.zeros(len(categories))

            for contribution in contribution_columns:
                values = percentage_data[contribution].values
                pos_values = np.where(values > 0, values, 0)
                neg_values = np.where(values < 0, values, 0)

                # Barres positives
                if np.any(pos_values):
                    bar = main_ax.bar(categories, pos_values, bar_width, 
                                     bottom=bottom_pos, 
                                     color=contribution_colors[contribution])
                    bars.append(bar[0])
                    labels.append(contribution.replace("%", ""))
                    bottom_pos += pos_values

                # Barres négatives
                if np.any(neg_values):
                    main_ax.bar(categories, neg_values, bar_width,
                                bottom=bottom_neg,
                                color=contribution_colors[contribution])
                    bottom_neg += neg_values

            # Ligne de séparation
            main_ax.axhline(0, color="black", linestyle="--", alpha=0.7)

            # Ajout des totaux
            for i, total in enumerate(total_impact):
                formatted_value = f"{total:.2f}" if 0.01 <= abs(total) <= 1000 else f"{total:.2E}"
                main_ax.text(i, max(bottom_pos[i], 0) + 1, formatted_value,
                           ha="center", va="bottom", fontsize=10,
                           color="black", fontweight="bold")

            # Configuration finale du graphique principal
            main_ax.set_title(f"Contributions relatives - {scenario_name.capitalize()}", 
                            fontsize=16, pad=40, fontweight="bold")
            main_ax.set_ylabel("(%)")
            main_ax.set_ylim(min(bottom_neg) - 5, max(bottom_pos))
            main_ax.set_xticks(range(len(categories)))
            main_ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
            main_ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Configuration de la légende
            legend_ax.axis("off")
            legend_ax.legend(bars, labels, title="Contributions", 
                           fontsize=10, loc="center")
            plt.tight_layout()

            # Stockage et nettoyage
            figures.append((main_fig, legend_fig))
            plt.close(main_fig)
            plt.close(legend_fig)

        return figures

    except Exception as e:
        st.error(f"Erreur de génération des graphiques : {str(e)}")
        return []


# Fonction pour énérer des graphiques empilés horizontales en pourcentage pour chaque scénario


def plot_relative_contribution_by_scenario_horizontal(scenario_tables):
    """
    Génère et retourne une liste de tuples (figure principale, légende) pour chaque scénario
    Version horizontale avec totaux à droite
    """
    try:
        global contributions_colors
        figures = []

        # Identifier toutes les contributions
        all_contributions = set()
        for table in scenario_tables.values():
            contribution_columns = [col for col in table.columns 
                                  if col.startswith("%") and col != "%Total d'impact"]
            all_contributions.update(contribution_columns)

        # Gestion des couleurs
        contribution_color_map = {}
        if contributions_colors is not None and not contributions_colors.empty:
            for contribution in sorted(all_contributions):
                contrib_name = contribution.replace("%", "").strip()
                match = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_name.lower()]
                if not match.empty:
                    contribution_color_map[contribution] = match["Hex Code"].iloc[0]
                else:
                    contribution_color_map[contribution] = "#000000"
        else:
            generic_colors = plt.get_cmap("tab10").colors
            for i, contribution in enumerate(sorted(all_contributions)):
                contribution_color_map[contribution] = generic_colors[i % len(generic_colors)]

        # Génération des graphiques pour chaque scénario
        for scenario_name, table in scenario_tables.items():
            # Création des figures
            main_fig, main_ax = plt.subplots(figsize=(10, 12))
            legend_fig, legend_ax = plt.subplots(figsize=(5, 3))
            
            # Paramètres communs
            categories = table["Catégorie d'impact"]
            contribution_columns = [col for col in table.columns 
                                  if col.startswith("%") and col != "%Total d'impact"]
            percentage_data = table[contribution_columns].astype(float)
            total_impact = table["Total d'impact"].astype(float)
            bar_height = 0.6
            bars = []
            labels = []

            # Empilement horizontal
            left_pos = np.zeros(len(categories))
            left_neg = np.zeros(len(categories))

            for contribution in contribution_columns:
                values = percentage_data[contribution].values
                pos_values = np.where(values > 0, values, 0)
                neg_values = np.where(values < 0, values, 0)

                # Barres positives
                if np.any(pos_values):
                    bar = main_ax.barh(categories, pos_values, bar_height,
                                      left=left_pos,
                                      color=contribution_color_map[contribution])
                    bars.append(bar[0])
                    labels.append(contribution.replace("%", ""))
                    left_pos += pos_values

                # Barres négatives
                if np.any(neg_values):
                    main_ax.barh(categories, neg_values, bar_height,
                                 left=left_neg,
                                 color=contribution_color_map[contribution])
                    left_neg += neg_values

            # Ligne de séparation verticale
            main_ax.axvline(0, color="black", linestyle="--", alpha=0.7)

            # Ajout des totaux
            for i, total in enumerate(total_impact):
                formatted_value = f"{total:.2f}" if 0.01 <= abs(total) <= 1000 else f"{total:.2E}"
                main_ax.text(max(left_pos[i], 0) + 5, i, formatted_value,
                           ha="left", va="center", fontsize=10,
                           color="black", fontweight="bold")

            # Configuration finale du graphique
            main_ax.set_title(f"Contributions horizontales - {scenario_name.capitalize()}", 
                            fontsize=16, pad=30, fontweight="bold")
            main_ax.set_xlabel("(%)")
            main_ax.set_xlim(min(left_neg) - 5, max(left_pos))
            main_ax.set_yticks(range(len(categories)))
            main_ax.set_yticklabels(categories, fontsize=12)
            main_ax.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Configuration de la légende
            legend_ax.axis("off")
            legend_ax.legend(bars, labels, title="Contributions", 
                           fontsize=10, loc="center")
            plt.tight_layout()

            # Stockage et nettoyage
            figures.append((main_fig, legend_fig))
            plt.close(main_fig)
            plt.close(legend_fig)

        return figures

    except Exception as e:
        st.error(f"Erreur de génération des graphiques horizontaux : {str(e)}")
        return []

def plot_stacked_bar_by_category(percentage_table, total_impact_table):
    """
    Génère et retourne une liste de tuples (figure principale, légende, nom de catégorie)
    """
    try:
        global contributions_colors
        figures = []

        # Vérification des données d'entrée
        if percentage_table.empty or total_impact_table.empty:
            st.warning("Données d'entrée vides")
            return []

        # Gestion des couleurs
        contrib_color_map = {}
        contributions = percentage_table.columns.levels[1]
        if contributions_colors is not None and not contributions_colors.empty:
            for contribution in contributions:
                contrib_name = contribution.replace("%", "").strip().lower()
                match = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_name]
                contrib_color_map[contribution] = match["Hex Code"].iloc[0] if not match.empty else "#000000"
        else:
            generic_colors = plt.get_cmap("tab10").colors
            for i, contribution in enumerate(contributions):
                contrib_color_map[contribution] = generic_colors[i % len(generic_colors)]

        # Génération des graphiques pour chaque catégorie
        for category in percentage_table.index:
            # Création des figures
            main_fig, main_ax = plt.subplots(figsize=(12, 7))
            legend_fig, legend_ax = plt.subplots(figsize=(5, 3))
            
            # Paramètres communs
            scenarios = percentage_table.columns.levels[0]
            bar_width = max(0.2, min(0.8, 1.5 / len(scenarios)))
            x_positions = np.arange(len(scenarios))
            
            # Gestion des piles
            bottom_pos = np.zeros(len(scenarios))
            bottom_neg = np.zeros(len(scenarios))
            bars = []
            labels = []

            # Pour chaque contribution
            for contribution in contributions:
                values = np.array([percentage_table.loc[category, (s, contribution)] for s in scenarios])
                
                # Séparation positif/négatif
                pos_values = np.where(values > 0, values, 0)
                neg_values = np.where(values < 0, values, 0)

                # Barres positives
                if np.any(pos_values):
                    bar = main_ax.bar(x_positions, pos_values, bar_width, 
                                    bottom=bottom_pos, 
                                    color=contrib_color_map[contribution])
                    bars.append(bar[0])
                    labels.append(contribution)
                    bottom_pos += pos_values

                # Barres négatives
                if np.any(neg_values):
                    main_ax.bar(x_positions, neg_values, bar_width,
                               bottom=bottom_neg,
                               color=contrib_color_map[contribution])
                    bottom_neg += neg_values

            # Ajout des totaux
            for i, scenario in enumerate(scenarios):
                try:
                    scenario_clean = scenario.split("(")[-1].replace(")", "").strip().lower()
                    total_value = total_impact_table.loc[category, scenario_clean]
                    formatted_value = f"{total_value:.2f}" if 0.01 <= abs(total_value) < 1000 else f"{total_value:.2E}"
                    main_ax.text(x_positions[i], max(bottom_pos[i], 0) + 5, formatted_value,
                               ha='center', va='bottom', fontsize=10, 
                               color='black', fontweight='bold')
                except KeyError:
                    continue

            # Configuration du graphique principal
            main_ax.set_title(f"Analyse des contributions - Catégorie : {category}", 
                            fontsize=16, pad=40, fontweight="bold")
            main_ax.set_ylabel("(%)", fontsize=14)
            main_ax.set_xticks(x_positions)
            main_ax.set_xticklabels([s.split("(")[-1].replace(")", "").strip() for s in scenarios], 
                                   rotation=45, ha="right", fontsize=12)
            main_ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Configuration de la légende
            legend_ax.axis("off")
            legend_ax.legend(bars, labels, title="Contributions", 
                            fontsize=10, loc="center")
            plt.tight_layout()

            # Stockage des figures AVEC le nom de catégorie
            figures.append((main_fig, legend_fig, category))

        return figures

    except Exception as e:
        st.error(f"Erreur de génération des graphiques : {str(e)}")
        return []


def plot_combined_graph_with_scenario_hatches(percentage_table, total_impact_table):
    """
    Génère 3 figures Streamlit :
    1. Graphique principal sans totaux
    2. Légende séparée
    3. Graphique avec totaux
    """
    try:
        global contributions_colors
        figures = []

        # Vérification des données
        if percentage_table.empty or total_impact_table.empty:
            st.warning("Données manquantes pour générer le graphique combiné")
            return []

        # Configuration initiale
        categories = percentage_table.index
        scenarios = percentage_table.columns.levels[0]
        contributions = percentage_table.columns.levels[1]
        num_scenarios = len(scenarios)
        bar_width = 0.9 / num_scenarios
        x_positions = np.arange(len(categories))

        # Création des motifs et couleurs
        scenario_hatches = ["//", "oo", "..", "xx", "--", "||", "++"]
        scenario_hatch_dict = {scenario: scenario_hatches[i%len(scenario_hatches)] 
                             for i, scenario in enumerate(scenarios)}
        
        # Mapping des couleurs (CORRECTION APPLIQUÉE)
        contrib_color_map = {}
        if contributions_colors is not None and not contributions_colors.empty:
            for contrib in contributions:
                clean_contrib = contrib.replace("%", "").strip().lower()
                match = contributions_colors[contributions_colors["Contributions"].str.lower() == clean_contrib]
                contrib_color_map[contrib] = match["Hex Code"].iloc[0] if not match.empty else "#000000"
        else:
            cmap = plt.get_cmap("tab10")
            contrib_color_map = {contrib: cmap(i%10) for i, contrib in enumerate(contributions)}

        # Fonction générique pour créer un graphique (CORRECTION CRITIQUE)
        def create_figure(show_totals=False):
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.set_title("Analyse combinée des scénarios" + (" avec totaux" if show_totals else ""), 
                        fontsize=18, pad=20, fontweight='bold')
            
            for i, scenario in enumerate(scenarios):
                bottom_pos = np.zeros(len(categories))
                bottom_neg = np.zeros(len(categories))
                
                for contrib in contributions:
                    # CORRECTION : Conversion explicite en array numpy
                    try:
                        values = percentage_table.xs((scenario, contrib), axis=1, level=[0,1]).values.flatten()
                    except KeyError:
                        continue
                    
                    # Conversion en float et gestion des NaN
                    values = np.nan_to_num(values.astype(float))
                    
                    # Séparation positif/négatif
                    pos_values = np.where(values > 0, values, 0)
                    neg_values = np.where(values < 0, values, 0)

                    # Barres positives
                    if np.any(pos_values):
                        ax.bar(x_positions + i*bar_width, pos_values, bar_width,
                               bottom=bottom_pos, color=contrib_color_map[contrib],
                               hatch=scenario_hatch_dict[scenario], edgecolor='black')
                        bottom_pos += pos_values
                    
                    # Barres négatives
                    if np.any(neg_values):
                        ax.bar(x_positions + i*bar_width, neg_values, bar_width,
                               bottom=bottom_neg, color=contrib_color_map[contrib],
                               hatch=scenario_hatch_dict[scenario], edgecolor='black')
                        bottom_neg += neg_values
                
                # Ajout des totaux si activé (CORRECTION DE L'INDEXATION)
                if show_totals:
                    scenario_clean = scenario.split("(")[-1].replace(")", "").strip()
                    try:
                        totals = total_impact_table[scenario_clean].values
                        for j, total in enumerate(totals):
                            ax.text(x_positions[j] + i*bar_width, bottom_pos[j] + 2,
                                   f"{total:.2E}" if abs(total) >= 1e3 else f"{total:.2f}",
                                   ha='center', va='bottom', rotation=90, fontsize=8)
                    except KeyError:
                        continue

            # Configuration des axes
            ax.set_xticks(x_positions + (num_scenarios-1)*bar_width/2)
            ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
            ax.set_ylabel("Contribution (%)", fontsize=14)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            return fig

        # Génération des figures
        figures.append(("Graphique principal", create_figure(False)))
        figures.append(("Graphique avec totaux", create_figure(True)))
        
        # Création de la légende
        legend_fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        
        # Légende des contributions
        contrib_handles = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black') 
                          for contrib, color in contrib_color_map.items()]
        
        # Légende des scénarios
        scenario_handles = [plt.Rectangle((0,0),1,1, facecolor='white', 
                                        hatch=hatch, edgecolor='black') 
                           for scenario, hatch in scenario_hatch_dict.items()]
        
        # Combinaison des légendes
        ax.legend(handles=contrib_handles + scenario_handles,
                 labels=list(contrib_color_map.keys()) + list(scenario_hatch_dict.keys()),
                 title="Légende - Contributions & Scénarios",
                 ncol=2, fontsize=8, loc='center')
        
        figures.append(("Légende combinée", legend_fig))

        return figures

    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique combiné : {str(e)}")
        return []


def generate_table_graph(df, table_name):
    """
    Génère un graphique affichant un tableau DataFrame intact avec mise en forme :
    - Les valeurs numériques sont affichées en notation scientifique (2E).
    - La première ligne (en-tête) et la première colonne sont en gras et colorées en lightblue.
    - L'espace est ajusté dynamiquement pour éviter tout chevauchement.
    
    :param df: DataFrame à afficher
    :param table_name: Nom du tableau pour le titre du graphique
    """

    # ✅ Convertir les valeurs numériques en notation scientifique 2E sans modifier le DataFrame original
    formatted_df = df.copy()
    formatted_df = formatted_df.applymap(lambda x: f"{x:.2E}" if isinstance(x, (int, float)) else x)

    # ✅ Définir dynamiquement la taille du graphique
    num_rows, num_cols = formatted_df.shape
    fig_width = max(10, num_cols * 1.5)  # Largeur ajustée en fonction des colonnes
    fig_height = max(6, num_rows * 0.5 + 1.5)  # Hauteur ajustée pour éviter le chevauchement

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()  # Cacher les axes

    # ✅ Calcul de l'espacement pour éviter les chevauchements
    title_y_position = 0.96  # Position haute par défaut
    table_y_position = 0.96 - (num_rows * 0.01)  # Ajustement dynamique en fonction du nombre de lignes
    table_y_position = max(0.70, table_y_position)  # Empêcher la table de descendre trop bas

    # ✅ Ajouter un titre en haut du graphique
    plt.figtext(0.5, title_y_position, f"Table: {table_name}", 
                fontsize=18, fontweight='bold', ha='center')

    # ✅ Créer le tableau bien **en dessous** du titre
    table = ax.table(cellText=formatted_df.values, 
                     colLabels=formatted_df.columns, 
                     rowLabels=formatted_df.index, 
                     loc='center', cellLoc='center', 
                     colWidths=[0.5] * formatted_df.shape[1])

    # ✅ Appliquer la mise en forme (taille et hauteur uniforme)
    for key, cell in table.get_celld().items():
        cell.set_fontsize(14)
        cell.set_height(0.08)

    # ✅ Coloration et mise en gras des **en-têtes** et de la **première colonne**
    header_color = 'lightblue'
    first_col_color = 'lightblue'

    # En-tête (première ligne)
    for i in range(len(formatted_df.columns)):  
        cell = table[0, i]  
        cell.set_text_props(weight='bold')
        cell.set_facecolor(header_color)

    # Première colonne (labels des lignes)
    for i in range(len(formatted_df.index)):  
        cell = table[i + 1, -1]  
        cell.set_text_props(weight='bold')
        cell.set_facecolor(first_col_color)

    # ✅ Ajuster correctement l'espacement pour **éviter les chevauchements**
    plt.subplots_adjust(top=table_y_position, bottom=0.1)

    # ✅ Afficher le tableau
    return fig
def find_file_path(file_name, start_directory=None):
    """
    Recherche récursive d'un fichier dans 'start_directory'.
    Si 'start_directory' n'est pas fourni, on utilise le répertoire utilisateur.
    """
    if start_directory is None:
        # Par défaut, on part du dossier utilisateur
        start_directory = os.path.expanduser("~")

    for root, dirs, files in os.walk(start_directory):
        if file_name in files:
            # Si le fichier est trouvé, on construit le chemin
            file_path = os.path.join(root, file_name)
            return file_path
    
    # Si on ne trouve pas le fichier, on retourne None
    return None

def generate_color_catalog_tables(file_path, sheet_name="Color catalog"):
    """
    Lit la feuille 'Color catalog' du fichier Excel et génère deux DataFrames :
      - contributions_colors : contient deux colonnes "Contributions" et "Hex Code"
        (seules les lignes où la colonne 'Contributions' est non vide sont retenues).
      - scenario_colors : contient deux colonnes "Scenario" et "Hex Code"
        (seules les lignes où la colonne 'Scenarios' est non vide sont retenues).
    
    La recherche des colonnes se fait de manière insensible à la casse.
    
    :param file_path: Chemin complet vers le fichier Excel.
    :param sheet_name: Nom de la feuille à lire (par défaut "Color catalog").
    :return: Tuple (contributions_colors, scenario_colors) ou (None, None) en cas d'erreur.
    """
    try:
        # Lecture de la feuille "Color catalog"
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Création d'un mapping des noms de colonnes en minuscules vers leur nom original
        col_map = {col.lower(): col for col in df.columns}
        
        # Vérifier et récupérer la colonne "Contributions"
        if "contributions" not in col_map:
            raise KeyError("La colonne 'Contributions' n'existe pas dans la feuille.")
        contributions_col = col_map["contributions"]
        
        # Vérifier et récupérer la colonne "Hex Code"
        if "hex code" not in col_map:
            raise KeyError("La colonne 'Hex Code' n'existe pas dans la feuille.")
        hex_col = col_map["hex code"]
        
        # Générer contributions_colors : filtrer les lignes où la colonne 'Contributions' n'est pas vide
        mask_contrib = df[contributions_col].notna() & (df[contributions_col].astype(str).str.strip() != "")
        contributions_colors = df[mask_contrib][[contributions_col, hex_col]].copy()
        contributions_colors.rename(columns={contributions_col: "Contributions", hex_col: "Hex Code"}, inplace=True)
        contributions_colors.reset_index(drop=True, inplace=True)
        
        # Vérifier et récupérer la colonne "Scenarios" (ou "scenarios")
        if "scenarios" not in col_map:
            raise KeyError("La colonne 'Scenarios' n'existe pas dans la feuille.")
        scenarios_col = col_map["scenarios"]
        
        # Générer scenario_colors : filtrer les lignes où la colonne 'Scenarios' n'est pas vide
        mask_scenario = df[scenarios_col].notna() & (df[scenarios_col].astype(str).str.strip() != "")
        scenario_colors = df[mask_scenario][[scenarios_col, hex_col]].copy()
        scenario_colors.rename(columns={scenarios_col: "Scenario", hex_col: "Hex Code"}, inplace=True)
        scenario_colors.reset_index(drop=True, inplace=True)
        
        print("Tableau contributions_colors généré avec succès :")
        print(contributions_colors.head())
        print("\nTableau scenario_colors généré avec succès :")
        print(scenario_colors.head())
        
        return contributions_colors, scenario_colors

    except Exception as e:
        print(f"Erreur lors de la génération des tableaux de couleurs : {e}")
        return None, None

def main():
    st.title("Analyse d'Impact Environnemental")
    
    # Upload du fichier Excel
    uploaded_file = st.file_uploader("Déposer le fichier Excel ici", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            # Générer les tableaux de couleurs depuis la feuille 'Color catalog'
            global contributions_colors, scenario_colors
            contributions_colors, scenario_colors = generate_color_catalog_tables(uploaded_file, sheet_name="Color catalog")
            
            # Analyse du fichier avec la feuille 'Data'
            initial_table, combined_table, total_impact_table, scenario_names_cleaned = analyze_excel_and_generate_tables(
                uploaded_file, sheet_name="Feuil2"
            )
            
            if initial_table is not None:
                # Génération des tableaux
                scenario_tables = generate_tables_by_scenario(initial_table, scenario_names_cleaned)
                percentage_table = generate_percentage_table(initial_table, total_impact_table)

                # Affichage des résultats
                st.subheader("Comparaison of scenarios")
                if total_impact_table is not None:
                   figures = plot_comparison_bar_chart(total_impact_table)
                   if figures:
                        st.pyplot(figures[0])  # Graphique principal
                        st.pyplot(figures[1])  # Légende
                        st.pyplot(figures[2])  # Version avec totaux
                
            if scenario_tables:
                      st.header("Analyse détaillée par scénario")
    
                      # Générer les figures
                      scenario_figures = plot_relative_contribution_by_scenario(scenario_tables)
    
                      # Afficher chaque paire graphique/légende
                      for idx, (main_fig, legend_fig) in enumerate(scenario_figures):
                            scenario_name = list(scenario_tables.keys())[idx]
        
                            st.subheader(f"Scénario : {scenario_name}")
        
                            # Création de colonnes pour l'affichage
                            col1, col2 = st.columns([4, 1])
        
                            with col1:
                                st.pyplot(main_fig)
            
                            with col2:
                                st.pyplot(legend_fig)
            
                            st.markdown("---")  # Séparateur entre les scénarios

                
            if scenario_tables:
                 st.header("Analyse horizontale par scénario")
    
                 # Générer les figures
                 horizontal_figures = plot_relative_contribution_by_scenario_horizontal(scenario_tables)
    
                 # Afficher chaque paire graphique/légende
                 for idx, (main_fig, legend_fig) in enumerate(horizontal_figures):
                       scenario_name = list(scenario_tables.keys())[idx]
        
                       st.subheader(f"Scénario : {scenario_name} (vue horizontale)")
        
                       # Création de colonnes pour l'affichage
                       col1, col2 = st.columns([4, 1])
        
                       with col1:
                              st.pyplot(main_fig)
            
                       with col2:
                              st.pyplot(legend_fig)
            
                       st.markdown("---")

            if percentage_table is not None and total_impact_table is not None:
                 category_figures = plot_stacked_bar_by_category(percentage_table, total_impact_table)
    
                 if category_figures:
                     st.header("Analyse détaillée par catégorie d'impact")
        
                     for fig_tuple in category_figures:
                           main_fig, legend_fig, category_name = fig_tuple
            
                           st.subheader(f"Catégorie : {category_name}")
            
                           # Création des colonnes pour l'affichage
                           col1, col2 = st.columns([4, 1])
            
                           with col1:
                                 st.pyplot(main_fig)
                           with col2:
                                 st.pyplot(legend_fig)
            
                           # Fermeture des figures APRÈS l'affichage
                           plt.close(main_fig)
                           plt.close(legend_fig)
            
                           st.markdown("---")

                # Graphique combiné final
            if percentage_table is not None and total_impact_table is not None:
                combined_figures = plot_combined_graph_with_scenario_hatches(percentage_table, total_impact_table)
    
                if combined_figures:
                  st.header("Vue combinée des scénarios")
        
                  # Afficher le graphique principal
                  st.subheader("Comparaison des scénarios")
                  for name, fig in combined_figures:
                      if name == "Graphique principal":
                         st.pyplot(fig)
                         plt.close(fig)
        
                  # Afficher la légende
                  st.subheader("Légende")
                  for name, fig in combined_figures:
                      if name == "Légende combinée":
                         st.pyplot(fig)
                         plt.close(fig)
        
                  # Afficher la version avec totaux
                  st.subheader("Version détaillée avec totaux")
                  for name, fig in combined_figures:
                      if name == "Graphique avec totaux":
                          st.pyplot(fig)
                          plt.close(fig)

    
                
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {str(e)}")

if __name__ == "__main__":
    main()

