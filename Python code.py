# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Fonction principale pour analyser le fichier Excel
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

    
# Fonction pour générer un tableau par scénario
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
        
        # 🔹 Normaliser les données pour chaque catégorie
        normalized_data = total_impact_table.copy()
        for category in normalized_data.index:
            max_value = normalized_data.loc[category].max()
            normalized_data.loc[category] = (normalized_data.loc[category] / max_value) * 100
        
        categories = normalized_data.index
        scenarios = normalized_data.columns
        num_scenarios = len(scenarios)
        bar_width = 0.9 / num_scenarios  # Ajustement dynamique de la largeur des barres
        x_positions = np.arange(len(categories))
        
        # 🔹 Gestion des couleurs des scénarios
        if scenario_colors is not None and not scenario_colors.empty:
            scenario_color_map = {
                row["Scenario"].strip().lower(): row["Hex Code"]
                for _, row in scenario_colors.iterrows()
            }
        else:
            scenario_color_map = None
            generic_colors = plt.get_cmap("tab10").colors
        
        # 🔹 Création de la figure principale
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = []
        labels = []

        for i, scenario in enumerate(scenarios):
            # Déterminer la couleur pour ce scénario
            if scenario_color_map is not None:
                # On normalise le nom pour la comparaison
                color = scenario_color_map.get(scenario.strip().lower(), "#000000")
            else:
                color = generic_colors[i % len(generic_colors)]

            bar = ax.bar(
                x_positions + i * bar_width,
                normalized_data[scenario],
                bar_width,
                color=color
            )
            bars.append(bar[0])
            labels.append(scenario)

        # 🔹 Paramétrage du graphique principal (SANS totaux)
        ax.set_title("Comparison of scenarios", fontsize=16, pad=30, fontweight="bold")
        ax.set_ylabel("(%)", fontsize=14)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, len(categories))
        ax.set_xticks(x_positions + (len(scenarios) - 1) * bar_width / 2)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

        # 🔹 2️⃣ Création d'une figure séparée pour la légende
        fig_legend, ax_legend = plt.subplots(figsize=(5, 3))
        ax_legend.axis("off")
        ax_legend.legend(
            bars, labels,
            title="Scenarios",
            fontsize=10,
            loc="center"
        )
        plt.tight_layout()
        plt.show()

        # 🔹 3️⃣ Création d'une figure pour afficher les totaux
        fig_totals, ax_totals = plt.subplots(figsize=(12, 7))

        for i, scenario in enumerate(scenarios):
            # Déterminer la couleur pour ce scénario (même logique)
            if scenario_color_map is not None:
                color = scenario_color_map.get(scenario.strip().lower(), "#000000")
            else:
                color = generic_colors[i % len(generic_colors)]

            bar = ax_totals.bar(
                x_positions + i * bar_width,
                normalized_data[scenario],
                bar_width,
                color=color
            )

        # 🔹 Ajouter les totaux au-dessus des barres
        for i, scenario in enumerate(scenarios):
            for j, value in enumerate(normalized_data[scenario]):
                total_value = total_impact_table.loc[categories[j], scenario]
                formatted_value = (
                    f"{total_value:.2f}" if 0.01 <= abs(total_value) <= 1000 else f"{total_value:.2E}"
                )
                ax_totals.text(
                    x_positions[j] + i * bar_width,
                    value + 2,  # Décalage vertical pour éviter les superpositions
                    formatted_value,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="black",
                    fontweight="bold",
                    rotation=90 if abs(total_value) < 0.01 else 0
                )

        # 🔹 Paramétrage du graphique AVEC les totaux
        ax_totals.set_title("Comparison of scenarios", fontsize=16, pad=80, fontweight="bold")
        ax_totals.set_ylabel("(%)", fontsize=14)
        ax_totals.set_ylim(0, 100)
        ax_totals.set_xlim(-0.5, len(categories))
        ax_totals.set_xticks(x_positions + (len(scenarios) - 1) * bar_width / 2)
        ax_totals.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax_totals.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erreur lors de la génération du graphique : {e}")






def plot_relative_contribution_by_scenario(scenario_tables):
    """
    Génère un diagramme en barres empilées pour chaque scénario,
    en optimisant l'affichage pour 17 catégories d'impact.
    Affiche également la légende dans un graphique séparé.
    La couleur de chaque contribution est attribuée selon l'option suivante :
      - Si le DataFrame global 'contributions_colors' contient des données, 
        la couleur est récupérée depuis sa colonne 'Hex Code'.
      - Sinon, on utilise une palette générique.
    """
    try:
        global contributions_colors  # Utilisation de la variable globale

        # 🔹 Identifier toutes les contributions uniques dans tous les scénarios
        all_contributions = set()
        for table in scenario_tables.values():
            contribution_columns = [col for col in table.columns if col.startswith("%") and col != "%Total d'impact"]
            all_contributions.update(contribution_columns)

        # 🔹 Attribution des couleurs en fonction de contributions_colors
        contribution_colors = {}
        if contributions_colors is not None and not contributions_colors.empty:
            # Option 2 : On utilise les codes hexadécimaux spécifiques
            for contribution in sorted(all_contributions):
                # Retirer le préfixe "%" pour obtenir le nom de la contribution
                contrib_name = contribution.replace("%", "").strip()
                # Recherche insensible à la casse dans le DataFrame contributions_colors
                row = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_name.lower()]
                if not row.empty:
                    hex_code = row["Hex Code"].iloc[0]
                else:
                    # Valeur par défaut si aucune correspondance trouvée
                    hex_code = "#000000"
                contribution_colors[contribution] = hex_code
        else:
            # Option 1 : On utilise des couleurs génériques
            generic_colors = plt.get_cmap("tab10").colors
            for i, contribution in enumerate(sorted(all_contributions)):
                contribution_colors[contribution] = generic_colors[i % len(generic_colors)]

        # 🔹 Vérifier qu'on a bien plusieurs scénarios
        if len(scenario_tables) == 0:
            print("Aucun scénario trouvé.")
            return

        # 🔹 Générer un graphique séparé pour chaque scénario
        for scenario, table in scenario_tables.items():
            fig, ax = plt.subplots(figsize=(16, 8))  # Largeur augmentée pour 17 catégories
            bars = []   # Stocker les handles pour la légende
            labels = [] # Stocker les noms des contributions

            # 🔹 Sélectionner les colonnes de contributions en pourcentage et convertir en float
            contribution_columns = [col for col in table.columns if col.startswith("%") and col != "%Total d'impact"]
            percentage_data = table[contribution_columns].astype(float)  # Conversion en float

            categories = table["Catégorie d'impact"]
            total_impact = table["Total d'impact"].astype(float)  # Lecture des totaux
            bar_width = 0.6  # Ajustement de la largeur des barres pour éviter les chevauchements
            bottom_stack_positive = np.zeros(len(categories))  # Base pour contributions positives
            bottom_stack_negative = np.zeros(len(categories))  # Base pour contributions négatives

            # 🔹 Ajouter les contributions une par une
            for contribution in contribution_columns:
                values = percentage_data[contribution].values  # Convertir en numpy array

                # Séparer les contributions positives et négatives
                positive_values = np.where(values > 0, values, 0)
                negative_values = np.where(values < 0, values, 0)

                # Tracer les contributions positives (vers le haut)
                if np.any(positive_values):
                    bar = ax.bar(
                        categories,
                        positive_values,
                        bar_width,
                        bottom=bottom_stack_positive,
                        color=contribution_colors[contribution],
                    )
                    bars.append(bar[0])  # Stocker un exemple de barre pour la légende
                    labels.append(contribution.replace("%", ""))  # Nettoyer le label
                    bottom_stack_positive += positive_values  # Mise à jour de la pile positive

                # Tracer les contributions négatives (vers le bas)
                if np.any(negative_values):
                    ax.bar(
                        categories,
                        negative_values,
                        bar_width,
                        bottom=bottom_stack_negative,
                        color=contribution_colors[contribution],
                        label="_nolegend_"
                    )
                    bottom_stack_negative += negative_values  # Mise à jour de la pile négative

            # 🔹 Ajouter une ligne horizontale pour séparer les valeurs positives et négatives
            ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)

            # 🔹 Ajouter les valeurs du total d'impact au-dessus de chaque barre
            for i, total in enumerate(total_impact):
                formatted_value = f"{total:.2f}" if 0.01 <= abs(total) <= 1000 else f"{total:.2E}"
                ax.text(
                    i,  # Position x
                    max(bottom_stack_positive[i], 0) + 1,  # Position y (au-dessus de la pile positive)
                    formatted_value,  # Texte affiché
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                    fontweight="bold"
                )

            # 🔹 Personnalisation du graphique
            ax.set_title(f"Relative contributions for the scenario : {scenario.capitalize()}", fontsize=16, pad=40, fontweight="bold")
            ax.set_ylabel("(%)", fontsize=12)
            ax.set_ylim(min(bottom_stack_negative) - 5, max(bottom_stack_positive))
            ax.set_xlim(-0.5, len(categories) - 0.5)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # 🔹 Affichage du graphique principal sans légende
            plt.tight_layout()
            plt.show()

            # 🔹 Création d'une figure séparée pour la légende
            fig_legend, ax_legend = plt.subplots(figsize=(5, 3))
            ax_legend.axis("off")
            ax_legend.legend(
                bars, labels,
                title="Contributions",
                fontsize=10,
                loc="center"
            )
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Erreur lors de la génération des graphiques : {e}")





# Fonction pour énérer des graphiques empilés horizontales en pourcentage pour chaque scénario


def plot_relative_contribution_by_scenario_horizontal(scenario_tables):
    """
    Génère un diagramme en barres empilées horizontalement pour chaque scénario,
    en optimisant l'affichage pour 17 catégories d'impact.
    Ajoute également les valeurs de 'Total d'impact' à droite de chaque barre.
    La couleur de chaque contribution est attribuée selon l'option suivante :
      - Si le DataFrame global 'contributions_colors' contient des données,
        la couleur est récupérée depuis sa colonne 'Hex Code'.
      - Sinon, on utilise une palette générique.
    """
    try:
        global contributions_colors  # Utilisation de la variable globale

        # 🔹 Identifier toutes les contributions uniques dans tous les scénarios
        all_contributions = set()
        for table in scenario_tables.values():
            contribution_columns = [col for col in table.columns if col.startswith("%") and col != "%Total d'impact"]
            all_contributions.update(contribution_columns)

        # 🔹 Attribution des couleurs en fonction de contributions_colors
        contribution_color_map = {}
        if contributions_colors is not None and not contributions_colors.empty:
            # Utiliser les codes hexadécimaux spécifiques
            for contribution in sorted(all_contributions):
                contrib_name = contribution.replace("%", "").strip()
                row = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_name.lower()]
                if not row.empty:
                    hex_code = row["Hex Code"].iloc[0]
                else:
                    hex_code = "#000000"  # Couleur par défaut
                contribution_color_map[contribution] = hex_code
        else:
            # Utiliser des couleurs génériques
            generic_colors = plt.get_cmap("tab10").colors
            for i, contribution in enumerate(sorted(all_contributions)):
                contribution_color_map[contribution] = generic_colors[i % len(generic_colors)]

        # 🔹 Vérifier qu'on a bien plusieurs scénarios
        if len(scenario_tables) == 0:
            print("Aucun scénario trouvé.")
            return

        # 🔹 Générer un graphique séparé pour chaque scénario
        for scenario, table in scenario_tables.items():
            fig, ax = plt.subplots(figsize=(10, 12))  # Largeur réduite, hauteur augmentée pour affichage horizontal

            # Sélectionner les colonnes de contributions en pourcentage et convertir en float
            contribution_columns = [col for col in table.columns if col.startswith("%") and col != "%Total d'impact"]
            percentage_data = table[contribution_columns].astype(float)

            categories = table["Catégorie d'impact"]
            total_impact = table["Total d'impact"].astype(float)
            bar_height = 0.6  # Hauteur des barres
            left_stack_positive = np.zeros(len(categories))  # Base pour contributions positives
            left_stack_negative = np.zeros(len(categories))  # Base pour contributions négatives

            # Ajouter les contributions une par une
            for contribution in contribution_columns:
                values = percentage_data[contribution].values

                # Séparer les contributions positives et négatives
                positive_values = np.where(values > 0, values, 0)
                negative_values = np.where(values < 0, values, 0)

                # Tracer les contributions positives (vers la droite)
                if np.any(positive_values):
                    ax.barh(
                        categories,
                        positive_values,
                        bar_height,
                        label=contribution.replace("%", ""),
                        left=left_stack_positive,
                        color=contribution_color_map[contribution]
                    )
                    left_stack_positive += positive_values

                # Tracer les contributions négatives (vers la gauche)
                if np.any(negative_values):
                    ax.barh(
                        categories,
                        negative_values,
                        bar_height,
                        label="_nolegend_",
                        left=left_stack_negative,
                        color=contribution_color_map[contribution]
                    )
                    left_stack_negative += negative_values

            # Ajouter une ligne verticale pour séparer les valeurs positives et négatives
            ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)

            # 🔹 Ajouter les valeurs du total d'impact à droite de chaque barre
            for i, total in enumerate(total_impact):
                formatted_value = f"{total:.2f}" if 0.01 <= abs(total) <= 1000 else f"{total:.2E}"
                ax.text(
                    max(left_stack_positive[i], 0) + 5,  # Position x (à droite de la pile positive)
                    i,  # Position y
                    formatted_value,  # Texte affiché
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="black",
                    fontweight="bold"
                )

            # Personnalisation du graphique
            ax.set_title(f"Relative contributions for the scenario : {scenario.capitalize()}", fontsize=16, pad=30, fontweight="bold")
            ax.set_xlabel("(%)", fontsize=12)
            ax.set_xlim(min(left_stack_negative) - 5, max(left_stack_positive))
            ax.set_ylim(-0.5, len(categories) - 0.5)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories, fontsize=12)
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            plt.tight_layout()
            plt.show()  # Affichage du graphique

    except Exception as e:
        print(f"Erreur lors de la génération des graphiques : {e}")












def plot_stacked_bar_by_category(percentage_table, total_impact_table):
    """
    Génère un graphique empilé pour chaque catégorie d'impact.
    Chaque graphique compare les contributions de chaque scénario pour une catégorie donnée.
    Les totaux des scénarios sont affichés au-dessus de chaque barre.
    La couleur de chaque contribution est attribuée selon l'option suivante :
      - Si le DataFrame global 'contributions_colors' contient des données,
        la couleur est récupérée depuis sa colonne 'Hex Code'.
      - Sinon, on utilise une palette générique.
    """
    try:
        global contributions_colors  # On utilise la variable globale

        # 🔹 Récupérer les catégories et scénarios
        categories = percentage_table.index
        scenarios = percentage_table.columns.levels[0]
        contributions = percentage_table.columns.levels[1]

        # 🔹 Normaliser les noms des scénarios pour éviter les erreurs
        scenario_labels = {scenario: scenario.split("(")[-1].replace(")", "").strip().lower() for scenario in scenarios}
        total_impact_labels = {sc.lower(): sc for sc in total_impact_table.columns}  # Normaliser les colonnes

        # 🔹 Vérifier la correspondance des scénarios
        for scenario in scenario_labels.values():
            if scenario not in total_impact_labels:
                print(f"⚠ Avertissement : Le scénario '{scenario}' n'est pas trouvé dans total_impact_table !")

        # 🔹 Créer une mapping pour la couleur des contributions
        contrib_color_map = {}
        # On considère ici l'ensemble des contributions tel que présent dans le DataFrame
        all_contributions = sorted(contributions)
        if contributions_colors is not None and not contributions_colors.empty:
            # Utiliser les codes hexadécimaux spécifiques
            for contribution in all_contributions:
                # On suppose que les noms dans le DataFrame sont identiques à ceux du MultiIndex
                # On retire le "%" s'il existe ou on normalise en minuscule
                contrib_name = contribution.replace("%", "").strip().lower()
                row = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_name]
                if not row.empty:
                    hex_code = row["Hex Code"].iloc[0]
                else:
                    hex_code = "#000000"  # Couleur par défaut si non trouvée
                contrib_color_map[contribution] = hex_code
        else:
            # Utiliser une palette générique
            generic_colors = plt.get_cmap("tab10").colors
            for i, contribution in enumerate(all_contributions):
                contrib_color_map[contribution] = generic_colors[i % len(generic_colors)]

        # 🔹 Générer un graphique pour chaque catégorie d'impact
        for category in categories:
            # Extraire les données pour cette catégorie
            category_data = percentage_table.loc[category]

            # Définition des positions des barres
            x_positions = np.arange(len(scenarios))
            bar_width = max(0.2, min(0.8, 1.5 / len(scenarios)))

            # 🔹 Initialiser la figure principale
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = []   # Stocker les handles pour la légende
            labels = [] # Stocker les contributions

            # 🔹 Initialiser les bases pour empiler les contributions positives et négatives
            bottom_stack_positive = np.zeros(len(scenarios))
            bottom_stack_negative = np.zeros(len(scenarios))

            # 🔹 Ajouter les contributions une par une
            for contribution in contributions:
                # Pour chaque scénario, extraire la valeur correspondant à la contribution courante
                contribution_values = np.array([
                    category_data.get((scenario, contribution), 0) for scenario in scenarios
                ])

                # Séparer les contributions positives et négatives
                positive_values = np.where(contribution_values > 0, contribution_values, 0)
                negative_values = np.where(contribution_values < 0, contribution_values, 0)

                # Ajouter une barre empilée pour les valeurs positives
                if np.any(positive_values):
                    bar = ax.bar(
                        x_positions,
                        positive_values,
                        width=bar_width,
                        bottom=bottom_stack_positive,
                        color=contrib_color_map.get(contribution, "#000000")
                    )
                    bars.append(bar[0])
                    labels.append(contribution)
                    bottom_stack_positive += positive_values

                # Ajouter une barre empilée pour les valeurs négatives
                if np.any(negative_values):
                    ax.bar(
                        x_positions,
                        negative_values,
                        width=bar_width,
                        bottom=bottom_stack_negative,
                        color=contrib_color_map.get(contribution, "#000000"),
                        label="_nolegend_"
                    )
                    bottom_stack_negative += negative_values

            # 🔹 Ajouter les valeurs totales au-dessus de chaque barre
            for i, scenario in enumerate(scenarios):
                scenario_cleaned = scenario_labels[scenario]
                if scenario_cleaned in total_impact_labels:
                    total_value = total_impact_table.loc[category, total_impact_labels[scenario_cleaned]]
                    formatted_value = f"{total_value:.2f}" if 0.01 <= abs(total_value) <= 1000 else f"{total_value:.2E}"
                    ax.text(
                        x_positions[i],
                        max(bottom_stack_positive[i], 0) + 5,
                        formatted_value,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="black",
                        fontweight="bold"
                    )

            # 🔹 Personnalisation des axes et du graphique
            ax.set_title(f"Analysis of contributions - Category : {category}", fontsize=16, pad=40, fontweight="bold")
            ax.set_ylabel("(%)", fontsize=14)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([scenario_labels[s] for s in scenarios], rotation=45, ha="right", fontsize=12)
            ax.set_xlim(x_positions[0] - (bar_width / 2), x_positions[-1] + (bar_width / 2))
            ax.set_ylim(min(bottom_stack_negative) - 5, max(bottom_stack_positive))
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # 🔹 Affichage du graphique principal sans légende
            plt.tight_layout()
            plt.show()

            # 🔹 Création d'une figure séparée pour la légende
            fig_legend, ax_legend = plt.subplots(figsize=(5, 3))
            ax_legend.axis("off")
            ax_legend.legend(
                bars, labels,
                title="Contributions",
                fontsize=10,
                loc="center"
            )
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"❌ Erreur lors de la génération des graphiques : {e}")











def plot_combined_graph_with_scenario_hatches(percentage_table, total_impact_table, show_totals=True):
    """
    Génère un graphique combiné avec toutes les catégories sur un seul graphique.
    Deux versions sont générées : 
      - Une avec les valeurs des totaux affichées au-dessus des barres.
      - Une sans les valeurs des totaux.
    La couleur de chaque contribution est attribuée selon l'option suivante :
      - Si le DataFrame global 'contributions_colors' contient des données,
        la couleur est récupérée depuis sa colonne 'Hex Code'.
      - Sinon, on utilise une palette générique.
    """
    try:
        global contributions_colors  # Utilisation de la variable globale
        
        # 🔹 Récupérer les catégories d'impact et scénarios
        categories = percentage_table.index
        scenarios = percentage_table.columns.levels[0]  # Scénarios dans percentage_table
        contributions = percentage_table.columns.levels[1]  # Contributions

        # Extraire uniquement le contenu entre parenthèses sans modifier la casse pour les scénarios
        scenario_labels = {
            scenario: scenario[scenario.find("(") + 1 : scenario.find(")")].strip()
            for scenario in scenarios
        }

        # Vérifier la correspondance avec total_impact_table
        total_impact_labels = {sc.strip(): sc for sc in total_impact_table.columns}

        # Associer les scénarios extraits avec ceux de total_impact_table
        scenario_labels_corrected = {
            original: total_impact_labels.get(cleaned, None)
            for original, cleaned in scenario_labels.items()
        }

        # 🔹 Définir la largeur des barres et les positions
        num_scenarios = len(scenarios)
        bar_width = 0.9 / num_scenarios
        x_positions = np.arange(len(categories))

        # 🔹 Définir un dictionnaire de couleurs pour les contributions
        # Si contributions_colors est fourni, on utilise les codes hexadécimaux,
        # sinon on utilise une palette générique.
        contrib_color_map = {}
        if contributions_colors is not None and not contributions_colors.empty:
            for contribution in sorted(contributions):
                # On retire le préfixe "%" s'il existe et on normalise en minuscule
                contrib_name = contribution.replace("%", "").strip().lower()
                row = contributions_colors[contributions_colors["Contributions"].str.lower() == contrib_name]
                if not row.empty:
                    hex_code = row["Hex Code"].iloc[0]
                else:
                    hex_code = "#000000"  # Valeur par défaut si non trouvée
                contrib_color_map[contribution] = hex_code
        else:
            generic_colors = plt.get_cmap("tab10").colors
            for i, contribution in enumerate(sorted(contributions)):
                contrib_color_map[contribution] = generic_colors[i % len(generic_colors)]

        # 🔹 Définir des motifs (hatch patterns) pour chaque scénario
        scenario_hatches = ["//", "oo", "..", "xx", "--", "ii", "hh"]
        scenario_hatch_dict = {scenario: scenario_hatches[i % len(scenario_hatches)] for i, scenario in enumerate(scenarios)}

        # 🔹 Stockage des valeurs finales des bottom stacks
        all_bottom_stack_negative = []
        all_bottom_stack_positive = []

        # 🔹 Initialiser la figure
        fig, ax = plt.subplots(figsize=(16, 8))

        # 🔹 Parcourir chaque scénario et tracer les barres empilées
        for i, scenario in enumerate(scenarios):
            bottom_stack_positive = np.zeros(len(categories))
            bottom_stack_negative = np.zeros(len(categories))

            for contribution in contributions:
                contribution_values = np.array([
                    percentage_table.loc[category, (scenario, contribution)]
                    if (scenario, contribution) in percentage_table.columns else 0
                    for category in categories
                ])

                # Séparer les contributions positives et négatives
                positive_values = np.where(contribution_values > 0, contribution_values, 0)
                negative_values = np.where(contribution_values < 0, contribution_values, 0)

                # Ajouter les contributions positives
                if np.any(positive_values):
                    ax.bar(
                        x_positions + i * bar_width - 0.4,
                        positive_values,
                        width=bar_width,
                        color=contrib_color_map.get(contribution, "#000000"),
                        bottom=bottom_stack_positive,
                        hatch=scenario_hatch_dict[scenario],
                        edgecolor="black"
                    )
                    bottom_stack_positive += positive_values

                # Ajouter les contributions négatives
                if np.any(negative_values):
                    ax.bar(
                        x_positions + i * bar_width - 0.4,
                        negative_values,
                        width=bar_width,
                        color=contrib_color_map.get(contribution, "#000000"),
                        bottom=bottom_stack_negative,
                        hatch=scenario_hatch_dict[scenario],
                        edgecolor="black"
                    )
                    bottom_stack_negative += negative_values

            # 🔹 Stocker les valeurs finales après empilement
            all_bottom_stack_negative.append(bottom_stack_negative)
            all_bottom_stack_positive.append(bottom_stack_positive)

            # 🔹 Ajouter les valeurs totales au-dessus des barres (optionnel)
            if show_totals:
                for j, category in enumerate(categories):
                    scenario_cleaned = scenario_labels[scenario]  # Nom normalisé
                    if scenario_cleaned in total_impact_labels:
                        total_value = total_impact_table.loc[category, total_impact_labels[scenario_cleaned]]
                        if 0.01 <= abs(total_value) <= 1000:
                            formatted_value = f"{total_value:.2f}"
                            rotation_angle = 0
                        else:
                            formatted_value = f"{total_value:.2E}"
                            rotation_angle = 90
                        ax.text(
                            x_positions[j] + i * bar_width - 0.35,
                            max(bottom_stack_positive[j], 0) + 2,
                            formatted_value,
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color="black",
                            fontweight="bold",
                            rotation=rotation_angle
                        )

        # 🔹 Calcul du maximum et du minimum global
        maximum_al = np.max(all_bottom_stack_positive)
        minimum_al = np.min(all_bottom_stack_negative)

        # 🔹 Configuration des axes et du graphique
        ax.set_title("Comparative Analysis of Scenarios: Relative Impact Contributions by Category", fontsize=16, pad=80, fontweight="bold")
        ax.set_ylabel("(%)", fontsize=16)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=16)
        ax.set_xlim(x_positions[0] - 0.8, x_positions[-1] + 0.5)
        ax.set_ylim(minimum_al - 2, maximum_al)
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)

        # 🔹 Légende combinée : Contributions et Scénarios
        legend_handles = []
        legend_labels = []

        # Ajouter les contributions à la légende
        for contribution, color in contrib_color_map.items():
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=10))
            legend_labels.append(contribution)

        # Ajouter les scénarios à la légende avec des rectangles remplis de motifs
        for scenario, hatch in zip(scenario_labels.values(), scenario_hatch_dict.values()):
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatch, lw=1.5)
            )
            legend_labels.append(scenario)

        # 🔹 Créer une nouvelle figure uniquement pour la légende
        fig_legend, ax_legend = plt.subplots(figsize=(5, 3))
        ax_legend.axis("off")
        ax_legend.legend(
            legend_handles,
            legend_labels,
            title="Contributions and Scenarios",
            fontsize=10,
            loc="center"
        )
        
        plt.tight_layout()
        plt.show()

        # 🔹 Générer une version sans les valeurs totales si demandé
        if show_totals:
            plot_combined_graph_with_scenario_hatches(percentage_table, total_impact_table, show_totals=False)

        return np.array(all_bottom_stack_negative), np.array(all_bottom_stack_positive)

    except Exception as e:
        print(f"❌ Erreur lors de la génération du graphique combiné : {e}")
        return None, None



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
    plt.show()

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

















# Exemple d'utilisation
if __name__ == "__main__":
    excel_file_name = "données_test.xlsx"
    file_path = find_file_path(excel_file_name)
    if file_path:
        print(f"Fichier trouvé : {file_path}")
    else:
        print(f"Le fichier '{excel_file_name}' est introuvable.")
    sheet_name = "Feuil2"

    # Analyse du fichier Excel et génération des tableaux
    initial_table, combined_table, total_impact_table, scenario_names_cleaned = analyze_excel_and_generate_tables(
        file_path, sheet_name=sheet_name
    )
    # Générer les tableaux de couleurs pour contributions et scenarios depuis la feuille "Color catalog"
    contributions_colors, scenario_colors = generate_color_catalog_tables(file_path, sheet_name="Color catalog")
    
    if initial_table is not None and combined_table is not None:
        print("\nLes deux tableaux principaux ont été générés avec succès.")

        # Génération des tableaux par scénario
        scenario_tables = generate_tables_by_scenario(initial_table, scenario_names_cleaned)

        # Génération du tableau en pourcentage
        percentage_table = generate_percentage_table(initial_table, total_impact_table)
        if percentage_table is not None:
            print("\nTableau en pourcentage généré avec succès :")
            print(percentage_table.head())

          
       # Générer le graphique
        generate_table_graph(total_impact_table, "Total Impact Distribution by Category and Scenario")
    

        # Génération de l'histogramme côte à côte (référentiel 100 %)
        plot_comparison_bar_chart(total_impact_table)

            

        # Génération des graphiques empilés en pourcentage par scénario
        plot_relative_contribution_by_scenario(scenario_tables)

      

        # Génération des graphiques empilés en pourcentage horizontaux par scénario
        plot_relative_contribution_by_scenario_horizontal(scenario_tables)

        # Génération des graphiques par catégorie
        plot_stacked_bar_by_category(percentage_table, total_impact_table)

            
        # Génération du graphique combiné pour toutes les catégories avec motifs pour les scénarios
        plot_combined_graph_with_scenario_hatches(percentage_table, total_impact_table)




