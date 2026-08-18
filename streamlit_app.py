"""Démonstrateur local de scoring de crédit explicable."""

import altair as alt
import pandas as pd
import streamlit as st

from inference import load_model, load_payload, predict_profile


APP_TITLE = "Comprendre une estimation de risque de crédit"
DISCLAIMER = (
    "Démonstration éducative issue d'un projet de formation. "
    "Elle n'est pas destinée à une décision réelle de crédit."
)


@st.cache_resource
def cached_resources():
    """Charge une seule fois les artefacts légers de la démonstration."""
    return load_payload(), load_model()


def format_number(value: float, decimals: int = 0) -> str:
    """Formate un nombre avec les conventions françaises."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", " ").replace(".", ",")


def render_factors(prediction) -> None:
    """Affiche les facteurs locaux les plus influents."""
    frame = pd.DataFrame(
        {
            "Facteur": [factor.label for factor in prediction.factors],
            "Description": [factor.description for factor in prediction.factors],
            "Influence": [factor.contribution for factor in prediction.factors],
            "Direction": [factor.direction for factor in prediction.factors],
        }
    )
    frame["Ordre"] = range(len(frame), 0, -1)

    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Influence:Q",
                title="Influence sur le score interne du modèle",
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y("Facteur:N", sort=alt.SortField("Ordre", order="descending"), title=None),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Vers moins de risque", "Vers plus de risque"],
                    range=["#2563EB", "#DC2626"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Facteur:N"),
                alt.Tooltip("Description:N"),
                alt.Tooltip("Direction:N"),
                alt.Tooltip("Influence:Q", format=".3f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Les barres décrivent le sens et l'intensité de l'influence dans "
        "l'échelle interne du modèle. Elles ne représentent pas des points "
        "de probabilité."
    )
    with st.expander("Comprendre les facteurs affichés"):
        for factor in prediction.factors:
            st.markdown(f"**{factor.label}** — {factor.description}")
        st.caption(
            "Ces définitions décrivent les variables disponibles dans le jeu "
            "de données ; elles ne donnent pas d'interprétation causale."
        )


def join_labels(labels: list[str]) -> str:
    """Assemble une courte liste avec une conjonction française."""
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} et {labels[-1]}"


def render_quick_read(prediction) -> None:
    """Résume la position et les principales influences sans surinterpréter."""
    percentile = int(round(prediction.target_quantile * 100))
    lower_factors = [
        factor.label
        for factor in prediction.factors
        if factor.direction == "Vers moins de risque"
    ][:2]
    higher_factors = [
        factor.label
        for factor in prediction.factors
        if factor.direction == "Vers plus de risque"
    ][:2]

    with st.container(border=True):
        st.markdown(":material/quick_reference_all: **Lecture rapide**")
        st.write(
            f"Ce profil se situe autour du **{percentile}e centile** : environ "
            f"{percentile} % des dossiers présentent un risque estimé inférieur "
            f"ou égal, et {100 - percentile} % un risque supérieur."
        )
        if lower_factors:
            lower_verb = "figure" if len(lower_factors) == 1 else "figurent"
            st.write(
                f"Dans le calcul interne, **{join_labels(lower_factors)}** "
                f"{lower_verb} parmi les principales influences vers moins de risque."
            )
        if higher_factors:
            higher_verb = "figure" if len(higher_factors) == 1 else "figurent"
            st.write(
                f"**{join_labels(higher_factors)}** {higher_verb} parmi les principales "
                "influences vers plus de risque."
            )
        st.caption(
            "Il s'agit d'influences statistiques propres au modèle, pas de "
            "relations de cause à effet."
        )


def format_profile_value(item: dict) -> str:
    """Formate une valeur de profil sans lui inventer une unité."""
    value = item["value"]
    unit = item.get("unit")
    if isinstance(value, float):
        decimals = 1 if unit == "ans" else 0
        value = format_number(value, decimals)
    return f"{value} {unit}" if unit else str(value)


def render_profile_card(prediction, payload: dict) -> None:
    """Présente les principales familles d'informations reçues par le modèle."""
    st.subheader(":material/person: Informations du profil")
    st.caption(
        "Une sélection lisible des informations d'origine. Le modèle traitait "
        f"au total {payload['model']['raw_feature_count']} variables d'entrée, "
        f"transformées ensuite en {payload['model']['feature_count']} variables."
    )
    columns = st.columns(3)
    icons = {
        "Situation personnelle": ":material/badge:",
        "Activité et ressources": ":material/work:",
        "Crédit et patrimoine": ":material/account_balance:",
    }
    for column, (group, items) in zip(columns, prediction.profile_summary.items()):
        with column.container(border=True, height="stretch"):
            st.markdown(f"{icons[group]} **{group}**")
            for item in items:
                label = item["label"]
                if item.get("sensitive"):
                    label += " · variable sensible"
                st.caption(label)
                st.write(format_profile_value(item))


def render_comparisons(prediction) -> None:
    """Situe les caractéristiques lisibles dans la population de référence."""
    columns = st.columns(2)
    for index, comparison in enumerate(prediction.comparisons):
        with columns[index % 2]:
            with st.container(border=True):
                decimals = 1 if comparison["key"] == "age" else 0
                st.markdown(f"**{comparison['label']}**")
                st.write(
                    f"{format_number(comparison['value'], decimals)} "
                    f"{comparison['unit']}"
                )
                st.progress(
                    int(round(comparison["percentile"])),
                    text=(
                        f"{comparison['percentile']:.0f}e centile de "
                        "l'échantillon"
                    ),
                )
                st.caption(
                    "Médiane : "
                    f"{format_number(comparison['population_median'], decimals)} "
                    f"{comparison['unit']}"
                )


def render_app() -> None:
    """Affiche le parcours principal de la démonstration."""
    st.set_page_config(
        page_title="Démonstrateur de scoring explicable",
        page_icon="📊",
        layout="wide",
    )

    st.title(APP_TITLE)
    st.caption("Projet de formation modernisé pour le portfolio de Julien Campagnaud")
    st.warning(DISCLAIMER)

    payload, model = cached_resources()
    profile_labels = {
        profile["alias"]: f"{profile['alias']} — {profile['level'].lower()}"
        for profile in payload["profiles"]
    }
    label_to_alias = {label: alias for alias, label in profile_labels.items()}

    with st.sidebar:
        st.header("Dossier de démonstration")
        selected_label = st.selectbox(
            "Choisir un profil",
            list(label_to_alias),
            help=(
                "Trois exemples issus d'un jeu public dé-identifié ont été "
                "choisis à différents niveaux de la distribution historique."
            ),
        )
        st.info(
            "Aucune donnée n'est saisie, transmise ou enregistrée. "
            "Le calcul est effectué localement."
        )

    selected_alias = label_to_alias[selected_label]
    prediction = predict_profile(selected_alias, payload=payload, model=model)

    st.header(f"{prediction.alias} — {prediction.level.lower()}")
    render_profile_card(prediction, payload)

    st.subheader(":material/query_stats: Estimation du modèle")
    risk_column, position_column = st.columns(2)
    with risk_column:
        with st.container(border=True):
            st.metric(
                "Risque estimé de défaut",
                f"{prediction.calibrated_risk:.2%}".replace(".", ","),
            )
            st.caption(
                "Probabilité calibrée calculée localement à partir du modèle "
                "historique porté vers un format moderne."
            )
    with position_column:
        with st.container(border=True):
            percentile = int(round(prediction.target_quantile * 100))
            st.metric("Position relative retenue", f"{percentile}e centile")
            st.progress(percentile)
            st.caption(
                "Position visée lors de la sélection dans les 48 744 dossiers "
                "du jeu de test. Ce n'est pas une catégorie bancaire."
            )

    render_quick_read(prediction)

    st.header("Ce qui influence cette estimation")
    st.write(
        "Le modèle combine de nombreuses variables. Voici les six influences "
        "locales les plus fortes pour ce dossier."
    )
    render_factors(prediction)

    st.header("Situer quelques caractéristiques")
    st.write(
        "Ces repères décrivent la position du dossier dans l'échantillon public. "
        "Les montants restent exprimés dans les unités du jeu de données source."
    )
    render_comparisons(prediction)

    with st.expander("Méthode et limites"):
        st.markdown(
            """
            - Les trois profils proviennent du jeu de données public dé-identifié
              Home Credit Default Risk.
            - Le démonstrateur utilise un modèle LightGBM entraîné pendant la
              formation et une calibration isotone historique.
            - Les profils sont fixes : l'application montre une chaîne
              d'inférence et d'explication, pas un formulaire de demande de crédit.
            - « Faible », « intermédiaire » et « plus élevé » indiquent une
              position relative dans l'échantillon, pas un jugement bancaire.
            - Une probabilité n'est ni une certitude ni une recommandation.
            - Le modèle historique contient notamment une variable de genre.
              Sa présence illustre la nécessité d'un audit d'équité avant tout
              usage sensible ; elle n'est pas dissimulée dans les explications.
            - Un usage réel demanderait une nouvelle validation des performances,
              une analyse des biais, une gouvernance des données, une surveillance
              continue, une intervention humaine et un cadre réglementaire adapté.
            """
        )

    with st.expander("Détails techniques"):
        st.markdown(
            f"""
            - Modèle : LightGBM, 100 arbres, chargé depuis un fichier texte natif.
            - Entrée : {payload['model']['feature_count']} variables déjà
              transformées pour chacun des trois exemples.
            - Explication locale : contributions calculées directement par
              LightGBM au moment de la consultation.
            - Architecture : une seule application locale, sans API distante.
            - Taille des artefacts : moins de 0,5 Mo.
            """
        )

    st.divider()
    st.caption(
        "Démonstration éducative — données publiques — aucune décision réelle "
        "de crédit."
    )


if __name__ == "__main__":
    render_app()
