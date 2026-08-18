"""Chargement et inférence portable pour les trois profils de démonstration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"
PROFILES_PATH = ARTIFACTS_DIR / "demo_profiles.json"
MODEL_PATH = ARTIFACTS_DIR / "credit_risk_model.txt"


FEATURE_LABELS = {
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "AMT_GOODS_PRICE": "Prix du bien financé",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi",
    "AMT_CREDIT": "Montant du crédit",
    "TOTALAREA_MODE": "Surface du logement",
    "OWN_CAR_AGE": "Âge du véhicule",
    "AMT_ANNUITY": "Annuité du crédit",
    "DAYS_BIRTH": "Âge",
    "DAYS_ID_PUBLISH": "Ancienneté du document d'identité",
    "FLAG_DOCUMENT_3": "Document justificatif n° 3",
    "NAME_EDUCATION_TYPE_Higher education": "Niveau d'études : supérieur",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": (
        "Niveau d'études : secondaire"
    ),
    "OCCUPATION_TYPE_Core staff": "Profession : personnel qualifié",
    "NAME_FAMILY_STATUS_Married": "Situation familiale : marié(e)",
    "CODE_GENDER_F": "Genre déclaré : femme",
    "FLAG_OWN_CAR_N": "Véhicule non déclaré",
    "NAME_INCOME_TYPE_Working": "Revenu d'activité salariée",
}

FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_1": (
        "Score normalisé provenant d'une source externe. Le jeu de données "
        "ne documente pas précisément sa composition."
    ),
    "EXT_SOURCE_2": (
        "Score normalisé provenant d'une source externe. Le jeu de données "
        "ne documente pas précisément sa composition."
    ),
    "EXT_SOURCE_3": (
        "Score normalisé provenant d'une source externe. Le jeu de données "
        "ne documente pas précisément sa composition."
    ),
    "AMT_GOODS_PRICE": "Prix du bien associé au financement demandé.",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi au moment de la demande.",
    "AMT_CREDIT": "Montant du crédit demandé.",
    "TOTALAREA_MODE": "Indicateur normalisé de la surface du logement.",
    "OWN_CAR_AGE": "Ancienneté du véhicule déclaré.",
    "AMT_ANNUITY": "Montant de l'annuité prévue pour le crédit.",
    "DAYS_BIRTH": "Âge de la personne au moment de la demande.",
    "DAYS_ID_PUBLISH": (
        "Ancienneté du document d'identité présenté lors de la demande."
    ),
    "FLAG_DOCUMENT_3": "Indique si le justificatif référencé n° 3 a été fourni.",
    "NAME_EDUCATION_TYPE_Higher education": (
        "Modalité indiquant un niveau d'études supérieures."
    ),
    "NAME_EDUCATION_TYPE_Secondary / secondary special": (
        "Modalité indiquant un niveau d'études secondaires."
    ),
    "OCCUPATION_TYPE_Core staff": (
        "Modalité de profession correspondant au personnel qualifié."
    ),
    "NAME_FAMILY_STATUS_Married": (
        "Modalité indiquant une situation familiale mariée."
    ),
    "CODE_GENDER_F": (
        "Genre féminin déclaré. Cette variable sensible nécessiterait un "
        "audit d'équité avant tout usage réel."
    ),
    "FLAG_OWN_CAR_N": "Modalité indiquant qu'aucun véhicule n'est déclaré.",
    "NAME_INCOME_TYPE_Working": (
        "Modalité indiquant des revenus provenant d'une activité salariée."
    ),
}


@dataclass(frozen=True)
class Factor:
    label: str
    description: str
    contribution: float
    direction: str


@dataclass(frozen=True)
class Prediction:
    alias: str
    level: str
    target_quantile: float
    raw_risk: float
    calibrated_risk: float
    factors: tuple[Factor, ...]
    comparisons: tuple[dict[str, Any], ...]
    profile_summary: dict[str, list[dict[str, Any]]]


def load_payload(path: Path = PROFILES_PATH) -> dict[str, Any]:
    """Charge le petit artefact JSON sans données personnelles directes."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_model(path: Path = MODEL_PATH) -> lgb.Booster:
    """Charge le booster LightGBM depuis son format texte portable."""
    return lgb.Booster(model_file=str(path))


def calibrate_probability(raw_probability: float, payload: dict[str, Any]) -> float:
    """Applique la calibration isotone historique exportée."""
    calibration = payload["model"]["calibration"]
    x_values = np.asarray(calibration["x_thresholds"], dtype=float)
    y_values = np.asarray(calibration["y_thresholds"], dtype=float)
    return float(
        np.interp(
            raw_probability,
            x_values,
            y_values,
            left=y_values[0],
            right=y_values[-1],
        )
    )


def friendly_feature_name(name: str) -> str:
    """Traduit les variables les plus visibles en libellés compréhensibles."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return name.replace("_", " ").capitalize()


def friendly_feature_description(name: str) -> str:
    """Explique une variable visible sans extrapoler sa signification."""
    return FEATURE_DESCRIPTIONS.get(
        name,
        "Variable issue du dossier et utilisée par le modèle après prétraitement.",
    )


def predict_profile(
    alias: str,
    *,
    payload: dict[str, Any] | None = None,
    model: lgb.Booster | None = None,
    factor_count: int = 6,
) -> Prediction:
    """Calcule le risque et les facteurs d'un profil de démonstration."""
    payload = payload or load_payload()
    model = model or load_model()

    try:
        profile = next(item for item in payload["profiles"] if item["alias"] == alias)
    except StopIteration as error:
        raise ValueError(f"Profil inconnu : {alias}") from error

    features = np.asarray(profile["transformed_features"], dtype=float).reshape(1, -1)
    raw_risk = float(model.predict(features)[0])
    calibrated_risk = calibrate_probability(raw_risk, payload)
    contributions = np.asarray(model.predict(features, pred_contrib=True)[0], dtype=float)

    feature_names = payload["model"]["feature_names"]
    feature_contributions = contributions[:-1]
    ordered = np.argsort(np.abs(feature_contributions))[::-1][:factor_count]
    factors = tuple(
        Factor(
            label=friendly_feature_name(feature_names[index]),
            description=friendly_feature_description(feature_names[index]),
            contribution=float(feature_contributions[index]),
            direction=(
                "Vers plus de risque"
                if feature_contributions[index] > 0
                else "Vers moins de risque"
            ),
        )
        for index in ordered
    )

    return Prediction(
        alias=profile["alias"],
        level=profile["level"],
        target_quantile=float(profile["target_quantile"]),
        raw_risk=raw_risk,
        calibrated_risk=calibrated_risk,
        factors=factors,
        comparisons=tuple(profile["comparisons"]),
        profile_summary=profile["profile_summary"],
    )
