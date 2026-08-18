# Démonstrateur de scoring de crédit explicable

Application Streamlit locale destinée au portfolio de Julien Campagnaud.

Cette démonstration modernise l'inférence d'un projet réalisé dans le cadre de la formation Data Scientist d'OpenClassrooms. Elle ne constitue ni une mission client ni un outil destiné à une décision réelle de crédit.

## État actuel

La première version fonctionnelle permet de :

- choisir entre trois profils issus du jeu public dé-identifié, situés aux 10e, 50e et 90e centiles du risque historique ;
- consulter une carte résumant les principales catégories d'informations reçues par le modèle ;
- calculer localement leur probabilité calibrée de défaut ;
- lire une synthèse de leur position relative et des principales influences ;
- afficher les six facteurs locaux les plus influents ;
- consulter un glossaire contextuel des facteurs affichés ;
- situer l'âge, le revenu, le crédit et l'annuité dans la population de référence ;
- consulter la méthode, les limites et les détails techniques.

Les trois risques affichés sont respectivement 0,39 %, 4,46 % et 17,73 %. Ils décrivent une position relative dans l'échantillon public et ne constituent pas des catégories bancaires.

## Lancer l'application sous Windows

Depuis ce dossier :

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m streamlit run streamlit_app.py
```

Streamlit ouvre normalement l'application dans le navigateur. Sinon, l'adresse locale est affichée dans le terminal, généralement `http://localhost:8501`.

Si l'environnement `.venv` existe déjà :

```powershell
.venv\Scripts\python -m streamlit run streamlit_app.py
```

## Lancer les vérifications

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Les tests vérifient notamment :

- le démarrage de l'application ;
- l'affichage des trois profils ;
- la reproduction des probabilités historiques par le modèle porté ;
- la présence d'explications locales ;
- le rejet propre d'un profil inconnu ;
- l'absence d'identifiant source dans l'artefact public.

## Architecture

```text
streamlit_app.py
    │
    ├── inference.py
    │     ├── artifacts/credit_risk_model.txt
    │     └── artifacts/demo_profiles.json
    │
    └── tests/
          ├── test_app.py
          └── test_inference.py
```

- Le modèle est chargé depuis le format texte natif de LightGBM.
- Les trois dossiers ont été transformés une fois dans l'environnement historique compatible.
- La calibration isotone est conservée sous forme de tableaux numériques et appliquée localement.
- Les contributions explicatives sont recalculées par LightGBM à chaque consultation.
- Aucune API, connexion réseau ou donnée volumineuse n'est nécessaire à l'exécution.

## Reproductibilité et provenance

Les artefacts ont été générés depuis les clones locaux audités des dépôts historiques avec :

```powershell
python "..\..\Audit technique Projet 7\export_portable_demo.py"
```

Ce script doit être exécuté dans un environnement compatible avec les dépendances historiques. La version locale du script d'audit reste séparée du démonstrateur.

Voir [MODEL_CARD.md](MODEL_CARD.md) pour la provenance, les vérifications et les limites du modèle.

## Limites

- Le modèle n'a pas été réentraîné : seule sa chaîne d'inférence a été portée vers un environnement moderne.
- La reproduction des trois prédictions ne remplace pas une nouvelle validation statistique complète.
- Les profils sont fixes et ne permettent aucune saisie de dossier réel.
- Les montants conservent les unités du jeu de données source et ne sont pas présentés comme des euros.
- Le déploiement public reste une démonstration éducative et ne doit pas être utilisé pour une décision réelle.

## Statut de publication

Le code source de cette version modernisée est publié dans un dépôt dédié. L'application est également accessible sur Streamlit Community Cloud :

<https://demonstrateur-scoring-credit.streamlit.app/>

Les dépôts GitHub historiques sont inchangés.
