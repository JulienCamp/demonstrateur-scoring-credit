# Artefacts du démonstrateur

Ce dossier contient uniquement les éléments nécessaires aux trois profils fixes :

- `credit_risk_model.txt` : booster LightGBM historique, exporté dans son format texte natif ;
- `demo_profiles.json` : vecteurs transformés, calibration, valeurs de contrôle, comparaisons agrégées et métadonnées.

Le fichier JSON ne contient aucun identifiant de dossier source. Les jeux de données complets, la chaîne scikit-learn historique et les modèles sérialisés avec joblib ne sont pas embarqués.

Ces fichiers sont régénérables avec le script local `Audit technique Projet 7/export_portable_demo.py` dans un environnement compatible avec les dépendances historiques.
