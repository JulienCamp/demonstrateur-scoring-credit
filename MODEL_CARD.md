# Fiche du modèle — démonstrateur de scoring de crédit

Date : 12 août 2026

## Finalité

Montrer, dans un cadre éducatif, comment une estimation probabiliste peut être calculée puis accompagnée d'éléments d'explication compréhensibles.

Le modèle ne doit pas être utilisé pour accorder, refuser ou recommander un crédit réel.

## Origine

- Projet : Projet 7 de la formation Data Scientist OpenClassrooms.
- Données : jeu public Home Credit Default Risk.
- Modèle historique : LightGBM entraîné avec sur-échantillonnage aléatoire dans une chaîne scikit-learn/imbalanced-learn.
- Calibration : régression isotone historique.
- Interface d'origine : API Flask et tableau de bord Streamlit séparés.
- Statut professionnel : projet de formation, pas mission client.

## Portage réalisé

Le classifieur a été exporté dans le format texte natif de LightGBM. Les trois profils ont été transformés avec la chaîne historique, puis conservés sous forme de vecteurs numériques. Les seuils de la calibration isotone ont également été exportés.

La version moderne charge ces éléments avec LightGBM 4.7.0 et ne dépend plus de Flask, Heroku, scikit-learn, imbalanced-learn, SHAP ou des jeux de données complets.

## Entrées et sorties

- Entrée publique : choix parmi trois profils fixes nommés A, B et C.
- Entrée interne : 158 variables transformées par profil.
- Sortie principale : probabilité calibrée de la classe « défaut ».
- Explication : six contributions locales LightGBM de plus forte amplitude.
- Comparaisons : centiles descriptifs sur quatre caractéristiques lisibles.

## Vérifications réalisées

- Chargement du modèle avec LightGBM 4.7.0 sous Python 3.11.
- Reproduction des trois probabilités brutes historiques à 12 décimales.
- Reproduction des trois probabilités calibrées historiques à 12 décimales.
- Calcul des contributions locales pour les trois profils.
- Tests automatisés du démarrage et de la sélection des profils.
- Test réel du serveur local et de son point de contrôle.
- Contrôle visuel sur écran d'ordinateur et largeur mobile.
- Vérification de l'absence d'identifiant source dans l'artefact du démonstrateur.

Ces contrôles prouvent la fidélité du portage pour les trois exemples. Ils ne constituent pas une nouvelle évaluation globale du modèle.

## Limites connues

- Les performances globales historiques n'ont pas encore été reproduites dans un protocole moderne complet.
- Les biais et l'équité entre groupes n'ont pas été réévalués.
- La calibration historique produit quelques valeurs extrêmes à 0 et 1 sur l'ensemble du jeu de test ; les trois profils de démonstration évitent ces extrêmes.
- Les variables issues de sources externes sont influentes mais décrites de façon limitée dans le jeu de données source.
- Le modèle historique utilise notamment une variable de genre. Elle est conservée pour assurer la fidélité du portage et peut apparaître dans l'explication ; cette présence constitue un signal explicite pour un futur audit d'équité, pas une validation de son usage.
- Les contributions indiquent le fonctionnement du modèle, pas une relation causale.
- Le modèle reflète les données et choix méthodologiques du projet de formation d'origine.

## Conditions minimales avant tout usage réel

Un usage opérationnel demanderait un nouveau cadrage métier, une base juridique adaptée, une validation indépendante, une analyse des biais, une gouvernance des données, des contrôles de sécurité, une surveillance dans le temps, une documentation réglementaire et une intervention humaine effective.

## Statut

Démonstrateur de portfolio dont le code source est publié dans un dépôt dédié. Exécution locale uniquement, sans application publique hébergée. Dépôts historiques préservés.
