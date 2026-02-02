# Projet_IA_GIBERT_SIMONNET_IPSA_B2_2025-2026
Projet académique en Intelligence Artificielle – IPSA B2 (2025–2026) Ce dépôt regroupe les travaux réalisés dans le cadre du projet d’IA mené par Gabriel Gibert et Kyllian Simonnet en deuxième année de bachelor à l’IPSA.

Description des fichiers du projet

1️⃣ Génération des données pour la classification

get_crops_for_train_classifier.py :

Ce script permet de générer le dossier crops_aircraft, contenant des images d’avions recadrées (croppées) à partir des images du jeu de données d’entraînement.

Les recadrages sont réalisés à partir des annotations XML du dataset de train.

Chaque avion est automatiquement classé dans un sous-dossier correspondant à sa classe.

Ce dossier constitue l’entrée du modèle de classification.

2️⃣ Entraînement du modèle de classification

train_classifier_efficientnetB1.py :

Ce script permet d’entraîner le modèle de classification d’avions, basé sur EfficientNet-B1.

Le modèle apprend à reconnaître la classe d’un avion à partir des images croppées.

À l’issue de l’entraînement, le modèle est sauvegardé sous le nom :
classifier_model.pth

3️⃣ Entraînement du modèle de détection

train_bounding_boxes.py :

Ce script permet d’entraîner le modèle de détection d’objets, chargé de localiser les avions dans les images.

Le modèle apprend à prédire les bounding boxes autour des avions.

Le modèle entraîné est sauvegardé sous le nom :
bounding_boxes_model.pth

4️⃣ Génération du fichier d’évaluation

evaluation_json.py :

Ce script permet de générer un fichier JSON conforme au format attendu pour la vérification automatique sur la plateforme d’évaluation.

Le fichier généré est nommé :
result_evaluation.json

Ce fichier est utilisé pour soumettre les résultats du modèle et obtenir les scores automatiquement.

5️⃣ Interface utilisateur (Streamlit)

app_UI_streamlit.py :

Ce fichier contient l’ensemble du code nécessaire à la création de l’interface graphique Streamlit, permettant :

de charger des images,

d’effectuer la détection des avions,

puis de classifier chaque avion détecté.

run_UI.py :

Afin de simplifier le lancement de l’interface utilisateur, ce script permet :

d’exécuter automatiquement les commandes nécessaires,

d’ouvrir directement l’interface Streamlit sans manipulation manuelle.
