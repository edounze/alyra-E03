# Evaluation numéro 2 Alyra (E03)

Ce projet consiste en un service d'API basé sur FastAPI pour la prédiction d'images à l'aide d'un modèle de réseau neuronal préalablement entraîné.

## Contenu du dossier

- `app/` : Contient les fichiers source de l'application FastAPI.
  - `main.py` : Le point d'entrée de l'application FastAPI.
  - `model.py` : Le script contenant les fonctions pour la prédiction à partir du modèle.
  - `preprocess.py` : Le script contenant les fonctions de prétraitement des images.
    model/ : Contient le modèle de réseau neuronal préalablement entraîné au format h5.
- `image_test_1.png` et `image_test_2.png` : Des exemples d'images à utiliser pour tester l'API.
- requirements.txt : Le fichier contenant les dépendances Python nécessaires pour exécuter l'application.

## Installation et exécution

1. Assurez-vous d'avoir Python et pip installés sur votre système.
2. Installez les dépendances en exécutant `pip install -r requirements.txt`.
3. Démarrez le serveur FastAPI en exécutant `uvicorn app.main:app --reload`.

## Test de la prédiction avec curl

Vous pouvez tester la prédiction en utilisant curl avec la commande suivante :

```bash
curl -X POST -F "file=@image_test_1.png" http://127.0.0.1:8000/predict
```

Assurez-vous de remplacer image_test_1.png par le chemin de votre image de test (2 images sont proposées à la racine de ce projet).
