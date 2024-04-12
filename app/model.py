import tensorflow as tf
import numpy as np

"""
Load a model from the specified path.

:param model_path: str, optional, the path to the model file.
:return: loaded Keras model
"""
def load_model(model_path="model/model.h5"):
    model = tf.keras.models.load_model(model_path)
    return model

"""
Make predictions using a given model on an image.

Parameters:
- model: the deep learning model to use for prediction
- image: the image to make predictions on

Returns:
- response_json: a JSON object containing the predicted class, probability, and a list of predictions
"""
def predict(model, image):

    # Prédictions sur l'image d'entrée en utilisant le modèle fourni
    predictions = model.predict(image)

    # Application de softmax pour obtenir des probabilités
    score = tf.nn.softmax(predictions[0])

    # Classe prédite en prenant l'indice de la probabilité maximale
    predicted_class = np.argmax(score)

    # Confiance de la prédiction en pourcentage
    predicted_confidence = 100 * np.max(score)

    # Convertir les prédictions en liste
    predictions_list = predictions.tolist()

    # On retourne une réponse en JSON
    response_json = {
        "class": str(predicted_class),
        "probability": str(predicted_confidence),
        "predictions": predictions_list
    }

    return response_json