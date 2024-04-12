# Python packages
import io

# Librairies
import numpy as np
from PIL import Image

"""
Preprocesses the input image by resizing it, converting it to a numpy array, and normalizing the pixel values.

Parameters:
image (bytes): The input image in bytes format.
size (tuple): The desired size to resize the image to. Defaults to (28, 28).

Returns:
numpy.ndarray: The preprocessed image array ready to be used as input for a model.
"""
def preprocess_image(image, size=(28, 28)):
    # Ouverture de l'image à partir des données bytes et la redimensionner à la taille spécifiée
    img = Image.open(io.BytesIO(image))
    img = img.resize(size)

    # Convertion de l'image en un tableau numpy et normaliser les valeurs des pixels en les divisant par 255
    img_array = np.array(img) / 255.0

    # On aplati l'image pour correspondre à la forme d'entrée attendue par le modèle
    img_array = img_array.reshape((1, 784))  # Flatten the image to match model input shape
    # img_array = np.expand_dims(img_array, axis=0)

    # On renvoie l'image prétraité
    return img_array