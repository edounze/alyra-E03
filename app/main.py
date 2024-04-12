# FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Model
from app.model import predict, load_model
from app.preprocess import preprocess_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("model/model.h5")

"""
A POST endpoint that takes an image file, preprocesses it, makes a prediction using a deep learning model, and returns the prediction. 

Parameters:
file (UploadFile): The image file to be processed.

Returns:
dict: A dictionary containing the prediction result.
"""
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image = await file.read()
    preprocessed_image = preprocess_image(image)
    prediction = predict(model, preprocessed_image)
    return {"prediction": prediction}