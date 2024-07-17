import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Function to preprocess a new image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make predictions
def predict_image(image_path, model):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class, prediction

# Load the trained model
model = load_model('/content/brain_tumor_detection_model.h5')

# Path to the new image
new_image_path = '/content/Y255.JPG'

# Make prediction
predicted_class, prediction = predict_image(new_image_path, model)

# Output the result
if predicted_class == 0:
    print("The model predicts: No brain tumor")
else:
    print("The model predicts: Brain tumor")

# Output the prediction probabilities
print("Prediction probabilities:", prediction)
