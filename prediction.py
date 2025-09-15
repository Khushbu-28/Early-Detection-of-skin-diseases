import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model('ham10000_skin_cnn_densenet121.h5')

# Class labels (should match your training)
class_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Image size expected by model
IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array

def predict_skin_disease(image_path):
    try:
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]
        predicted_class = class_labels[class_index]

        print(f"Predicted Skin Disease: {predicted_class}")
        print(f"Confidence Score: {confidence:.4f}")
        return predicted_class, confidence

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(val_error)
    except Exception as e:
        print(f"Unexpected error: {e}")


# Provide your image path here, or use a file dialog to get it interactively
test_image_path = r'D:\Project_6month\melanoma.jpg'

predict_skin_disease(test_image_path)
