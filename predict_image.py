import os
import cv2
import numpy as np
from joblib import load

def load_model_and_scaler(model_path="final_model.pkl", scaler_path="scaler.pkl"):
    """
    Load the trained model and scaler from disk.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler file not found. Ensure you have trained and saved the model.")

    model = load(model_path)
    scaler = load(scaler_path)
    return model, scaler


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess an input MRI image:
    - Load in grayscale
    - Resize to target size
    - Flatten into a 1D feature vector
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Error: The file {image_path} does not exist.")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Unable to load image from {image_path}. Please check the file format.")

    img = cv2.resize(img, target_size)
    img = img.flatten()
    return img


def predict_dementia_risk(image_path, model, scaler):
    """
    Predict the dementia risk for an input MRI image using the trained model.
    """
    # Preprocess the image
    img = preprocess_image(image_path)

    # Scale the image features
    img_scaled = scaler.transform([img])

    # Predict probabilities for each class
    probabilities = model.predict_proba(img_scaled)[0]
    class_labels = model.classes_

    # Get the most likely class and its probability
    most_likely_class = class_labels[probabilities.argmax()]
    likelihood = probabilities.max() * 100  # Convert to percentage

    return most_likely_class, likelihood, dict(zip(class_labels, probabilities * 100))


if __name__ == "__main__":
    import argparse

    # Argument parser for CLI usage
    parser = argparse.ArgumentParser(description="Predict Alzheimer's risk from an MRI image.")
    parser.add_argument("image_path", type=str, help="Path to the MRI image file.")
    parser.add_argument("--model_path", type=str, default="final_model.joblib", help="Path to the saved model file.")
    parser.add_argument("--scaler_path", type=str, default="scaler.joblib", help="Path to the saved scaler file.")
    args = parser.parse_args()

    try:
        # Load the model and scaler
        model, scaler = load_model_and_scaler(args.model_path, args.scaler_path)

        # Predict the dementia risk
        most_likely_class, likelihood, class_probabilities = predict_dementia_risk(args.image_path, model, scaler)

        # Print results
        print(f"Prediction Results for Image: {args.image_path}")
        print(f"Most Likely Class: {most_likely_class}")
        print(f"Likelihood: {likelihood:.2f}%")
        print("\nClass Probabilities:")
        for class_name, prob in class_probabilities.items():
            print(f"  {class_name}: {prob:.2f}%")

    except Exception as e:
        print(str(e))