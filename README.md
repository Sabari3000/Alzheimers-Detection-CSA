
# Alzheimer's Disease Prediction Using Crow Search Algorithm (CSA)

This project predicts the likelihood of Alzheimer's disease development based on MRI scans using a machine learning model optimized with the Crow Search Algorithm (CSA). It supports multiple dementia stages (Non-Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia).

---

## Features

- **Crow Search Algorithm (CSA)**: A bio-inspired optimization technique.
- **Image Processing**: Preprocessing and feature extraction from MRI images.
- **Prediction**: The model outputs the likelihood of Alzheimer's disease development.

---

## Dataset

The dataset contains MRI images divided into the following categories:
- **Non-Demented**: 2,500 images
- **Very Mild Dementia**: 2,500 images
- **Mild Dementia**: 2,500 images
- **Moderate Dementia**: 488 images

From Kaggle - https://www.kaggle.com/datasets/ninadaithal/imagesoasis

---

## Project Structure

```
project_directory/
├── New Data/                    # Folder containing MRI images
├── Train/                       # Folder containing training images
├── CSA.ipynb                    # Jupyter file of project
├── README.md                    # Project documentation
├── final_model.joblib           # Trained model file
├── final_model.pkl              # Trained model file
├── predict_image.py             # Script to predict the risk of Alzheimer's from an MRI image
├── requirements.txt             # List of Python dependencies
├── scaler.joblib                # Scaler file for feature normalization
├── scaler.joblib                # Scaler file for feature normalization
└── train_model.py               # Script to train the model


```

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Sabari3000/Alzheimers-Detection-CSA
    cd Alzheimers-Detection-CSA
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Training the Model
To train the model on your dataset, run the following command:
```bash
python train_model.py
```
This script will preprocess the images, train the model using the CSA, and save the trained model and scaler.

### 2. Predicting Alzheimer's Risk
To predict the Alzheimer's disease risk from an MRI image, use the `predict_image.py` script:
```bash
python predict_image.py "path_to_image.jpg"
```
The script automatically loads the model and scaler and returns the predicted class and likelihood.

---

## Example Prediction

```bash
python predict_image.py "New Data/Very mild Dementia/OAS1_0380_MR1_mpr-4_149.jpg"
```

**Output:**
```
Loading model from final_model.joblib using joblib...
Loading scaler from scaler.joblib...
The most likely class is 'Very Mild Dementia' with a probability of 85.23%.
```

---

## Dependencies

- Python 3.8+
- Libraries (see `requirements.txt`):
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `opencv-python`
    - `joblib`
    - `matplotlib`

---

## Notes

- Ensure the dataset is properly organized into folders (`Train/`) before training.
- The prediction script automatically detects and loads available model and scaler files.

---

## Acknowledgments

This project leverages the Crow Search Algorithm (CSA) to optimize machine learning models for Alzheimer's prediction. Special thanks to the creators of the dataset used for training.
