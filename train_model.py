import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump
from tqdm import tqdm
import random

# Crow Search Algorithm Implementation
class CrowSearchAlgorithm:
    def __init__(self, n_crows, max_iter, dim, lower_bound, upper_bound, fitness_function):
        self.n_crows = n_crows
        self.max_iter = max_iter
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_function = fitness_function

    def optimize(self):
        # Initialize crows' positions and memory
        crows = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_crows, self.dim))
        memory = np.copy(crows)
        best_solution = None
        best_fitness = -np.inf

        # Optimization loop
        for iter in range(self.max_iter):
            for i in range(self.n_crows):
                # Generate random position for crow i
                r = random.random()
                random_crow = random.randint(0, self.n_crows - 1)
                if r < 0.8:  # Awareness probability
                    crows[i] = memory[random_crow] + np.random.uniform(-1, 1, self.dim)
                else:
                    crows[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

                # Bound positions
                crows[i] = np.clip(crows[i], self.lower_bound, self.upper_bound)

                # Evaluate fitness
                fitness = self.fitness_function(crows[i])
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = crows[i]
                    memory[i] = crows[i]

            print(f"Iteration {iter + 1}/{self.max_iter}, Best Fitness: {best_fitness:.4f}")

        return best_solution, best_fitness


# Load and preprocess images
def load_data(dataset_path, target_size=(128, 128)):
    X, y = [], []
    class_labels = os.listdir(dataset_path)
    for label in class_labels:
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            print(f"Processing images for class: {label}")
            for img_name in tqdm(os.listdir(class_path)):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    X.append(img.flatten())
                    y.append(label)
    return np.array(X), np.array(y)


# Fitness function for CSA
def fitness_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    return accuracy_score(y_val, y_pred)


# Main script
if __name__ == "__main__":
    # Dataset path
    dataset_path = "Train"  # Update this with your dataset path
    print("Loading and preprocessing images...")
    X, y = load_data(dataset_path)
    print(f"Loaded {len(X)} images.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define bounds for CSA
    lower_bound = [50, 5]  # Min values for n_estimators and max_depth
    upper_bound = [200, 20]  # Max values for n_estimators and max_depth

    # Initialize CSA
    csa = CrowSearchAlgorithm(
        n_crows=5,
        max_iter=20,
        dim=2,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        fitness_function=fitness_function
    )

    print("Starting Crow Search Optimization...")
    best_params, best_fitness = csa.optimize()
    print(f"Best parameters: n_estimators={int(best_params[0])}, max_depth={int(best_params[1])}")
    print(f"Best validation accuracy: {best_fitness:.4f}")

    # Train final model
    print("Training the final model...")
    final_model = RandomForestClassifier(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        random_state=42
    )
    final_model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_test_pred = final_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the model and scaler
    print("Saving the model and scaler...")
    dump(final_model, "final_model.joblib")
    dump(scaler, "scaler.joblib")
    print("Model and scaler saved.")
