# ============================================================================
# TRAINING SCRIPT FOR CLINICAL TRIAL DROPOUT PREDICTION
# ============================================================================
# This script trains machine learning models to predict patient dropout
# from clinical trials. It implements a complete ML pipeline including:
# - Data loading and splitting
# - Model training (Logistic Regression and Random Forest)
# - Model evaluation using ROC AUC and Recall metrics
# - Model persistence (saving trained models to disk)
# ============================================================================

# ============================================================================
# IMPORTS: External Libraries and Functions
# ============================================================================

# pandas: Data manipulation library for working with structured data (CSV files)
import pandas as pd

# joblib: Efficient serialization library for saving/loading ML models
# More efficient than pickle for large numpy arrays and scikit-learn models
import joblib

# json: For saving metrics to a JSON file for easy viewing
import json

# train_test_split: Function to split dataset into training and testing sets
# This ensures we evaluate model performance on unseen data
from sklearn.model_selection import train_test_split

# LogisticRegression: A linear model for binary classification
# Simple, fast, and interpretable - good as a baseline model
from sklearn.linear_model import LogisticRegression

# RandomForestClassifier: Ensemble of decision trees for classification
# More powerful than logistic regression, can capture non-linear patterns
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics:
# - roc_auc_score: Measures model's ability to distinguish between classes (0-1, higher is better)
# - recall_score: Measures what % of actual positives are correctly identified (important for dropout)
from sklearn.metrics import roc_auc_score, recall_score

# ============================================================================
# GLOBAL CONFIGURATION: Constants used throughout the script
# ============================================================================

# Path to the processed dataset (output from preprocessing pipeline)
DATA_PATH = "data/processed/clinical_trials_processed.csv"

# Name of the target column we're trying to predict (dropout: 0=stayed, 1=dropped out)
TARGET = "dropout"

# Random seed for reproducibility - ensures same results every time we run the script
# 42 is a popular choice (reference to "Hitchhiker's Guide to the Galaxy")
RANDOM_STATE = 42


# ============================================================================
# FUNCTION: load_dataset
# ============================================================================
# Purpose: Load the processed dataset and separate features from target variable
# 
# Parameters:
#   path (str): File path to the CSV dataset
#
# Returns:
#   X (DataFrame): Feature matrix (all columns except the target)
#   y (Series): Target vector (only the dropout column)
# ============================================================================
def load_dataset(path: str):
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(path)
    
    # Create feature matrix X by dropping the target column
    # This leaves only the input features (age, gender, trial_duration, etc.)
    X = df.drop(columns=[TARGET])
    
    # Create target vector y by selecting only the dropout column
    # This is what we're trying to predict (0 or 1)
    y = df[TARGET]
    
    # Return both as a tuple (X, y)
    return X, y


# ============================================================================
# FUNCTION: train_and_evaluate
# ============================================================================
# Purpose: Train a model and evaluate its performance on test data
#
# Parameters:
#   model: Machine learning model object (must have fit, predict, predict_proba methods)
#   X_train (DataFrame): Training features
#   X_test (DataFrame): Testing features
#   y_train (Series): Training labels
#   y_test (Series): Testing labels
#
# Returns:
#   metrics (dict): Dictionary containing evaluation metrics (roc_auc, recall)
# ============================================================================
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # STEP 1: Train the model on training data
    # The fit() method learns patterns from X_train to predict y_train
    model.fit(X_train, y_train)

    # STEP 2: Get probability predictions for the positive class (dropout=1)
    # predict_proba returns 2D array: [[prob_class_0, prob_class_1], ...]
    # We select [:, 1] to get only the probability of class 1 (dropout)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # STEP 3: Get hard predictions (0 or 1) for each test sample
    # predict() uses a threshold (usually 0.5) to convert probabilities to class labels
    y_pred = model.predict(X_test)

    # STEP 4: Calculate evaluation metrics
    metrics = {
        # ROC AUC: Measures model's ability to rank positive cases higher than negative
        # Range: 0.5 (random) to 1.0 (perfect). Good models typically > 0.7
        "roc_auc": roc_auc_score(y_test, y_proba),
        
        # Recall (Sensitivity): What % of actual dropouts did we catch?
        # Formula: True Positives / (True Positives + False Negatives)
        # Important for clinical trials to identify at-risk patients early
        "recall": recall_score(y_test, y_pred),
    }
    
    # Return the metrics dictionary
    return metrics


# ============================================================================
# FUNCTION: main
# ============================================================================
# Purpose: Main training pipeline that orchestrates the entire workflow
# ============================================================================
def main():
    # ========================================================================
    # STEP 1: Load the dataset
    # ========================================================================
    print("Loading dataset...")
    X, y = load_dataset(DATA_PATH)

    # ========================================================================
    # STEP 2: Split data into training (80%) and testing (20%) sets
    # ========================================================================
    # Why split? To evaluate model performance on unseen data
    # - Training set: Used to train the model (learn patterns)
    # - Testing set: Used to evaluate model (simulate real-world performance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,                      # Input: features and target
        test_size=0.2,             # 20% for testing, 80% for training
        stratify=y,                # Keep same class distribution in train/test
        random_state=RANDOM_STATE  # Set seed for reproducibility
    )

    # Dictionary to store results from all models
    results = {}

    # ========================================================================
    # STEP 3: Train and evaluate LOGISTIC REGRESSION (Simple Baseline)
    # ========================================================================
    print("\nTraining Logistic Regression...")
    
    # Create a Logistic Regression model
    # max_iter=1000: Maximum iterations for optimization algorithm to converge
    logreg = LogisticRegression(max_iter=1000)
    
    # Train the model and get evaluation metrics
    results["logreg"] = train_and_evaluate(
        logreg, X_train, X_test, y_train, y_test
    )
    
    # Save the trained model to disk for later use (deployment, inference)
    # .pkl = pickle file format (serialized Python object)
    joblib.dump(logreg, "models/logreg.pkl")

    # ========================================================================
    # STEP 4: Train and evaluate RANDOM FOREST (Stronger Baseline)
    # ========================================================================
    print("Training Random Forest...")
    
    # Create a Random Forest model
    # Random Forest = Ensemble of decision trees (combines predictions from many trees)
    rf = RandomForestClassifier(
        n_estimators=200,          # Number of decision trees (more = better but slower)
        random_state=RANDOM_STATE  # Set seed for reproducibility
    )
    
    # Train the model and get evaluation metrics
    results["random_forest"] = train_and_evaluate(
        rf, X_train, X_test, y_train, y_test
    )
    
    # Save the trained Random Forest model to disk
    joblib.dump(rf, "models/random_forest.pkl")

    # ========================================================================
    # STEP 5: Display evaluation results for all models
    # ========================================================================
    print("\n" + "="*60)
    print("Model Evaluation Results")
    print("="*60)
    
    # Loop through results dictionary and print metrics for each model
    for model, metrics in results.items():
        print(f"{model}: {metrics}")
    
    print("="*60)
    
    # Save metrics to JSON file for easy viewing and tracking
    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nTraining complete! Models saved to 'models/' directory.")
    print("Evaluation metrics saved to 'models/evaluation_metrics.json'")



# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
# This pattern ensures main() only runs when script is executed directly
# (not when imported as a module in another script)
if __name__ == "__main__":
    main()

