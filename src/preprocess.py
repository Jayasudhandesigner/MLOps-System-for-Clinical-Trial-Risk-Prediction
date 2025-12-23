import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

from ingest import load_data

NUMERIC_FEATURES = [
    "age",
    "visits_completed",
    "adverse_events",
    "days_in_trial",
]

CATEGORICAL_FEATURES = [
    "gender",
    "treatment_group",
    "trial_phase",
]

TARGET = "dropout"


def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def preprocess_data(input_path: str, output_path: str):
    df = load_data(input_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    joblib.dump(preprocessor, "data/processed/preprocessor.pkl")

    processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )
    processed_df[TARGET] = y.values

    processed_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/clinical_trials.csv",
        output_path="data/processed/clinical_trials_processed.csv"
    )
    print("Preprocessing complete")