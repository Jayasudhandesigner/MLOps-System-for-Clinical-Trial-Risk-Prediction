import pandas as pd

REQUIRED_COLUMNS = [
    "patient_id",
    "age",
    "gender",
    "treatment_group",
    "trial_phase",
    "visits_completed",
    "adverse_events",
    "days_in_trial",
    "dropout"
]

def load_data(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["patient_id"].isnull().any():
        raise ValueError("patient_id contains null values")

    if df["dropout"].isin([0, 1]).all() is False:
        raise ValueError("dropout must be binary (0 or 1)")

    return df


if __name__ == "__main__":
    data = load_data("data/raw/clinical_trials.csv")
    print("Data loaded successfully")
    print(data.median())
