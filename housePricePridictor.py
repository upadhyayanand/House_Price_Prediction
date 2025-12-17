import pandas as pd
import numpy as np
import joblib


def feature_engineering(df):
    df = df.copy()
    
    df["amenities_score"] = (
        df["parking"] +
        df["airconditioning"] +
        df["guestroom"]
    )

    df["area_location_interaction"] = df["area"] * df["location_score"]

    df["amenities_environment_interaction"] = (
        df["amenities_score"] * df["environment_score"]
    )

    return df
def predict_price(input_csv):
    # Load input data
    df = pd.read_csv(input_csv)

    # Load saved artifact
    artifact = joblib.load("best_model.pkl")

    model = artifact["model"]
    model_type = artifact["model_type"]
    scaler = artifact["scaler"]
    feature_cols = artifact["features"]

    # Feature engineering
    df = feature_engineering(df)

    # Select correct columns & order
    X = df[feature_cols]

    # Scaling only if needed
    if model_type == "linear_regression":
        X = scaler.transform(X)

    # Prediction
    predictions = model.predict(X)

    return predictions


if __name__ == "__main__":
    preds = predict_price(".\\Data\\userInput.csv")
    print("Predicted Prices:")
    print(preds)
