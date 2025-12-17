import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib

def data_prepare(df):
    # -------- Feature Engineering --------
    df["amenities_score"] = (
        df["parking"] +
        df["airconditioning"] +
        df["guestroom"]
    )

   
    df["area_location_interaction"] = df["area"] * df["location_score"]

    df["amenities_environment_interaction"] = (
        df["amenities_score"] * df["environment_score"]
    )

    # -------- Features & Target --------
    X = df.drop("price", axis=1)
    y = df["price"]

    # -------- Train-Test Split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Scaling (Only for Linear Regression) --------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{name} Performance")
    print("-" * 30)
    print("R2 Score :", r2)
    print("MAE      :", mae)
    print("RMSE     :", rmse)

    return r2

def main():
    df = pd.read_csv("D:\\Downloads\\Housing_Improved.csv")

    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = data_prepare(df)

    # -------- Linear Regression --------
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_r2 = evaluate_model(
        "Linear Regression",
        lr,
        X_test_scaled,
        y_test
    )

    # -------- Random Forest --------
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_r2 = evaluate_model(
        "Random Forest",
        rf,
        X_test,
        y_test
    )

    # -------- Feature Importance --------
    feature_importance = pd.Series(
        rf.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nRandom Forest Feature Importance")
    print(feature_importance)

    # -------- Pick Best Model --------
    if rf_r2 > lr_r2:
        joblib.dump(
            {
                "model_type": "random_forest",
                "model": rf,
                "scaler": None,
                "features": X_train.columns.tolist()
            },
            "best_model.pkl"
        )
        print("\n✅ Random Forest selected and saved as best_model.pkl")
    else:
        joblib.dump(
            {
                "model_type": "linear_regression",
                "model": lr,
                "scaler": scaler,
                "features": X_train.columns.tolist()
            },
            "best_model.pkl"
        )
        print("\n✅ Linear Regression selected and saved as best_model.pkl")


if __name__ == "__main__":
    main()
