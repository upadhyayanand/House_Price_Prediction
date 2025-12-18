**🏠 Housing Price Prediction – End-to-End ML Pipeline**

This project demonstrates a real-world machine learning pipeline for predicting housing prices using Linear Regression and Random Forest, including feature engineering, model evaluation, best-model selection, and production-ready inference.

The pipeline automatically selects the best performing model and applies the correct preprocessing during prediction.

**🚀 Key Highlights**

-End-to-end ML workflow (training → evaluation → inference)

-Feature engineering with interaction features

-Comparison of Linear Regression vs Random Forest

-Automatic best model selection

-Safe model serialization with metadata

-Model-aware prediction (scaled vs unscaled inputs)

-Clean separation of training and prediction logic


**🧠 Machine Learning Approach**

-Models Used

-Linear Regression

-Random Forest Regressor

-Evaluation Metrics

-R² Score

-Mean Absolute Error (MAE)

-Root Mean Squared Error (RMSE)

-The model with the highest R² score is automatically selected and saved.

**🔧 Feature Engineering**

The following engineered features are used to capture real-world pricing behavior:

**Feature	**                                  ** Description**
-amenities_score	                          -Sum of parking, AC, and guestroom
-area_location_interaction	                -Area × Location score
-amenities_environment_interaction	       - Amenities × Environment score

**Why Interaction Features Matter**

-House prices are non-linear and conditional:

-A large house in a good location is far more valuable than a large house in a poor location.

-Random Forest models learn such patterns naturally, which is why these features often rank highly in feature importance.



**📂 Project Structure**

Housing-Price-Prediction/
│
├── train.py                # Model training & evaluation
├── predict.py              # Model-aware prediction
├── best_model.pkl          # Saved best model with metadata
├── Data/
│   ├── Housing_Improved.csv
│   └── userInput.csv
├── README.md
└── requirements.txt

**🏗️ Training Pipeline**

*Steps*

Load dataset

Perform feature engineering

Split into train & test sets

Train both models

Evaluate performance

Save the best model with metadata

**Run Training:**

python train.py

**This generates:**

best_model.pkl

**🔮 Prediction Pipeline**

The prediction system:

Detects which model was selected

Applies scaling only if required

Ensures feature order consistency

Produces reliable predictions

**Run Prediction**
python predict.py


Example:

preds = predict_price("./Data/userInput.csv")


**🧩 Model Metadata (Production-Grade)**

-Each saved model includes:

{
  "model_type": "random_forest" or "linear_regression",
  "model": trained_model,
  "scaler": scaler_or_None,
  "features": feature_columns
}


This guarantees training–inference consistency.


**📊 Why Random Forest Often Wins**

-Learns non-linear relationships

-Handles feature interactions naturally

-Robust to outliers

-No scaling required

-Linear Regression is still included as a strong baseline model.


**🛠️ Tech Stack**

-Python

-Pandas, NumPy

-Scikit-learn

-Joblib

**📌 Future Improvements**

-Cross-validation

-Hyperparameter tuning

-Sklearn Pipelines

-FastAPI inference API

-Model monitoring & logging


**🎯 Interview-Ready Summary**

“This project implements a production-ready ML pipeline that compares linear and tree-based models, selects the best performer automatically, and ensures correct preprocessing during inference using saved metadata.”
