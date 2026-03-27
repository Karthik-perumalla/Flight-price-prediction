# Flight Price Prediction using Regularized Linear Models

import pandas as pd
import numpy as np
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score

st.title("Flight Price Prediction using Linear Models")

# ================= LOAD DATA =================
def load_file(filepath):
    st.info("Loading dataset...")
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.dataframe(df)
    return df

# ================= PREPROCESS =================
def preprocess_data(df):
    categorical_features = [
        'airline','source_city','departure_time',
        'stops','arrival_time','destination_city','class'
    ]

    encoded_df = pd.get_dummies(df[categorical_features], drop_first=True)
    numerical_features = df[['duration','days_left']]

    X = pd.concat([encoded_df, numerical_features], axis=1)
    y = df['price']

    return X, y

# ================= HYPERPARAMETER TUNING =================
def hyper_tuning(X_train, y_train):
    st.write("Running hyperparameter tuning...")

    # Choose model type
    model = ElasticNet()

    param_dist = {
        "alpha": [0.01, 0.1, 1, 10],
        "l1_ratio": [0.1, 0.5, 0.7, 0.9]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    st.success("Model trained successfully!")
    st.write("Best Params:", search.best_params_)
    st.write("Best Score:", search.best_score_)

    return search.best_estimator_

# ================= EVALUATE =================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    st.write("Test R2 Score:", score)

# ================= SAVE =================
def save_model(model):
    joblib.dump(model, "model.pkl", compress=3)
    st.success("Model saved!")

# ================= MAIN =================
def main():
    df = load_file("df_test.csv")

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=58
    )

    model = hyper_tuning(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

if __name__ == "__main__":
    main()