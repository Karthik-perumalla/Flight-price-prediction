import pandas as pd
import streamlit as st
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV


st.title("Flight Price Prediction using Machine Learning")



def load_file(filepath):
    df = pd.read_csv(filepath)
    st.dataframe(df)
    return df



def preprocess_data(df):
    df = df.drop(columns="flight")
    X = df.drop(columns="price")
    y = df["price"]
    return X, y



def model_pipeline(X, model):

    cat_cols = X.select_dtypes(include="object").columns.to_list()
    num_cols = X.select_dtypes(include="number").columns.to_list()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def select_model(X_train, y_train, X_test, y_test):

    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(),
    }

    results = []
    best_model = None
    best_score = -float("inf")

    for name, model in models.items():

        st.write(f"Running {name}...")

        pipeline = model_pipeline(X_train, model)

      
        if name == "Ridge":
            param_dist = {
                'model__alpha': [0.1, 1, 10]
            }

        elif name == "Lasso":
            param_dist = {
                'model__alpha': [0.01, 0.1],
                'model__max_iter': [1000]
            }


        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=2,   
            cv=2,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        score = r2_score(y_test, y_pred)

       
        if score > best_score:
            best_score = score
            best_model = search.best_estimator_

        results.append({
            "Model": name,
            "R2 Score": score,
            "Best Params": search.best_params_
        })

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

    return results_df, best_model



def train_best_model(best_model, X_train, X_test, y_train, y_test):

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)  

    st.subheader("Best Model Performance")
    st.write("R2 Score:", r2_score(y_test, y_pred))

    
    joblib.dump(best_model, "best_model.joblib")
    st.success("Model saved as best_model.joblib")



def main():

    df = load_file("processed.csv")

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results, best_model = select_model(X_train, y_train, X_test, y_test)

    st.subheader("Model Comparison")
    st.dataframe(results)

    train_best_model(best_model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()