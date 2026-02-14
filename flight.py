# Flight Price Prediction using Random forest Classifier

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def load_file(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    return df
    
def preprocess_data(df):
    print("Preprocessing the data.......")
    categorical_features = [
        'airline',
        'source_city',
        'departure_time',
        'stops',
        'arrival_time',
        'destination_city',
        'class'
    ]
    encoded_df = pd.get_dummies(
        df[categorical_features],
        drop_first=True,
        dtype=int
    )
    numerical_features = df[['duration', 'days_left']]
    
    X = pd.concat([encoded_df,numerical_features],axis =1)
    y = df['price']
    return X,y
    
def hyper_tuning(X_train,y_train):
    print("Running hyperparameter tuning....")
    
    
    model = RandomForestRegressor(random_state = 42)  
    param_dist = {
    "n_estimators" : [50,100,200],
    "max_depth":[5,10,15,20],
    "max_features":['sqrt','log2',0.5]
    }
    search = RandomizedSearchCV(model,
    param_distributions = param_dist,
    n_iter = 10,
    cv = 3,
    scoring = 'r2',
    random_state = 42,
    n_jobs = -1
    )
    
    search.fit(X_train,y_train)
    
    print("Best params:",search.best_params_)
    print("Best cv score:",search.best_score_)
    
    return search.best_estimator_
    
def evaluate_model(model,X_test,y_test):
    print("Evaluating the model.......")
    y_pred = model.predict(X_test)
    score = r2_score(y_test,y_pred)
    print("test r2_score:",score)
    
def save_model(model):
    joblib.dump(model,"flight.pkl")
    print("Saved the model ....")

    
def main():
    
    df = load_file('data_1.csv')
    X,y = preprocess_data(df)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state = 58)
    model = hyper_tuning(X_train,y_train)
    evaluate_model(model,X_test,y_test)
    save_model(model)
    
main()
