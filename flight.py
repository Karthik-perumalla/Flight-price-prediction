# Flight Price Prediction using Random forest Classifier

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def load_file(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    return df
    
def preprocess_data(df):
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
    
def train_model(X_train,y_train):
    print("Training RandomForestRegressor model...")
    
    model = RandomForestRegressor(n_estimators = 400,
    random_state = 0,
    n_jobs = -1)
    
    model.fit(X_train,y_train)
    return model
    
def evaluate_model(model,X_test,y_test):
    print("Evaluating Model...")
    
    predictions = model.predict(X_test)
    score = r2_score(y_test,predictions)
    print(f"model R2 score :{score:.4f}")

def save_model(model,filename = 'flprice.pkl'):
    joblib.dump(model,filename)
    print(f"Model saved Successfully as {filename}")
    
    
    
def main():
    df = load_file('data_1.csv')
    X,y = preprocess_data(df)
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    model = train_model(X_train,y_train)
    
    evaluate_model(model,X_test,y_test)
    save_model(model)
    
if __name__ == '__main__':
    main()