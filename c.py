print("Support Vector machine Algorithm")

print()

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score,confusion_matrix

def load_file():
    print("Step 2: Loading the Dataset....\n")
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns = iris.feature_names)
    df['target'] = iris.target
    print(df)
    return df
 
def data_prep(df):
    print("\nStep 3: Preparing the data X and y....\n")
    X = df.drop('target',axis =  1).values
    y = df.target.values
    return X,y
    
def split_data(X,y):
    print("Step 4:Splitting the DataSet.....\n")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=58)
    return X_train,X_test,y_train,y_test
    
def model_create():
    print("Step 5:Creating the Model.....\n")
    model=SVC()
    return model
    
def train_model(model,X_train,y_train,X_test,y_test):
    print("Step 6:Training the model......\n")
    model.fit(X_train,y_train)
    score1 = model.score(X_train,y_train)
    score2 = model.score(X_test,y_test)
    print(f"the training score is:{score1*100:.2f}%")
    print()
    print(f"the testing score is:{score2*100:.2f}%")

def model_predict(model,X_test):
    print("\nstep 7:Prediction using model....\n")
    y_pred=model.predict(X_test)
    print()
    print(y_pred)
    return y_pred
    
def main():
    print("Step 1: Importing the Libraries....\n")
    df=load_file()
    X,y=data_prep(df)
    X_train,X_test,y_train,y_test=split_data(X,y)
    model=model_create()
    train_model(model,X_train,y_train,X_test,y_test)
    y_pred=model_predict(model,X_test)
    f1=f1_score(y_test,y_pred,average = 'weighted')
    cm = confusion_matrix(y_test,y_pred)
    print()
    print(f"the f1 score is:{f1*100:.2f}%")
    print(f"the confusion_matrix is:{cm}")
    
main()