print("KNN")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def load_dataset():
    print("Step 2: Loading the Dataset.....\n")
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns = iris.feature_names)
    print(df)
    return df,iris
 
def data_prep(df,iris):
    print("Step 3:Data Preparation......\n")
    X = df.iloc[:,:].values
    y = iris.target
    print(X)
    print()
    print(y)
    return X,y
    
def split_data(X,y):
    print("\nStep 4: Splitting the dataset......\n")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state =58)
    return X_train,X_test,y_train,y_test
    
def get_k(X_train,X_test,y_train,y_test):
    print("Creating the Model........\n")
    model = KNeighborsClassifier(n_neighbors = 5)
    error = []
    k_range = range(1,15)
    for i in k_range:
        
        kn = KNeighborsClassifier(n_neighbors = i)
        kn.fit(X_train,y_train)
        
        y_pred = kn.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        err = 1-acc
        error.append(err)
        
        
    # plt.figure(figsize=(10,6))
    # plt.plot(k_range,error,marker = 'o')
    # plt.xlabel('No of Neighbors')
    # plt.ylabel('error')
    # plt.show()
    
    best_k = k_range[error.index(min(error))]
    print("Best n_neihbours is :",best_k)
    print("Minimum error",min(error)*100)
    return best_k
    
    
def model_train(best_k,X_train,y_train):
    print("Training the Model.......")
    knn = KNeighborsClassifier(n_neighbors = best_k)
    knn.fit(X_train,y_train)
    return knn
   
    
def model_evaluation(knn,X_test,y_test):
    print("Predicting using model......... ")
    y_pred = knn.predict(X_test)
    print(y_pred)
    score = accuracy_score(y_test,y_pred)
    print(score)
    
    
def main():
    print("Step 1:Importing the Libraries......\n")
    df,iris=load_dataset()
    X,y = data_prep(df,iris)
    X_train,X_test,y_train,y_test=split_data(X,y)
    best_k = get_k(X_train,X_test,y_train,y_test)
    knn = model_train(best_k,X_train,y_train)
    model_evaluation(knn,X_test,y_test)
    
if __name__=="__main__":
    main()