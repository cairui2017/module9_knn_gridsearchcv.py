import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def main():
    # Collect training data
    N = int(input("Enter the number of training data points (N): "))
    train_data = np.zeros((N, 2))
    for i in range(N):
        x, y = input(f"Enter x and y for training data point {i+1} (separated by space): ").split()
        train_data[i] = [float(x), int(y)]
        
    X_train = train_data[:, 0].reshape(-1, 1)
    y_train = train_data[:, 1]
    
    # Collect test data
    M = int(input("Enter the number of test data points (M): "))
    test_data = np.zeros((M, 2))
    for i in range(M):
        x, y = input(f"Enter x and y for test data point {i+1} (separated by space): ").split()
        test_data[i] = [float(x), int(y)]
        
    X_test = test_data[:, 0].reshape(-1, 1)
    y_test = test_data[:, 1]
    
    # Setup the kNN classifier and GridSearchCV
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 11))}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    
    # Fit the model and find the best k
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    
    # Predict and calculate test accuracy
    predictions = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    
    print(f"The best k for the kNN Classification method is: {best_k}")
    print(f"The corresponding test accuracy is: {test_accuracy}")

if __name__ == "__main__":
    main()

