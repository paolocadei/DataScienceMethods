import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


def tune_ridge_regression(X_train, y_train, X_test, param_grid=None, cv=15):

    warnings.filterwarnings("ignore")
    # Define a default parameter grid if none is provided
    if param_grid is None:
        param_grid = {
            'alpha': [0, 0.1, 10.0, 100.0, 1000.0],
        }

    # Create a Ridge regression model
    ridge = Ridge()

    # Set up GridSearchCV with the Ridge model and parameter grid
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    return best_model, best_params, grid_search, y_pred


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_clusters_features_vs_target(X_clusters, y_clusters, n_cols=3, figsize=(15, 10)):
    """
    Plots each feature in the clusters against the target variable, using different colors for each cluster.

    Parameters:
    X_clusters (list of array-like): List of feature matrices, one for each cluster.
    y_clusters (list of array-like): List of target vectors, one for each cluster.
    n_cols (int): Number of columns for the subplots.
    figsize (tuple): Size of the entire figure.
    """
    # Assuming all clusters have the same number of features
    n_features = X_clusters[0].shape[1]
    n_rows = (n_features + n_cols - 1) // n_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Feature vs. Target (Colored by Cluster)", fontsize=16)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot each feature against the target variable for each cluster
    colors = ['blue', 'orange']  # Colors for the two clusters
    for i in range(n_features):
        ax = axes[i]
        for j, (X, y) in enumerate(zip(X_clusters, y_clusters)):
            ax.scatter(X[:, i], y, color=colors[j], alpha=0.6, label=f'Cluster {j+1}')
        ax.set_title(f"Feature {i+1} vs. Target (y)")
        ax.set_xlabel(f"Feature {i+1}")
        ax.set_ylabel("Target (y)")

    # Hide any unused subplots if there are any
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    # Add legend to the plot
    fig.legend(labels=[f'Cluster {i+1}' for i in range(len(X_clusters))], loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



from sklearn.cluster import KMeans

from sklearn.cluster import KMeans
import numpy as np

def divide_data_by_kmeans_distance(X_train, y_train, n_clusters=2, random_state = 40):
    """
    Divides both training and testing datasets into clusters using KMeans on the training data.
    Assigns test data to clusters based on the minimum distance to the cluster centroids of the training data.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data target.
    X_test (array-like): Testing data features.
    y_test (array-like): Testing data target.
    n_clusters (int): Number of clusters for KMeans.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Divided training and testing datasets for each cluster.
    """
    # Perform KMeans clustering on the training data
    kmeans = KMeans(n_clusters=n_clusters, random_state = random_state)
    kmeans.fit(X_train)
    cluster_labels_train = kmeans.labels_

    # Divide the training data based on the KMeans cluster labels
    X_train_clusters = [X_train[cluster_labels_train == i] for i in range(n_clusters)]
    y_train_clusters = [y_train[cluster_labels_train == i] for i in range(n_clusters)]

    return (X_train_clusters, y_train_clusters), kmeans.cluster_centers_, [0, 1]

import numpy as np

def assign_test_data_to_clusters_with_labels(X_test, cluster_means, cluster_labels):
    """
    Assigns test data to clusters based on the minimum distance to the cluster means.
    The assigned clusters correspond to the labels of the closest training cluster.

    Parameters:
    X_test (array-like): Test data features.
    cluster_means (array-like): Means of the clusters obtained from training data.
    cluster_labels (array-like): Labels of the clusters.

    Returns:
    tuple: Test data divided into clusters (X1_test, X2_test) along with the original indices of the data points
    for each cluster.
    """
    # Calculate the distance of each point in X_test to each cluster mean
    dist_to_means = np.linalg.norm(X_test[:, np.newaxis] - cluster_means, axis=2)

    # Determine the closest cluster mean for each test point
    closest_mean_indices = np.argmin(dist_to_means, axis=1)

    # Get the original indices for each cluster
    idx1 = np.where(closest_mean_indices == cluster_labels[0])[0]
    idx2 = np.where(closest_mean_indices == cluster_labels[1])[0]

    # Assign the test data to clusters based on the corresponding cluster labels
    X1_test = X_test[idx1]
    X2_test = X_test[idx2]
    
    return X1_test, X2_test, idx1, idx2


def assign_val_data_to_clusters_with_labels(X_test, y_test, cluster_means, cluster_labels):
    """
    Assigns test data and target values to clusters based on the minimum distance to the cluster means.
    The assigned clusters correspond to the labels of the closest training cluster.

    Parameters:
    X_test (array-like): Test data features.
    y_test (array-like): Test data target values.
    cluster_means (array-like): Means of the clusters obtained from training data.
    cluster_labels (array-like): Labels of the clusters.

    Returns:
    tuple: Test data and target values divided into clusters (X1_test, y1_test, X2_test, y2_test) based on 
    the corresponding cluster labels.
    """
    # Calculate the distance of each point in X_test to each cluster mean
    dist_to_means = np.linalg.norm(X_test[:, np.newaxis] - cluster_means, axis=2)
    # Determine the closest cluster mean for each test point
    closest_mean_indices = np.argmin(dist_to_means, axis=1)
    
    # Assign the test data to clusters based on the corresponding cluster labels
    X1_test = X_test[closest_mean_indices == cluster_labels[0]]
    y1_test = y_test[closest_mean_indices == cluster_labels[0]]
    X2_test = X_test[closest_mean_indices == cluster_labels[1]]
    y2_test = y_test[closest_mean_indices == cluster_labels[1]]
    
    return X1_test, y1_test, X2_test, y2_test


def remove_outliers(X, y, threshold=3):
    # Combine X and y into a single DataFrame for easier processing
    data = pd.DataFrame(X)
    data['target'] = y

    # Calculate Q1 and Q3 for each feature
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Filter data within the acceptable range
    data_filtered = data[~((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)]

    # Separate the filtered data back into features and target
    X_filtered = data_filtered.drop(columns=['target']).values
    y_filtered = data_filtered['target'].values
    return X_filtered, y_filtered

def remove_outliers_y(X, y, threshold=1.5):
    # Convert y into a DataFrame for easier processing
    y_df = pd.DataFrame(y, columns=['target'])

    # Calculate Q1 and Q3 for y
    Q1 = y_df['target'].quantile(0.25)
    Q3 = y_df['target'].quantile(0.75)
    IQR = Q3 - Q1

    # Identify the outliers in y based on the IQR and threshold
    y_filtered = y_df[~((y_df['target'] < (Q1 - threshold * IQR)) | (y_df['target'] > (Q3 + threshold * IQR)))]

    # Get the indices of the filtered (non-outlier) y values
    filtered_indices = y_filtered.index

    # Filter X based on the indices of the non-outlier y values
    X_filtered = X[filtered_indices]

    # Get the filtered y values
    y_filtered = y[filtered_indices]

    print(len(y_df)- len(y_filtered))

    return X_filtered, y_filtered


def standardize_data(X_train, X_test):

    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both the training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def create_polynomial_features(X_train, X_test, degree=2, include_bias=False):

    # Create a PolynomialFeatures object
    poly_transformer = PolynomialFeatures(degree=degree, include_bias=include_bias)

    # Fit the transformer on the training data and transform both the training and testing data
    X_train_poly = poly_transformer.fit_transform(X_train)
    X_test_poly = poly_transformer.transform(X_test)

    return X_train_poly, X_test_poly

def apply_pca(X_train, X_test, n_components=0.95):

    pca_transformer = PCA(n_components=n_components)

    # Fit the PCA transformer on the training data and transform both the training and testing data
    X_train_pca = pca_transformer.fit_transform(X_train)
    X_test_pca = pca_transformer.transform(X_test)

    return X_train_pca, X_test_pca

def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))



from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import warnings

def tune_kernel_ridge_regression(X_train, y_train, X_test, param_grid=None, cv=5):
    """
    Tunes the Kernel Ridge regression model using GridSearchCV with a given parameter grid.
    Uses negative mean squared error as the scoring metric.

    Parameters:
    X_train (array-like): The training data features.
    y_train (array-like): The training target values.
    X_test (array-like): The testing data features.
    param_grid (dict): The parameter grid to search over. If None, a default grid will be used.
    cv (int): Number of cross-validation folds.

    Returns:
    best_model (KernelRidge): The best Kernel Ridge regression model found by GridSearchCV.
    best_params (dict): The best hyperparameters found by GridSearchCV.
    grid_search (GridSearchCV): The fitted GridSearchCV object.
    y_pred (array-like): Predictions made by the best model on the test set.
    """

    print(X_train.shape, X_test.shape)

    # Define a default parameter grid if none is provided
    if param_grid is None:
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'kernel': ['linear','rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4]
        }

    # Create a Kernel Ridge regression model
    kernel_ridge = KernelRidge()

    # Set up GridSearchCV with the Kernel Ridge model and parameter grid
    grid_search = GridSearchCV(
        estimator=kernel_ridge,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    return best_model, best_params, grid_search, y_pred

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import GridSearchCV

def tune_gaussian_process_regression(X_train, y_train, X_test, param_grid=None, cv=5):
    """
    Tunes the Gaussian Process regression model using GridSearchCV with a given parameter grid.
    Uses negative mean squared error as the scoring metric.

    Parameters:
    X_train (array-like): The training data features.
    y_train (array-like): The training target values.
    X_test (array-like): The testing data features.
    param_grid (dict): The parameter grid to search over. If None, a default grid will be used.
    cv (int): Number of cross-validation folds.

    Returns:
    best_model (GaussianProcessRegressor): The best Gaussian Process regression model found by GridSearchCV.
    best_params (dict): The best hyperparameters found by GridSearchCV.
    grid_search (GridSearchCV): The fitted GridSearchCV object.
    y_pred (array-like): Predictions made by the best model on the test set.
    """
    
    print(X_train.shape, X_test.shape)

    # Define a default parameter grid if none is provided
    if param_grid is None:
        param_grid = {
            'alpha': [1e-2, 1e-3, 1e-4],
            'kernel': [
                RBF(length_scale=10.0), 
                Matern(length_scale=10.0, nu=1.5), 
                RationalQuadratic(length_scale=10.0, alpha=0.1), 
                #ExpSineSquared(length_scale=1.0, periodicity=3.0)
            ]
        }

    # Create a Gaussian Process regression model
    gaussian_process = GaussianProcessRegressor()

    # Set up GridSearchCV with the Gaussian Process model and parameter grid
    grid_search = GridSearchCV(
        estimator=gaussian_process,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        #verbose=1,
    )

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    return best_model, best_params, grid_search, y_pred


def tune_knn_regression(X_train, y_train, X_test, param_grid=None, cv=5):
    """
    Tunes the K-Nearest Neighbors regression model using GridSearchCV with a given parameter grid.
    Uses negative mean squared error as the scoring metric.

    Parameters:
    X_train (array-like): The training data features.
    y_train (array-like): The training target values.
    X_test (array-like): The testing data features.
    param_grid (dict): The parameter grid to search over. If None, a default grid will be used.
    cv (int): Number of cross-validation folds.

    Returns:
    best_model (KNeighborsRegressor): The best KNN regression model found by GridSearchCV.
    best_params (dict): The best hyperparameters found by GridSearchCV.
    grid_search (GridSearchCV): The fitted GridSearchCV object.
    y_pred (array-like): Predictions made by the best model on the test set.
    """

    print(X_train.shape, X_test.shape)

    # Define a default parameter grid if none is provided
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 10],  # Number of neighbors to use
            'weights': ['uniform', 'distance'],  # Weight function used in prediction
            'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
        }

    # Create a KNeighborsRegressor model
    knn = KNeighborsRegressor()

    # Set up GridSearchCV with the KNN model and parameter grid
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        # verbose=1,  # Uncomment this line to get detailed output
    )

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    return best_model, best_params, grid_search, y_pred




