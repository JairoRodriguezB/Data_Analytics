
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def compute_within_class_variance(df, features, class_column):
    """
    Compute the Within-Class Variance for given features in a dataset.
    
    Parameters:
    - df (DataFrame): The dataset containing features and class labels.
    - features (list): List of feature column names.
    - class_column (str): Name of the class column.

    Returns:
    A dictionary containing the within-class variance for each feature.

    """
    # Compute class priors (Pj)
    class_counts = df[class_column].value_counts()
    total_samples = len(df)
    class_priors = class_counts / total_samples

    # Compute within-class variance
    within_class_variance = {}

    for feature in features:
        sw_i = 0  # Initialize sum for feature i
        for class_label, Pj in class_priors.items():
            class_data = df[df[class_column] == class_label][feature]
            sigma_ji = class_data.var()  # Variance of i-th feature in class j
            sw_i += Pj * sigma_ji  # Weighted sum
        within_class_variance[feature] = sw_i

    i = 1
    for feature, variance in within_class_variance.items():
        print(f'-> Within-Class Variance for Feature {i} ({feature}): {variance}')
        i += 1
    
    return within_class_variance


def compute_between_class_variance(df, features, class_column):
    """
    Compute the Between-Class Variance for given features in a dataset.

    The Between-Class Variance measures how much the class means deviate from 
    the overall mean of each feature, weighted by the prior probability of each class.

    Parameters:
    df (DataFrame): The dataset containing features and class labels.
    features (list): List of feature column names to compute variance for.
    class_column (str): Name of the column that contains class labels.

    Returns:
    dict: A dictionary where keys are feature names and values are the 
          between-class variance computed for each feature.
    """
    # Compute class priors (Pj)
    class_counts = df[class_column].value_counts()
    total_samples = len(df)
    class_priors = class_counts / total_samples

    # Compute overall mean for each feature
    overall_means = df[features].mean()

    # Compute between-class variance
    between_class_variance = {}

    for feature in features:
        sb_i = 0  # Initialize sum for feature i
        for class_label, Pj in class_priors.items():
            class_mean = df[df[class_column] == class_label][feature].mean()
            sb_i += Pj * (class_mean - overall_means[feature]) ** 2  # Weighted squared difference
        between_class_variance[feature] = sb_i

    for i, (feature, variance) in enumerate(between_class_variance.items(), start=1):
        print(f'-> Between-Class Variance for Feature {i} ({feature}): {variance}')

    return between_class_variance


def batch_perceptron(X, y, max_iter=100):
    """
    Batch Perceptron Algorithm for binary classification.

    Parameters:
    X (ndarray): Feature matrix (samples x features).
    y (ndarray): Labels (+1 or -1).
    max_iter (int): Maximum number of iterations.

    Returns:
    Final weight vector, number of epochs, list of misclassifications per epoch
    """
    w = np.zeros(X.shape[1])  # Initialize weights to zero
    misclassification_history = []  # Store misclassification count per epoch

    for epoch in range(1, max_iter + 1):
        misclassified = []

        # Identify misclassified samples
        for i in range(len(X)):
            if np.sign(np.dot(w, X[i])) != y[i]:
                misclassified.append(i)

        # Store the number of errors in this epoch
        misclassification_history.append(len(misclassified))

        # If no misclassifications, perceptron has converged
        if not misclassified:
            print(f"Perceptron converged after {epoch} epochs.")
            return w, epoch, misclassification_history

        # Update weights using the batch perceptron rule
        w += np.sum(y[misclassified][:, None] * X[misclassified], axis=0)

    print("Perceptron did not converge within the given iterations.")
    return w, max_iter, misclassification_history


def least_squares_classifier(X, y):
    """
    Least Squares Classifier.

    Parameters:
    X: Feature matrix.
    y: Labels (+1 or -1).

    Returns:
    w: Weight vector.
    """

    # Compute the least squares solution
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    return w


def least_squares_multiclass(X, T):
    """
    Least Squares Classifier.

    Parameters:
    X: Feature matrix.
    T: labels matrix

    Returns:
    W: Weight matrix.
    """
 
    # Compute the least squares solution
    W = np.linalg.pinv(X.T @ X) @ X.T @ T
    return W


def plot_feature_vectors_with_boundary(df, feature_x, feature_y, w, classes, model_names):
    """
    Plots one or two feature vector decision boundary graphs in the same figure

    Parameters:
    df (DataFrame): The dataset containing feature values and species_code.
    feature_x (str): The feature to plot on the X-axis.
    feature_y (str): The feature to plot on the Y-axis.
    w: A single weight vector/matrix or a list of two weight vectors/matrices.
    classes: Dictionary mapping species_code values to species names.
    model_names: Custom titles for each decision boundary.
    """
    num_plots = len(w)
   
    # Create subplots (1 row, N columns)
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))

    # If only one plot
    if num_plots == 1:
        axes = [axes]

    # Obtain the original limits of the scatter plot
    x_min, x_max = df[feature_x].min() - 0.5, df[feature_x].max() + 0.5
    y_min, y_max = df[feature_y].min() - 0.5, df[feature_y].max() + 0.5

    # Generate X values for the decision boundary
    x_vals = np.linspace(x_min, x_max, 100)

    for idx, (ax, w_i) in enumerate(zip(axes, w)):
        # Plot feature vectors with class labels
        for class_code, class_name in classes.items():
            ax.scatter(df[df['species_code'] == class_code][feature_x], 
                       df[df['species_code'] == class_code][feature_y], 
                       label=f"{class_name} ({class_code})")

        # Case 1: Binary classification (w_i is a vector)
        if w_i.ndim == 1:
            decision_boundary = -(w_i[0] / w_i[1]) * x_vals - (w_i[2] / w_i[1])
            ax.plot(x_vals, decision_boundary, 'k--', label="Decision Boundary")

        # Case 2: Multiclass classification (w_i is a matrix)
        else:
            num_classes = w_i.shape[1]
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    # Compute the decision boundary between class i and j
                    w_diff = w_i[:, i] - w_i[:, j]  # Difference between weight vectors
                    decision_boundary = -(w_diff[0] / w_diff[1]) * x_vals - (w_diff[2] / w_diff[1])
                    ax.plot(x_vals, decision_boundary, '--', label=f"Boundary {classes[i]} vs {classes[j]}")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.legend()
        ax.set_title(f"{model_names[idx]} Decision Boundary")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


