from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class ModelTrainer:
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Encodes the target variable (y_train and y_test)
    # -------------------------------------------------------------------------------------------------------------------------------------
    def encode_target(self, y_train, y_test):
        """
        Encodes the target variable using one-hot encoding.

        Parameters:
        - y_train: Training target variable.
        - y_test: Testing target variable.

        Returns:
        - y_train_final: Encoded training target variable.
        - y_test_final: Encoded testing target variable.
        - y_encoder: The fitted OneHotEncoder instance.
        """
        y_encoder = OneHotEncoder(drop='first', sparse_output=False)
        y_train_final = y_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_final = y_encoder.transform(y_test.values.reshape(-1, 1)).ravel()
        return y_train_final, y_test_final, y_encoder

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Trains a Logistic Regression model
    # -------------------------------------------------------------------------------------------------------------------------------------
    def train_model(self, X_train, y_train_final):
        """
        Trains a Logistic Regression model using the provided training data.

        Parameters:
        - X_train: Training feature set.
        - y_train_final: Encoded training target variable.

        Returns:
        - log_clf: Trained Logistic Regression model.
        """
        log_clf = LogisticRegression(random_state=0, max_iter=800)
        log_clf.fit(X_train, y_train_final)
        print("Logistic Regression model training completed.")
        return log_clf

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Makes predictions on the test data
    # -------------------------------------------------------------------------------------------------------------------------------------
    def predict(self, log_clf, X_test):
        """
        Makes predictions using the trained Logistic Regression model.

        Parameters:
        - log_clf: Trained Logistic Regression model.
        - X_test: Testing feature set.

        Returns:
        - y_pred: Predicted target variable for the test set.
        """
        y_pred = log_clf.predict(X_test)
        print("Predictions on test data completed.")
        return y_pred




class ModelTrainerRF:
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Encodes the target variable (y_train and y_test)
    # -------------------------------------------------------------------------------------------------------------------------------------
    def encode_target(self, y_train, y_test):
        """
        Encodes the target variable using one-hot encoding.

        Parameters:
        - y_train: Training target variable.
        - y_test: Testing target variable.

        Returns:
        - y_train_final: Encoded training target variable.
        - y_test_final: Encoded testing target variable.
        - y_encoder: The fitted OneHotEncoder instance.
        """
        y_encoder = OneHotEncoder(drop='first', sparse_output=False)
        y_train_final = y_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_final = y_encoder.transform(y_test.values.reshape(-1, 1)).ravel()
        return y_train_final, y_test_final, y_encoder

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Trains a Random Forest model
    # -------------------------------------------------------------------------------------------------------------------------------------
    def train_model(self, X_train, y_train_final):
        """
        Trains a Random Forest model using the provided training data.

        Parameters:
        - X_train: Training feature set.
        - y_train_final: Encoded training target variable.

        Returns:
        - rf_clf: Trained Random Forest model.
        """
        rf_clf = RandomForestClassifier(
            n_estimators=200,              # Number of trees
            max_depth=20,                  # Maximum depth of the trees
            min_samples_split=5,           # Minimum number of samples to split a node
            min_samples_leaf=2,            # Minimum number of samples at a leaf node
            max_features='sqrt',           # Number of features to consider for each split
            random_state=42                     
        )

        # Train the model
        rf_clf.fit(X_train, y_train_final)
        print("Random Forest model training completed with manually set hyperparameters.")
        return rf_clf


    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Makes predictions on the test data
    # -------------------------------------------------------------------------------------------------------------------------------------
    def predict(self, rf_clf, X_test):
        """
        Makes predictions using the trained Random Forest model.

        Parameters:
        - rf_clf: Trained Random Forest model.
        - X_test: Testing feature set.

        Returns:
        - y_pred: Predicted target variable for the test set.
        """
        y_pred = rf_clf.predict(X_test)
        print("Predictions on test data completed.")
        return y_pred

