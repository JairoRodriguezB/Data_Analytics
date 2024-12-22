import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class DataManager:

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Loads the dataset from a CSV file
    # -------------------------------------------------------------------------------------------------------------------------------------
    def load_data(self, folder, file_name):
        """
        Loads the dataset from a CSV file located in the specified folder.

        Parameters:
        - folder: Directory where the dataset file is stored.
        - file_name: Name of the CSV file containing the dataset.
        """

        file_path = os.path.join(folder, file_name)
        self.data = pd.read_csv(file_path)
        print()
        print(f"Data loaded successfully from: {file_path}!")


    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Preprocesses data by removing missing values and splitting the dataset
    # -------------------------------------------------------------------------------------------------------------------------------------
    def preprocess_data(self, outcome, continuous_features, categorical_features, iqr_factor=1.5):
        """
        Performs basic preprocessing on the dataset by removing rows with missing values, 
        selecting outcome and feature columns, and splitting the dataset into training and testing sets.

        Parameters:
        - outcome: Name of the target variable column.
        - continuous_features: List of continuous feature column names.
        - categorical_features: List of categorical feature column names.
        - iqr_factor: Factor for handling outliers (not currently implemented).

        Returns:
        - X_train: Training feature set.
        - X_test: Testing feature set.
        - y_train: Training target variable.
        - y_test: Testing target variable.
        """

        self.data = self.data.dropna(axis=0)

        # Select outcome variable
        y = self.data[outcome]

        # Select features
        X = self.data[continuous_features + categorical_features]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        print(f"Data split into training and testing sets: {X_train.shape}, {X_test.shape}")
        return X_train, X_test, y_train, y_test
    

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Encodes categorical features for training and testing sets
    # -------------------------------------------------------------------------------------------------------------------------------------
    def encode_data(self, X_train, X_test, categorical_features):
        """
        Encodes categorical features using one-hot encoding and combines them with continuous features.

        Parameters:
        - X_train: Training feature set.
        - X_test: Testing feature set.
        - categorical_features: List of categorical feature column names.

        Returns:
        - X_train_encoded: Training feature set with encoded categorical features.
        - X_test_encoded: Testing feature set with encoded categorical features.
        """

        # Select the training and test features that need to be encoded
        X_train_to_encode = X_train[categorical_features]
        X_test_to_encode = X_test[categorical_features]

        X_encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_train_encoded = X_encoder.fit_transform(X_train_to_encode)
        X_test_encoded = X_encoder.transform(X_test_to_encode)

        # Convert encoded arrays to dataframes
        X_train_encoded_df = pd.DataFrame(data=X_train_encoded, columns=X_encoder.get_feature_names_out())
        X_test_encoded_df = pd.DataFrame(data=X_test_encoded, columns=X_encoder.get_feature_names_out())

        # Concatenate encoded features with the continuous features
        X_train_encoded = pd.concat(
            [X_train.drop(columns=categorical_features).reset_index(drop=True), X_train_encoded_df],
            axis=1
        )

        X_test_encoded = pd.concat(
            [X_test.drop(columns=categorical_features).reset_index(drop=True), X_test_encoded_df],
            axis=1
        )

        #print("Features encoded successfully!")
        return X_train_encoded, X_test_encoded


