from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from collections import Counter

class DataBalancer:


    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Balances the training dataset using SMOTE or ADASYN
    # -------------------------------------------------------------------------------------------------------------------------------------
    def balance_data(self, X_train, y_train, method):
        """
        Balances the training dataset using the specified method (SMOTE or ADASYN).
        Ensures that the classes in the dataset have equal representation by generating synthetic samples.
        
        Parameters:
        - X_train: Feature set of the training data.
        - y_train: Target variable of the training data.
        - method: Balancing technique to use ("SMOTE" or "ADASYN").

        Returns:
        - X_train_balanced: Balanced feature set.
        - y_train_balanced: Balanced target variable.
        """

        if method == "SMOTE":
            sampler = SMOTE(k_neighbors =5, random_state=42)
            print("Applying SMOTE to balance the data")
        elif method == "ADASYN":
            sampler = ADASYN(n_neighbors=5, random_state=42)
            print("Applying ADASYN to balance the data")
        else:
            raise ValueError("Unrecognized method. Use 'SMOTE' or 'ADASYN'.")

        X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)

        #print(f"Class distribution before balancing ({method}): {Counter(y_train)}")
        #print(f"Class distribution after balancing ({method}): {Counter(y_train_balanced)}")

        return X_train_balanced, y_train_balanced


    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Computes feature ranges (min and max values)
    # -------------------------------------------------------------------------------------------------------------------------------------
    def get_feature_ranges(self, X_train):
        """
        Computes the minimum and maximum values for each feature in the training dataset.
        These ranges can be used for validation or threshold generation in subsequent analyses.
        
        Parameters:
        - X_train: Feature set of the training data.

        Returns:
        - feature_ranges: Dictionary mapping each feature to its (min, max) range.
        """

        feature_ranges = {}

        # Ranges for X_train (features)
        for column in X_train.columns:
            min_val = X_train[column].min()
            max_val = X_train[column].max()
            feature_ranges[column] = (min_val, max_val)

        return feature_ranges
    
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Generates thresholds for features
    # -------------------------------------------------------------------------------------------------------------------------------------
    def generate_thresholds(self, X_train, percentage):
        """
        Generates thresholds for each feature in the training dataset.
        These thresholds are based on a specified percentage of the feature's range or binary encoding.
        
        Parameters:
        - X_train: Feature set of the training data.
        - percentage: Percentage of the feature's range to use for threshold calculation.

        Returns:
        - thresholds: Dictionary mapping each feature to its calculated threshold.
        """
        thresholds = {}
        for column in X_train.columns:
            unique_values = X_train[column].unique()
            
            if len(unique_values) == 2:  # If the feature is binary
                thresholds[column] = 0.5  
            else:
                col_range = X_train[column].max() - X_train[column].min()
                thresholds[column] = col_range * (percentage / 100)
        
        return thresholds