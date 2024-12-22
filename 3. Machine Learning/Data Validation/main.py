import os
import subprocess
import pickle
from data_manager import DataManager
from model_trainer import ModelTrainerRF, ModelTrainer  
from data_balancer import DataBalancer
from visualization import Visualization

def main():
    
    # ========================================================================================================================================
    #                                                   DATA LOADING, PREPROCESSING, AND ENCODING
    # ========================================================================================================================================
    
    folder = os.path.join(os.getcwd(), "Datasets")
    model_choice = "RandomForest"  # Model selection: "RandomForest" or "Logistic"
    balance_method = "SMOTE"  # Set to "SMOTE" to use SMOTE, "ADASYN" to use ADASYN, or None for no balancing
    
    # Dataset
    file_name = "HR_capstone_dataset.csv"
    outcome = "left"
    continuous_features = ["satisfaction_level", "last_evaluation", "number_project", 
                               "average_montly_hours", "time_spend_company"]
    categorical_features = ["Work_accident", "promotion_last_5years", "Department", "salary"]

    # Create an instance of DataManager and load the data
    data_manager = DataManager()
    try:
        data = data_manager.load_data(folder, file_name)
    except FileNotFoundError as e:
        print(e)
        return


    print()
    print("======================= Preprocessing and Balance Data =======================")
    print()
    # Preprocess data
    iqr_factor = 1.5
    X_train, X_test, y_train, y_test = data_manager.preprocess_data(outcome, continuous_features, categorical_features, iqr_factor)

    # Encode the data
    X_train_encoded, X_test_encoded = data_manager.encode_data(X_train, X_test, categorical_features)



    # ========================================================================================================================================
    #                                                      DATA BALANCING AND VALIDATION
    # ========================================================================================================================================
    # Balancing Process: "SMOTE", "ADASYN", or None

    data_balancer = DataBalancer()

    if balance_method in ["SMOTE", "ADASYN"]:
        X_train_balanced, y_train_balanced = data_balancer.balance_data(X_train_encoded, y_train, method=balance_method)
    
        # Separate original and synthesized instances
        num_originals = len(X_train_encoded)
        X_originals = X_train_balanced.iloc[:num_originals]
        X_synthesized = X_train_balanced.iloc[num_originals:]
        y_originals = y_train_balanced[:num_originals]
        y_synthesized = y_train_balanced[num_originals:]

        # Save the data for validation
        with open("data_for_validation.pkl", "wb") as f:
            pickle.dump((X_originals, X_synthesized, y_originals, y_synthesized), f)

        print(f"X_train_encoded (original): {X_originals.shape}")
        print(f"X_train_balanced (synthesized): {X_synthesized.shape}")

        print()
        print("Running tests with pytest...")
        result = subprocess.run(["pytest", "-s", "data_validator.py"], capture_output=True, text=True)

        # Display pytest output in the terminal
        print(result.stdout)

        if result.returncode == 0:
            print("Tests passed. Continuing with model training...")
            print()
        else:
            print("Tests failed. Training will not continue.")
            return  # Stop if tests fail
    else:
        X_train_balanced, y_train_balanced = X_train_encoded, y_train
        # Save the data in a temporary file for pytest to use
        with open("data_for_validation.pkl", "wb") as f:
            pickle.dump((X_train_encoded, X_train_balanced, y_train_balanced), f)


    # ========================================================================================================================================
    #                                                MODEL TRAINING, PREDICTION, AND EVALUATION
    # ========================================================================================================================================

    # Model training and evaluation based on model_choice
    if model_choice == "RandomForest":
        model_trainer = ModelTrainerRF()
        y_train_final, y_test_final, y_encoder = model_trainer.encode_target(y_train_balanced, y_test)
        model = model_trainer.train_model(X_train_balanced, y_train_final)
    elif model_choice == "Logistic":
        model_trainer = ModelTrainer()
        y_train_final, y_test_final, y_encoder = model_trainer.encode_target(y_train_balanced, y_test)
        model = model_trainer.train_model(X_train_balanced, y_train_final)
    else:
        print("Unrecognized model. Please choose 'RandomForest' or 'Logistic'.")
        return 

    # Make predictions
    y_pred = model_trainer.predict(model, X_test_encoded)

    print()

    # Visualization of results
    visualization = Visualization()
    visualization.plot_confusion_matrix(model, X_test_encoded, y_test_final, y_pred)
    visualization.print_classification_report(y_test_final, y_pred)

if __name__ == "__main__":
    main() 
