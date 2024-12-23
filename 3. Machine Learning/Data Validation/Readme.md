# test


**Objective:** The objective of the project is to develop a machine learning pipeline that addresses class imbalance while ensuring data quality and model reliability. This includes a general implementing data preprocessing, synthesizing additional data using techniques like SMOTE and ADASYN, validating the integrity of synthesized data, and improving classification performance through evaluation of models such as Logistic Regression and Random Forest.

**Dataset:** This project utilizes a dataset titled HR_capstone_dataset.csv, which contains self-reported information from employees of a fictional multinational vehicle manufacturing corporation. The dataset comprises 10 columns and 14,999 rows, with each row representing the self-reported data of a unique employee.

**Scenario:** The scenario associated with the dataset involves the fictional company facing a significant challenge with high employee turnover, which impacts both its finances and efforts to maintain a supportive corporate culture. To address this, the dataset includes key information such as job titles, departments, project counts, average monthly hours, and other relevant factors. The response variable, 'left', indicates whether or not an employee has left the company.

**Project Architecture and Module Overview**

This section outlines the modules that constitute the project, detailing their functionality and role within the overall workflow. Each module is designed to address specific tasks such as data preprocessing, class balancing, validation, model training, and performance visualization.

<div style="text-align: justify;">
- **data_balancer.py (Data Balancing and Threshold Generation):** This module manages class imbalance in the dataset using advanced techniques such as SMOTE and ADASYN, generating synthetic samples for minority classes to ensure balanced distributions. Additionally, it provides functions to calculate feature ranges from the original dataset and generate thresholds, for the validation of synthesized data
- **data_manager.py (Data Loading and Preprocessing):** This module handles the initial steps of data management, including loading datasets from CSV files, preprocessing the data, and splitting it into training and testing sets. It also manages tasks like handling missing values and encoding categorical features.
- **data_validator.py (Validation of Synthesized Data):** This module ensures the integrity and reliability of the synthesized data by performing a series of validation tests. It includes tests for feature range compliance, class proximity, correlation consistency, and statistical distribution. Additionally, the module generates detailed Excel reports for each test.
- **main.py (Workflow Coordination):** This module serves as the central hub for coordinating the entire workflow of the project. It manages tasks such as loading and preprocessing data, balancing classes, training models, and evaluating their performance. Additionally, it integrates data validation by running automated tests with pytest to ensure data quality before proceeding to model training.
- **model_trainer.py (Classification Model Training):** This module facilitates the training, prediction, and evaluation of classification models. It supports two types of models: Logistic Regression and Random Forest. The module encodes the target variable, trains the selected model using the provided training data, and generates predictions on the test data.
- **visualization.py (Metrics Visualization):** This module handles the graphical and textual representation of model performance metrics. It includes functions to compute and display the confusion matrix, providing insights into the classification accuracy and errors. Additionally, it generates a detailed classification report, including precision, recall, and F1-score for each class.
</div>
