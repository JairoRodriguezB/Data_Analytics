import pytest
import pickle 
from data_balancer import DataBalancer
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test: Validates feature ranges in synthesized data
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_feature_range_validation():
    """
    Validates that synthesized data points fall within the original feature ranges and generates a validation report.

    Returns:
    - None (assertions are raised on failure).
    """
    # Load the processed data
    with open("data_for_validation.pkl", "rb") as f:
        X_originals, X_synthesized, _, _ = pickle.load(f)

    data_balancer = DataBalancer()

    # Get the ranges of the original features
    feature_ranges = data_balancer.get_feature_ranges(X_originals)

    # Validate that EACH POINT in the synthesized data respects the original ranges
    for column in X_synthesized.columns:
        min_val, max_val = feature_ranges[column]
        
        for value in X_synthesized[column]:
            assert value >= min_val, f"{column} feature has a value {value} below the original minimum of {min_val}"
            assert value <= max_val, f"{column} feature has a value {value} above the original maximum of {max_val}"

    print()
    print("---- Feature range validation test passed successfully ----")
    # Generate Excel report after validation
    generate_excel_report(feature_ranges, X_synthesized)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: Generates an Excel report for feature range validation
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_excel_report(feature_ranges, X_synthesized, filename="feature_range_validation_report.xlsx"):
    """
    Generates a detailed and summary Excel report to validate if synthesized features fall within their original feature ranges.
    
    Parameters:
    - feature_ranges: Dictionary of (min, max) for each feature.
    - X_synthesized: Synthesized feature dataset.
    - filename: Name of the output Excel file.

    Returns:
    - None (saves an Excel file).
    """

    report_data = []
    test_case_id = 1  # Start numbering test cases

    # Collect detailed data for the main sheet
    for column in X_synthesized.columns:
        min_val, max_val = feature_ranges[column]
        for value in X_synthesized[column]:
            report_data.append({
                "Test Case": test_case_id,
                "Feature": column,
                "Value": value,
                "Min Value Allowed": min_val,
                "Max Value Allowed": max_val,
                "Validation Status": "Pass" if min_val <= value <= max_val else "Out of Range"
            })
            test_case_id += 1  # Increment test case ID

    # Convert detailed data to DataFrame
    detailed_report_df = pd.DataFrame(report_data)

    # Collect summary data
    summary_data = []
    for column, (min_val, max_val) in feature_ranges.items():
        summary_data.append({
            "Feature": column,
            "Min Value Allowed": min_val,
            "Max Value Allowed": max_val,
            "Total Test Cases": len(X_synthesized[column])
        })

    # Convert summary data to DataFrame
    summary_report_df = pd.DataFrame(summary_data)

    # Add total test cases to the summary
    total_test_cases = len(detailed_report_df)
    summary_metadata = pd.DataFrame({
        "Metric": ["Total Test Cases"],
        "Value": [total_test_cases]
    })

    # Write both sheets to Excel
    with pd.ExcelWriter(filename) as writer:
        detailed_report_df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_report_df.to_excel(writer, sheet_name="Summary", index=False)
        summary_metadata.to_excel(writer, sheet_name="Summary Metadata", index=False)
        
    print(f"-> Excel report generated successfully: {filename}")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test: Validates proximity of synthesized instances to original data
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_class_proximity_validation(percentage=2):
    """
    Validates that synthesized data points are within a reasonable proximity to the original data based on Manhattan distances.

    Parameters:
    - percentage: Percentage threshold for feature-wise proximity calculation.

    Returns:
    - None (assertions are raised on failure).
    """

    # Load data saved from main.py
    with open("data_for_validation.pkl", "rb") as f:
        X_originals, X_synthesized, y_originals, y_synthesized = pickle.load(f)

    balancer = DataBalancer()
    feature_thresholds = balancer.generate_thresholds(X_originals, percentage)
    total_threshold = sum(feature_thresholds.values())
    #print(f"Feature thresholds: {feature_thresholds}")
    #print(f"Total threshold: {total_threshold}")

    # Group original data by class
    original_data_by_class = {}
    for class_label in np.unique(y_originals):
        original_data_by_class[class_label] = X_originals[y_originals == class_label]

    # Validate proximity for each synthesized instance
    results = []
    for i in range(len(X_synthesized)):
        synth_instance = X_synthesized.iloc[i, :]
        class_label = y_synthesized.iloc[i]
        original_class_instances = original_data_by_class[class_label]

        # Calculate Manhattan distances
        manhattan_distances = pairwise_distances(
            original_class_instances.values,
            synth_instance.values.reshape(1, -1),
            metric='manhattan'
        ).flatten()

        min_distance = manhattan_distances.min()
        is_valid = min_distance < total_threshold

        # Append results for the report
        results.append({
            "Synthesized Instance": i,
            "Class Label": class_label,
            "Min Manhattan Distance": min_distance,
            "Threshold Allowed": total_threshold,
            "Validation Status": "Pass" if is_valid else "Fail"
        })

    # Generate Excel report regardless of test results
    generate_excel_report_for_proximity(results)
    
    # Assert all instances passed
    assert all(result["Validation Status"] == "Pass" for result in results), "Some synthesized instances failed the proximity validation."
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------    



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------    
# Function: Generates an Excel report for proximity validation results
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------    
def generate_excel_report_for_proximity(results, filename="class_proximity_validation_report.xlsx"):
    """
    Generates an Excel report summarizing the results of class proximity validation.

    Parameters:
    - results: List of proximity validation results.
    - filename: Name of the output Excel file.

    Returns:
    - None (saves an Excel file).
    """

    # Convert results to DataFrame
    report_df = pd.DataFrame(results)

    # Write to Excel
    report_df.to_excel(filename, index=False)
    print()
    print("---- Class proximity Validation test passed successfully ----")
    print(f"-> Excel report generated successfully: {filename}")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test: Validates correlation consistency using RMSE
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_correlation_validation_rmse(tolerance=0.1):
    """
    Validates the consistency of correlation matrices between the original and synthesized data using the Root Mean Square Error (RMSE) as a metric.

    Parameters:
    - tolerance: Maximum allowed RMSE value for validation to pass.

    Returns:
    - None (assertions are raised on failure).
    """

    # Load processed data
    with open("data_for_validation.pkl", "rb") as f:
        X_originals, X_synthesized, y_originals, _ = pickle.load(f)

    # Filter original data for class 1 only
    class_label = 1
    X_original_class1 = X_originals[y_originals == class_label]

    # Calculate the correlation matrix for the original class 1 and synthesized datasets
    original_corr = X_original_class1.corr()
    synthesized_corr = X_synthesized.corr()

    # Flatten the correlation matrices for comparison
    original_flat = original_corr.values.flatten()
    synthesized_flat = synthesized_corr.values.flatten()

    # Calculate the RMSE
    mse = np.mean((original_flat - synthesized_flat) ** 2)
    rmse = np.sqrt(mse)
    # print(f"RMSE of correlation matrices (Class 1): {rmse}")

    # Verify that the RMSE is within the tolerance level
    assert rmse <= tolerance, (
        f"RMSE between correlation matrices for Class 1 is too high: {rmse}. "
        f"Allowed tolerance: {tolerance}"
    )
    print()
    print("---- Correlation Validation test passed successfully ----")

    generate_correlation_report(original_corr, synthesized_corr, rmse, tolerance)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: Generates a report for correlation validation results
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_correlation_report(original_corr, synthesized_corr, rmse, tolerance, filename="correlation_validation_report.xlsx"):
    """
    Generates an Excel report comparing the correlation matrices of the original and synthesized data and includes the RMSE summary.

    Parameters:
    - original_corr: Correlation matrix of the original data.
    - synthesized_corr: Correlation matrix of the synthesized data.
    - rmse: Calculated Root Mean Square Error.
    - tolerance: Allowed tolerance for RMSE.
    - filename: Name of the output Excel file.

    Returns:
    - None (saves an Excel file).
    """

    # Convert matrices to DataFrames
    original_corr_df = pd.DataFrame(original_corr, columns=original_corr.columns, index=original_corr.index)
    synthesized_corr_df = pd.DataFrame(synthesized_corr, columns=synthesized_corr.columns, index=synthesized_corr.index)

    # Flatten matrices to compare individual correlations
    diff_matrix = (original_corr - synthesized_corr).abs()

    # Create RMSE summary
    rmse_summary = pd.DataFrame({
        "Metric": ["RMSE", "Tolerance", "Test Status"],
        "Value": [rmse, tolerance, "Pass" if rmse <= tolerance else "Fail"]
    })

    # Create significant changes report
    changes = []
    for i in range(len(original_corr)):
        for j in range(len(original_corr)):
            if i != j:  # Avoid diagonal elements
                changes.append({
                    "Feature Pair": f"{original_corr.index[i]}-{original_corr.columns[j]}",
                    "Original Correlation": original_corr.iloc[i, j],
                    "Synthesized Correlation": synthesized_corr.iloc[i, j],
                    "Difference": diff_matrix.iloc[i, j]
                })
    changes_df = pd.DataFrame(changes)

    # Write to Excel
    with pd.ExcelWriter(filename) as writer:
        original_corr_df.to_excel(writer, sheet_name="Original Correlation", index=True)
        synthesized_corr_df.to_excel(writer, sheet_name="Synthesized Correlation", index=True)
        rmse_summary.to_excel(writer, sheet_name="RMSE Summary", index=False)
        changes_df.to_excel(writer, sheet_name="Significant Changes", index=False)

    print(f"-> Excel report generated successfully: {filename}")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test: Validates statistical distribution consistency
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_statistical_distribution(tolerance=0.5):
    """
    Validates that the statistical distribution (mean and variance) of the synthesized data is consistent with the original data within a specified tolerance.

    Parameters:
    - tolerance: Maximum allowed percentage difference for mean and variance.

    Returns:
    - None (assertions are raised on failure).
    """ 
    # Load processed data
    with open("data_for_validation.pkl", "rb") as f:
        X_originals, X_synthesized, y_originals, _ = pickle.load(f)

    # Filter original data for class 1 only
    class_label = 1
    X_original_class1 = X_originals[y_originals == class_label]

    # Initialize results
    results = []
    all_features_pass = True  # Track global status

    # Compare mean and variance for each feature
    for column in X_synthesized.columns:
        original_mean = X_original_class1[column].mean()
        synthesized_mean = X_synthesized[column].mean()

        original_variance = X_original_class1[column].var()
        synthesized_variance = X_synthesized[column].var()

        # Calculate relative differences
        mean_diff = abs(original_mean - synthesized_mean) / abs(original_mean) if original_mean != 0 else 0
        variance_diff = abs(original_variance - synthesized_variance) / abs(original_variance) if original_variance != 0 else 0

        # Append results for reporting
        mean_status = "Pass" if mean_diff <= tolerance else "Fail"
        variance_status = "Pass" if variance_diff <= tolerance else "Fail"
        results.append({
            "Feature": column,
            "Original Mean": original_mean,
            "Synthesized Mean": synthesized_mean,
            "Mean Difference (%)": mean_diff * 100,
            "Original Variance": original_variance,
            "Synthesized Variance": synthesized_variance,
            "Variance Difference (%)": variance_diff * 100,
            "Mean Status": mean_status,
            "Variance Status": variance_status
        })

        # Check global status
        if mean_status == "Fail" or variance_status == "Fail":
            all_features_pass = False

    # Assert global test status
    assert all_features_pass, "Some features failed the statistical distribution validation."

    print()
    print("---- Statistical Distribution test passed successfully ----")
    # Generate Excel report for results
    generate_statistical_report(results)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: Generates a report for statistical distribution validation
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_statistical_report(results, filename="statistical_distribution_report.xlsx", threshold=20):
    """
    Generates an Excel report summarizing the results of statistical distribution validation.

    Parameters:
    - results: List of validation results for each feature.
    - filename: Name of the output Excel file.
    - threshold: Percentage threshold to highlight significant differences.

    Returns:
    - None (saves an Excel file).
    """

    test_case_data = []
    high_difference_data = [] 

    for result in results:
        test_case_data.append({
            "Feature": result["Feature"],
            "Property": "Mean",
            "Original Value": result["Original Mean"],
            "Synthesized Value": result["Synthesized Mean"],
            "Difference (%)": result["Mean Difference (%)"],
            "Status": result["Mean Status"]
        })

        if result["Mean Difference (%)"] > threshold:
            high_difference_data.append({
                "Feature": result["Feature"],
                "Property": "Mean",
                "Difference (%)": result["Mean Difference (%)"]
            })

        # Add a row for the Variance test case
        test_case_data.append({
            "Feature": result["Feature"],
            "Property": "Variance",
            "Original Value": result["Original Variance"],
            "Synthesized Value": result["Synthesized Variance"],
            "Difference (%)": result["Variance Difference (%)"],
            "Status": result["Variance Status"]
        })

        if result["Variance Difference (%)"] > threshold:
            high_difference_data.append({
                "Feature": result["Feature"],
                "Property": "Variance",
                "Difference (%)": result["Variance Difference (%)"]
            })

    test_case_df = pd.DataFrame(test_case_data)

    high_difference_df = pd.DataFrame(high_difference_data)

    # Write to Excel
    with pd.ExcelWriter(filename) as writer:
        test_case_df.to_excel(writer, sheet_name="Statistical Test Cases", index=False)
        if not high_difference_df.empty:
            high_difference_df.to_excel(writer, sheet_name="High Differences", index=False)

    print(f"-> Excel report generated successfully: {filename}")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------




