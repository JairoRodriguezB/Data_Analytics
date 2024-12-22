import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

class Visualization:

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Plots the confusion matrix
    # -------------------------------------------------------------------------------------------------------------------------------------
    def plot_confusion_matrix(self, model, X_test_final, y_test_final, y_pred):
        """
        Computes and displays the confusion matrix for the provided model predictions.

        Parameters:
        - model: Trained model used for predictions.
        - X_test_final: Test feature set.
        - y_test_final: True labels for the test set.
        - y_pred: Predicted labels for the test set.

        Returns:
        - None (displays the confusion matrix plot).
        """
        # Compute confusion matrix
        log_cm = confusion_matrix(y_test_final, y_pred, labels=model.classes_)

        # Display confusion matrix
        log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=model.classes_)
        log_disp.plot()
        plt.show()


    # -------------------------------------------------------------------------------------------------------------------------------------
    # Method: Prints the classification report
    # -------------------------------------------------------------------------------------------------------------------------------------
    def print_classification_report(self, y_test_final, y_pred):
        """
        Prints the classification report including precision, recall, and F1 score.

        Parameters:
        - y_test_final: True labels for the test set.
        - y_pred: Predicted labels for the test set.

        Returns:
        - None (prints the classification report).
        """
        target_labels = ['Predicted would not leave', 'Predicted would leave']
        print(classification_report(y_test_final, y_pred, target_names=target_labels))
