from sklearn.metrics import classification_report


def run_classification_report():
    """
    run classification report
    """
    """
    Output is the following:
    
                    precision  recall  f1-score   support

        class 0       0.67      1.00      0.80         2
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.50      0.67         2

    avg / total       0.67      0.60      0.59         5
    """
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 0, 2, 1, 0]
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_true, y_pred, target_names=target_names))


