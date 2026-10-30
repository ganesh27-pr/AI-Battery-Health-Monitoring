"""Metric helpers."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def regression_metrics(y_true, y_pred):
    return {'MAE': float(mean_absolute_error(y_true, y_pred)), 'MSE': float(mean_squared_error(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'R2': float(r2_score(y_true, y_pred))}

def classification_metrics(y_true, y_pred):
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(y_true,y_pred)), 'precision_macro': float(p), 'recall_macro': float(r),
            'f1_macro': float(f), 'confusion_matrix': confusion_matrix(y_true,y_pred).tolist(),
            'classification_report': classification_report(y_true,y_pred, zero_division=0, output_dict=True)}
