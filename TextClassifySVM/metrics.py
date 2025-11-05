'''evaluate model performance using various metrics'''
from sklearn import metrics

def evaluate_model(y_true: list, y_pred: list)-> dict:
    '''Evaluate model performance and return metrics'''
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }