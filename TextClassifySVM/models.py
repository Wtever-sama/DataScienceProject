'''build model using SMO algorithm'''
from sklearn.svm import SVC

def model(params: dict|None=None)-> SVC:
    svc = SVC(
        C=params.get('C', 1.0),
        kernel=params.get('kernel', 'rbf'),
        gamma=params.get('gamma', 'auto'),
    )
    return svc
