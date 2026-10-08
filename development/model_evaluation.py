import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_models(X_test, y_test, trained_models):

    results = []

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred, zero_division=0)*100
        recall = recall_score(y_test,y_pred, zero_division=0)*100
        f1 = f1_score(y_test,y_pred,zero_division=0)*100

        results.append({
        "Model":name,
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall": recall,
        "F1_score":f1 
        })

    results_df = pd.DataFrame(results)


    return results_df




    


