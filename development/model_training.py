from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def train_models(X_train,y_train):

    models = { "logistic_regression":LogisticRegression(),
            "decision_tree" : DecisionTreeClassifier(),
            "random_forest" : RandomForestClassifier(),
            "naive_bayes" : GaussianNB()
            }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train,y_train)

        trained_models[name]=model
        
    return trained_models