import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

class MultinomialNaiveBayesClassifier:

    def __init__(self):
        self.dimensions = ['I_E', 'N_S', 'T_F', 'J_P']

        self.models = {
            dim: MultinomialNB()
            for dim in self.dimensions
        }

    def fit(self, X_train, y_train_df):

        for dim in self.dimensions:
            self.models[dim].fit(X_train, y_train_df[dim])

        return self

    def predict(self, x_test):

        predictions = {}

        for dim in self.dimensions:
            predictions[dim] = self.models[dim].predict(x_test)

        return pd.DataFrame(predictions)

    def evaluate(self, x_test, y_test):

        results = {}

        predictions = self.predict(x_test)

        for dim in self.dimensions:
            y_pred = predictions[dim]

            accuracy = accuracy_score(
                y_test[dim], 
                y_pred
            )   

            precision = precision_score(
                y_test[dim], 
                y_pred,
                average = "macro",
                zero_division = 0
            )

            recall = recall_score(
                y_test[dim], 
                y_pred,
                average = "macro",
                zero_division = 0
            )

            f1 = f1_score(
                y_test[dim], 
                y_pred,
                average = "macro",
                zero_division = 0
            )

            report = classification_report(
                y_test[dim], 
                y_pred,
                zero_division = 0
            )

            print(f"\n{'=' * 50}")
            print(f"Evaluation for dimension: {dim}")
            print(f"{'=' * 50}")
            print(f"Accuracy: {accuracy:.4f}\n")
            print(report)

            results[dim] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "classification_report": report
            }

        return results