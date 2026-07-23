import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class SingleLogisticRegression:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, caminho):
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        joblib.dump(self.model, caminho)

class LogisticRegressionClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = {}

    def fit(self, X, y_df):
        print("Starting training process...")
        for col in y_df.columns:
            print(f" -> Training model for {col}...")
            model_instance = SingleLogisticRegression(**self.kwargs)
            model_instance.fit(X, y_df[col])
            self.models[col] = model_instance
        print("All 4 models trained successfully! ✅")
        return self

    def predict(self, X):
        predictions = {}
        for col, model in self.models.items():
            predictions[col] = model.predict(X)
        return pd.DataFrame(predictions)

    def evaluate(self, X_test, y_test_df):
        for col in y_test_df.columns:
            preds = self.models[col].predict(X_test)
            acc = accuracy_score(y_test_df[col], preds)
            print("\n" + "=" * 50)
            print(f"Evaluation for dimension: {col}")
            print("=" * 50)
            print(f"Accuracy: {acc:.4f}\n")
            print(classification_report(y_test_df[col], preds))