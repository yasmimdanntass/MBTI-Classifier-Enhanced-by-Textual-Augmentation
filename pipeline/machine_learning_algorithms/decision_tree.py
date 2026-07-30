import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


class DecisionTree:
    def __init__(self, random_state=42, **dt_kwargs):

        self.dimensions = ['I_E', 'N_S', 'T_F', 'J_P']
        self.models = {
            dim: DecisionTreeClassifier(
                criterion='gini',
                random_state=random_state,
                **dt_kwargs
            )
            for dim in self.dimensions
        }

    def fit(self, X_train, y_train_df):
       
        print("Starting training process...")
        for dim in self.dimensions:
            print(f" -> Training model for dimension {dim}...")
            self.models[dim].fit(X_train, y_train_df[dim])
        print("All 4 models trained successfully! ✅")
        return self

    def evaluate(self, X_test, y_test_df):
        
        predictions = self.predict(X_test)
        for dim in self.dimensions:
            print(f"\n{'='*50}")
            print(f"Evaluation for dimension: {dim}")
            print(f"{'='*50}")
            preds = predictions[dim]
            acc = accuracy_score(y_test_df[dim], preds)
            print(f"Accuracy: {acc:.4f}\n")
            print(classification_report(y_test_df[dim], preds))
        return predictions

    def predict(self, X):
        
        predictions = {}
        for dim in self.dimensions:
            predictions[dim] = self.models[dim].predict(X)
        return pd.DataFrame(predictions)

