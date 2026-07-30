from sklearn.ensemble import RandomForestClassifier as _RF
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


class RandomForestClassifier:
    def __init__(self, random_state=42, param_grid=None, cv=2, n_jobs=2):
        
        self.dimensions = ['I_E', 'N_S', 'T_F', 'J_P']
        self.param_grid = param_grid or {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = {dim: _RF(random_state=random_state) for dim in self.dimensions}
        self.best_params_ = {}

    def fit(self, X, y_train_df):
        
        print("Starting training process...")
        for dim in self.dimensions:
            print(f" -> Training model for {dim}...")
            opt = GridSearchCV(
                estimator=_RF(random_state=self.random_state),
                param_grid=self.param_grid,
                cv=self.cv,
                scoring='f1_macro',
                n_jobs=self.n_jobs,
            )
            opt.fit(X, y_train_df[dim])
            self.models[dim] = opt.best_estimator_
            self.best_params_[dim] = opt.best_params_
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
            print(classification_report(y_test_df[dim], preds, zero_division=0))
        return predictions

    def predict(self, X):
        return pd.DataFrame({dim: self.models[dim].predict(X) for dim in self.dimensions})
