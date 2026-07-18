import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

class XGBoostClassifier:
    def __init__(self, random_state=42, **xgb_kwargs):
        """
        Initializes 4 independent XGBoost Classifiers.
        Any extra arguments passed here (like max_depth, learning_rate) 
        will be applied to all 4 models.
        """
        # The 4 target columns we created earlier
        self.dimensions = ['I_E', 'N_S', 'T_F', 'J_P']
        
        # Dictionary storing one model per dimension
        self.models = {
            dim: xgb.XGBClassifier(
                random_state=random_state,
                eval_metric='logloss', 
                **xgb_kwargs
            ) 
            for dim in self.dimensions
        }
        
    def fit(self, X_train, y_train_df):
        """
        Trains the 4 separate models.
        X_train: The feature matrix (e.g., from TF-IDF, Word2Vec, or BERT)
        y_train_df: A pandas DataFrame containing the 4 binary columns
        """
        print("Starting training process...")
        for dim in self.dimensions:
            print(f" -> Training model for {dim}...")
            self.models[dim].fit(X_train, y_train_df[dim])
        print("All 4 models trained successfully! ✅")
        return self
        
    def evaluate(self, X_test, y_test_df):
        """
        Evaluates the models and prints a classification report for each dimension.
        """
        for dim in self.dimensions:
            print(f"\n{'='*50}")
            print(f"Evaluation for dimension: {dim}")
            print(f"{'='*50}")
            y_pred = self.models[dim].predict(X_test)
            
            acc = accuracy_score(y_test_df[dim], y_pred)
            print(f"Accuracy: {acc:.4f}\n")
            print(classification_report(y_test_df[dim], y_pred))

    def predict(self, X):
        """
        Predicts the 4 dimensions and reconstructs the final MBTI string.
        Returns a DataFrame with the binary predictions AND the final string.
        """
        predictions = {}
        for dim in self.dimensions:
            predictions[dim] = self.models[dim].predict(X)
            
        pred_df = pd.DataFrame(predictions)
        
        # Reconstruct the MBTI String based on binary values
        type_strings = []
        for _, row in pred_df.iterrows():
            letter1 = row['I_E'] 
            letter2 = row['N_S']
            letter3 = row['T_F']
            letter4 = row['J_P']
            type_strings.append(f"{letter1}{letter2}{letter3}{letter4}")
            
        pred_df['predicted_type'] = type_strings
        return pred_df
