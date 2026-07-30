import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class PreFittedVotingClassifier:
   
    def __init__(self, estimators, voting='soft'):
        self.estimators = estimators
        self.voting = voting
        
    def fit(self, X, y):
        # Base estimators are already fitted.
        return self
        
    def predict(self, X):
        if self.voting == 'soft':
            # Soft voting: average predicted probabilities
            probas = np.array([clf.predict_proba(X) for name, clf in self.estimators])
            avg_proba = np.mean(probas, axis=0)
            return np.argmax(avg_proba, axis=1)
        else:
            # Hard voting: majority vote
            preds = np.array([clf.predict(X) for name, clf in self.estimators])
            return stats.mode(preds, axis=0)[0].flatten()

class MBTIEnsembleClassifier:
    def __init__(self, models_dict, ensemble_type='voting', **kwargs):
        """
        models_dict: dict of {name: wrapper_instance}
                     e.g. {'lr': mbti_clf_lr, 'xgb': mbti_clf_xgb}
        ensemble_type: 'voting', 'stacking', or 'blending'
        """
        self.dimensions = ['I_E', 'N_S', 'T_F', 'J_P']
        self.models = {}
        self.ensemble_type = ensemble_type
        
        for dim in self.dimensions:
            estimators = [(name, clf.models[dim]) for name, clf in models_dict.items()]
            
            if ensemble_type == 'voting':
                # Uses pre-fitted estimators (no retraining needed)
                self.models[dim] = PreFittedVotingClassifier(estimators=estimators, voting='soft')
                
            elif ensemble_type == 'stacking':
                # Standard Stacking retrains base estimators using cross-validation (takes longer)
                self.models[dim] = StackingClassifier(
                    estimators=estimators, 
                    final_estimator=LogisticRegression(),
                    cv=5,
                    n_jobs=-1,
                    **kwargs
                )
                
            elif ensemble_type == 'blending':
                # Blending uses pre-fitted base estimators and only trains the meta-model 
                # on the provided hold-out validation set.
                self.models[dim] = StackingClassifier(
                    estimators=estimators, 
                    final_estimator=LogisticRegression(),
                    cv="prefit", 
                    n_jobs=-1,
                    **kwargs
                )
            else:
                raise ValueError("ensemble_type must be 'voting', 'stacking', or 'blending'")

    def fit(self, X_train, y_train_df):
        print(f"Starting {self.ensemble_type} ensemble training process...")
        for dim in self.dimensions:
            print(f" -> Fitting {self.ensemble_type} model for {dim}...")
            self.models[dim].fit(X_train, y_train_df[dim])
        print("All 4 ensemble dimensions fitted successfully! ✅")
        return self

    def predict(self, X):
        predictions = {}
        for dim in self.dimensions:
            predictions[dim] = self.models[dim].predict(X)
            
        pred_df = pd.DataFrame(predictions)
        
        # Reconstruct MBTI String
        type_strings = []
        for _, row in pred_df.iterrows():
            letter1 = row['I_E'] 
            letter2 = row['N_S']
            letter3 = row['T_F']
            letter4 = row['J_P']
            type_strings.append(f"{letter1}{letter2}{letter3}{letter4}")
            
        pred_df['predicted_type'] = type_strings
        return pred_df

    def evaluate(self, X_test, y_test_df):
        for dim in self.dimensions:
            print(f"\n{'='*50}")
            print(f"Evaluation for dimension: {dim} ({self.ensemble_type.upper()})")
            print(f"{'='*50}")
            y_pred = self.models[dim].predict(X_test)
            
            acc = accuracy_score(y_test_df[dim], y_pred)
            print(f"Accuracy: {acc:.4f}\n")
            print(classification_report(y_test_df[dim], y_pred))
