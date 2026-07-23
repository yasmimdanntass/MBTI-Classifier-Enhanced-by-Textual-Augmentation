from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

class RandomForestModel:

    #Starting with hyperparameter optimization and evaluation
    def __init__(self, random_state=42):
        self.base_model = RandomForestClassifier(random_state=random_state)

        self.param_grid = {
            'n_estimators' : [50, 100],
            'max_depth' : [10, 20]
        }
        self.best_model = None

       
    #Returns a GridSearch object configured for 5-fold Cross-Validation
    def get_optimizer(self):
        return GridSearchCV(
            estimator=self.base_model,
            param_grid=self.param_grid,
            cv=2,
            scoring='f1_macro',
            n_jobs=2
        )
    
    def set_best_model(self, model):
        self.best_model = model

    def predict(self, x_test):
        if not self.best_model:
            raise ValueError("The best model has not been set. Please run Grid Search first!")
        return self.best_model.predict(x_test)
    
    #Evaluates the optimized model and returns standard metrics
    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average='macro',
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average='macro',
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average='macro',
            zero_division=0
        )

        report = classification_report(
            y_test,
            predictions,
            zero_division=0
        )
        return accuracy, precision, recall, f1, report