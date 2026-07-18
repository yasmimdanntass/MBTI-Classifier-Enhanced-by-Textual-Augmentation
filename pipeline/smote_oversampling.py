from imblearn.over_sampling import SMOTE
import pandas as pd

class SMOTEOversampler:
    def __init__(self, random_state=42, **kwargs):
        """
        Initializes the SMOTE oversampler.
        **kwargs allows passing parameters like k_neighbors to SMOTE.
        """
        self.smote = SMOTE(random_state=random_state, **kwargs)
        
    def fit_resample(self, X, y):
        """
        Applies SMOTE to balance the classes.
        
        Important Tip for MBTI:
        Since you have 4 different binary targets, you can't easily SMOTE them all at once.
        The best approach is to pass the ORIGINAL 16-class MBTI 'type' column as 'y'.
        SMOTE will generate new synthetic examples to balance all 16 MBTI types.
        Then, you can split that balanced 'type' column into the 4 binary columns!
        """
        print(f"Original dataset shape: X={X.shape}, y={y.shape}")
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        # Keep it as a Pandas Series if it originally was one
        if isinstance(y, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y.name)
            
        print(f"Resampled dataset shape: X={X_resampled.shape}, y={y_resampled.shape}")
        return X_resampled, y_resampled
